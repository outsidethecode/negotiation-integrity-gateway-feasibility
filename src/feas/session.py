"""Session and epoch engine for the base scope.

One seller, one counterparty at a time, a fixed mandate per epoch, at most one
commitment per session, offers expire at session end, simulated settlement.
The pooled history (ledger) persists across all sessions of an epoch.
"""
import math
import random
from .gate import Ledger, Gate, DirectChecker, MandateStore, YES, NO, REFUSED
from .agents import ConventionalSeller, PreviewSeller, CooperativeBuyer, AdaptiveProber
from .preview import Preview


def sample_from_prior(rng, model, prior_name):
    prior = model.priors[prior_name]
    states = model.states
    total = sum(prior[s] for s in states)
    x = rng.randrange(total)
    for s in states:
        x -= prior[s]
        if x < 0:
            return s
    raise RuntimeError("unreachable")


def build_scenarios(cfg, model):
    """Mandates come from the documented generator prior; buyer parameters are
    drawn independently of the mandate. Same scenarios for every configuration."""
    rng = random.Random(cfg["seed"])
    b = cfg["buyers"]
    epochs = []
    for e in range(cfg["epochs"]):
        mandate = sample_from_prior(rng, model, cfg["generator_prior"])
        sessions = []
        for j, kind in enumerate(cfg["session_plan"]):
            if kind == "cooperative":
                sessions.append({"kind": kind,
                                 "v": rng.randint(b["valuation"]["min"], b["valuation"]["max"]),
                                 "d": rng.randint(b["max_quantity"]["min"], b["max_quantity"]["max"]),
                                 "discount": rng.randint(b["initial_discount"]["min"], b["initial_discount"]["max"])})
            else:
                sessions.append({"kind": kind})
        epochs.append({"epoch": e, "mandate": mandate, "canary": f"CANARY-{cfg['seed']}-{e}-{rng.getrandbits(48):012x}",
                       "sessions": sessions})
    return epochs


def zopa_exists(mandate, v, m, d):
    r, k = mandate
    return v >= r and m <= min(d, k)


def run_session(cfg, model, checker, ledger, seller, buyer, sid, uses_gate, hook=None):
    rounds = cfg["round_limit"]
    buyer.start(rounds)
    seller.start(buyer.m, buyer.d)
    stats = {"checks": 0, "yes": 0, "no": 0, "refused": 0, "rounds": 0, "no_action": 0, "stale_preview": 0}
    deal = None
    for t in range(rounds):
        stats["rounds"] = t + 1
        bid = buyer.bid(t, ledger=ledger, seller=seller, model=model, uses_gate=uses_gate)
        ledger.record({"type": "bid", "session": sid, "round": t, "price": bid[0], "qty": bid[1]})
        if hook:
            hook("turn", sid, t, {"bid": bid, "qty_range": (buyer.m, buyer.d)})
        seller.last_preview_version = None
        action = seller.decide(t, bid)
        if action[0] == "pass":
            # An unusable LLM reply: no binding action, no check, history unchanged.
            stats["no_action"] += 1
            ledger.record({"type": "no_action", "session": sid, "round": t, "reason": action[1]})
            continue
        kind = "accept" if action[0] == "accept" else "offer"
        price, qty = action[1]
        # A preview is valid only for the history version it was computed at. Nothing is
        # recorded between the agent's decision and this check, so the versions must match.
        pv = seller.last_preview_version
        outcome, ev = checker.check(sid, "seller", kind, price, qty)
        if pv is not None:
            ev["preview_version"] = pv
            if pv != ev["version"]:
                stats["stale_preview"] += 1
        stats["checks"] += 1
        stats[outcome] += 1
        seller.observe(action, outcome)
        if hook:
            hook("outcome", sid, t, {"action": kind, "price": price, "qty": qty, "outcome": outcome})
        if outcome != YES:
            continue
        if kind == "accept":
            deal = {"price": price, "qty": qty, "how": "seller_accepted_bid", "round": t}
            break
        ledger.record({"type": "offer_open", "session": sid, "round": t, "price": price, "qty": qty})
        if buyer.accepts(t, price, qty):
            # validation only: the offer is open and the terms match; no new check
            ledger.record({"type": "buyer_accepts", "session": sid, "round": t, "price": price, "qty": qty})
            deal = {"price": price, "qty": qty, "how": "buyer_accepted_offer", "round": t}
            break
        if hasattr(seller, "offer_not_accepted"):
            seller.offer_not_accepted()
    if deal:
        ledger.record({"type": "settled", "session": sid, "price": deal["price"], "qty": deal["qty"],
                       "source": "trusted_settlement_simulator"})
    ledger.record({"type": "session_end", "session": sid, "deal": bool(deal)})
    return deal, stats


def run_config(cfg, model, scenarios, config_name, hook_factory=None, uses_gate=None, seller_factory=None):
    """seller_factory(ep, sid, ledger) -> seller, used for LLM configurations. It
    receives the trusted scenario so the run script can arm the outgoing-prompt
    scan; the seller it returns must not hold the mandate or the canary."""
    if uses_gate is None:
        uses_gate = config_name in ("E", "E_preview", "D_dryrun")
    out = {"config": config_name, "epochs": []}
    for ep in scenarios:
        store = MandateStore(*ep["mandate"], canary=ep["canary"])
        ledger = Ledger(model)
        checker = Gate(model, store, ledger) if uses_gate else DirectChecker(model, store, ledger)
        sat = model.satisfiability_report()
        ep_out = {"epoch": ep["epoch"], "satisfiable": sat["satisfiable"], "sessions": []}
        if uses_gate and not sat["satisfiable"]:
            ep_out["skipped"] = "unsatisfiable initial policy; no session started"
            out["epochs"].append(ep_out)
            continue
        for j, sp in enumerate(ep["sessions"]):
            sid = f"e{ep['epoch']}s{j}"
            if seller_factory is not None:
                seller = seller_factory(ep, sid, ledger)
            elif config_name == "E_preview":
                seller = PreviewSeller(cfg["seller_public"])
                seller.set_preview(Preview(model, ledger).snapshot)
            else:
                seller = ConventionalSeller(cfg["seller_public"])
            if sp["kind"] == "cooperative":
                buyer = CooperativeBuyer(sp["v"], sp["d"], sp["discount"], cfg["buyers"]["accept_tolerance"])
            else:
                buyer = AdaptiveProber(cfg["generator_prior"])
            hook = hook_factory(ep, sid, store) if hook_factory else None
            n_before = len(checker.timings)
            deal, stats = run_session(cfg, model, checker, ledger, seller, buyer, sid, uses_gate, hook)
            r, k = store.reveal_for_audit()  # audit oracle only
            worst, worst_prior = model.worst_window(ledger.consistent)
            rec = {"session": sid, "kind": sp["kind"], "deal": deal, **stats,
                   "in_mandate": (deal is None) or (deal["price"] >= r and deal["qty"] <= k),
                   "owner_surplus": (deal["qty"] * (deal["price"] - r)) if deal else 0,
                   "worst_window_after": str(worst), "worst_window_after_float": float(worst),
                   "worst_prior": worst_prior, "exceeds_threshold": worst > model.tau,
                   "gate_seconds": checker.timings[n_before:]}
            if sp["kind"] == "cooperative":
                m = math.ceil(sp["d"] / 2)
                rec.update({"valuation": sp["v"], "qty_range": [m, sp["d"]],
                            "zopa": zopa_exists((r, k), sp["v"], m, sp["d"])})
            ep_out["sessions"].append(rec)
        ep_out["events"] = ledger.events
        out["epochs"].append(ep_out)
    return out

