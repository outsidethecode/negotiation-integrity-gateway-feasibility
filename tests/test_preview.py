"""Information-boundary checks for the answerability preview (revision of
9 October 2026). Passing these tests is evidence about this implementation on
these cases, not a general proof."""
import inspect
import os
import random
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from feas import preview as preview_mod, agents as agents_mod, llm_agent as llm_mod  # noqa: E402
from feas.model import build_model, load_config  # noqa: E402
from feas.gate import Ledger, Gate, MandateStore, REFUSED, YES, NO  # noqa: E402
from feas.preview import Preview, PreviewSnapshot  # noqa: E402
from feas.agents import PreviewSeller  # noqa: E402
from feas.session import build_scenarios, run_config  # noqa: E402

CFG = load_config(os.path.join(ROOT, "config", os.environ.get("FEAS_POLICY", "base") + ".json"))
FORBIDDEN = ("MandateStore", "reveal_for_audit", "_floor", "_limit", "canary", "store")


def random_history(model, seed, steps=25):
    """A gate-mediated query walk; returns the ledger, gate and mandate."""
    rng = random.Random(seed)
    mandate = rng.choice(model.states)
    ledger = Ledger(model)
    gate = Gate(model, MandateStore(*mandate, canary="X"), ledger)
    for _ in range(steps):
        gate.check("t", "seller", "offer", rng.randint(45, 95), rng.randint(1, 11))
    return rng, mandate, ledger, gate


def traced_epoch(model, ep, mandate):
    """Run one epoch of E_preview with the given actual mandate; return the
    sequence of (preview version, preview cells the seller consulted, action,
    gate outcome) for every seller decision."""
    trace = []

    def factory(_ep, sid, ledger):
        seller = PreviewSeller(CFG["seller_public"])
        prov = Preview(model, ledger).snapshot

        def recording_provider():
            snap = prov()
            trace.append({"sid": sid, "version": snap.version, "snap": snap})
            return snap
        seller.set_preview(recording_provider)
        orig_decide, orig_observe = seller.decide, seller.observe

        def decide(t, bid):
            a = orig_decide(t, bid)
            trace[-1]["action"] = a
            return a

        def observe(action, outcome):
            trace[-1]["outcome"] = outcome
            orig_observe(action, outcome)
        seller.decide, seller.observe = decide, observe
        return seller

    run = run_config(CFG, model, [dict(ep, mandate=mandate)], "E_preview", uses_gate=True, seller_factory=factory)
    flat = [(x["sid"], x["version"], tuple(sorted(x["snap"]._cache.items())), x["action"], x.get("outcome"))
            for x in trace]
    return flat, run


class TestPreviewBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model(CFG)
        cls.scen = build_scenarios(CFG, cls.model)

    def test_preview_code_holds_no_mandate_reference(self):
        src = inspect.getsource(preview_mod)
        code = src.split('"""', 2)[2]  # skip the module docstring, which names what is excluded
        for word in FORBIDDEN:
            self.assertIsNone(re.search(rf"\b{word}\b", code), f"preview.py refers to {word}")
        self.assertEqual(list(inspect.signature(Preview.__init__).parameters), ["self", "model", "ledger"])
        self.assertEqual(list(inspect.signature(PreviewSnapshot.__init__).parameters),
                         ["self", "model", "consistent", "version"])

    def test_agent_code_holds_no_mandate_reference(self):
        for obj in (agents_mod.PreviewSeller, agents_mod.ConventionalSeller, llm_mod.LLMSeller):
            src = inspect.getsource(obj)
            for word in ("MandateStore", "reveal_for_audit", "_floor", "_limit", "canary"):
                self.assertIsNone(re.search(rf"\b{word}\b", src), f"{obj.__name__} refers to {word}")

    def test_ledger_and_snapshot_hold_no_store(self):
        _, _, ledger, gate = random_history(self.model, 3)
        snap = Preview(self.model, ledger).snapshot()
        for obj in (ledger, snap):
            for v in vars(obj).values():
                self.assertNotIsInstance(v, MandateStore)
        self.assertIsInstance(gate.store, MandateStore)  # only the gate holds it

    def test_identical_previews_for_every_mandate_consistent_with_the_history(self):
        """End to end: rerun each epoch with alternative actual mandates that are
        consistent with that epoch's full recorded history. Every preview the seller
        consulted, every action and every gate outcome must be identical."""
        for ep in self.scen:
            base_trace, run = traced_epoch(self.model, ep, tuple(ep["mandate"]))
            ledger = Ledger(self.model)
            for ev in run["epochs"][0]["events"]:
                ledger.record({k: v for k, v in ev.items() if k not in ("version", "preview_version")})
            consistent = ledger.consistent
            self.assertIn(tuple(ep["mandate"]), consistent)
            alts = [s for s in consistent if s != tuple(ep["mandate"])]
            alts = alts[:: max(1, len(alts) // 6)][:6]
            self.assertTrue(alts, f"epoch {ep['epoch']}: no alternative consistent mandate to test")
            for alt in alts:
                alt_trace, _ = traced_epoch(self.model, ep, alt)
                self.assertEqual(alt_trace, base_trace, f"epoch {ep['epoch']}: preview depended on the mandate {alt}")
            # negative control: a mandate the history excludes must change some outcome
            excluded = [s for s in self.model.states if s not in set(consistent)]
            if excluded:
                ctrl_trace, _ = traced_epoch(self.model, ep, excluded[len(excluded) // 2])
                self.assertNotEqual(ctrl_trace, base_trace, "negative control did not change the trace")

    def test_same_history_same_preview_and_refusal_but_answers_may_differ(self):
        """At a fixed history, two mandates both consistent with it must get the
        same preview and the same answer-or-refuse decision. For an answerable
        check whose two possible answers are both permitted by the declared
        policy, the truthful answers differ (yes for one, no for the other), and
        everything after that may differ. Shown end to end on E-preview epochs."""
        from feas.gate import decide_refusal
        diverged_later, cases = 0, 0
        for ep in self.scen:
            base_trace, run = traced_epoch(self.model, ep, tuple(ep["mandate"]))
            checks = [ev for ev in run["epochs"][0]["events"] if ev["type"] == "check"]
            led = Ledger(self.model)
            split = None
            for ev in run["epochs"][0]["events"]:
                if ev["type"] == "check" and ev["outcome"] != REFUSED:
                    p, q = ev["price"], ev["qty"]
                    yes = [s for s in led.consistent if self.model.within(s, p, q)]
                    no = [s for s in led.consistent if not self.model.within(s, p, q)]
                    if yes and no:
                        split = (checks.index(ev), p, q, list(led.consistent), yes, no)
                        break
                led.record({k: v for k, v in ev.items() if k not in ("version", "preview_version")})
            if split is None:
                continue
            i, p, q, before, yes, no = split
            # both possible answers are permitted under the declared policy (no policy change)
            self.assertFalse(decide_refusal(self.model, before, p, q))
            for sub in (yes, no):
                for name in self.model.priors:
                    self.assertLessEqual(self.model.max_window(name, sub), self.model.tau)
            actual = tuple(ep["mandate"])
            other = (no if actual in yes else yes)[len(no if actual in yes else yes) // 2]
            alt_trace, _ = traced_epoch(self.model, ep, other)
            cases += 1
            # identical up to and including the decision to answer: previews, actions, versions
            self.assertEqual(alt_trace[:i], base_trace[:i])
            self.assertEqual(alt_trace[i][:4], base_trace[i][:4])
            # the truthful answers differ
            self.assertEqual({base_trace[i][4], alt_trace[i][4]}, {YES, NO})
            # afterwards the histories differ, so later previews or actions may differ
            if alt_trace[i + 1:] != base_trace[i + 1:]:
                diverged_later += 1
        self.assertGreater(cases, 0, "no informative answered check found; test has no power")
        self.assertGreater(diverged_later, 0, "expected later behaviour to diverge in some epoch")

    def test_snapshot_is_bound_to_its_version(self):
        """Behavioural: after the history moves on, an old snapshot still answers
        exactly as the history at its own version dictates, and differs from a
        fresh snapshot where the two histories disagree."""
        from feas.gate import decide_refusal
        grid = [(p, q) for p in range(50, 96, 3) for q in range(1, 11)]
        for seed in range(40):
            rng, mandate, ledger, gate = random_history(self.model, 700 + seed, steps=3)
            snap = Preview(self.model, ledger).snapshot()
            v0, c0 = ledger.version, list(ledger.consistent)
            for _ in range(60):
                gate.check("t", "seller", "offer", rng.randint(45, 95), rng.randint(1, 11))
            then = {t: not decide_refusal(self.model, c0, *t) for t in grid}
            now = {t: not decide_refusal(self.model, ledger.consistent, *t) for t in grid}
            if then == now:
                continue
            self.assertGreater(ledger.version, v0)
            self.assertEqual(snap.version, v0)
            self.assertEqual({t: snap.answerable(*t) for t in grid}, then)
            fresh = Preview(self.model, ledger).snapshot()
            self.assertEqual({t: fresh.answerable(*t) for t in grid}, now)
            return
        self.fail("no history found where answerability changed; test has no power")

    def test_every_check_records_a_current_preview_version(self):
        run = run_config(CFG, self.model, self.scen, "E_preview")
        n = 0
        for ep in run["epochs"]:
            for ev in ep.get("events", []):
                if ev["type"] == "check":
                    self.assertEqual(ev.get("preview_version"), ev["version"])
                    n += 1
            for s in ep["sessions"]:
                self.assertEqual(s["stale_preview"], 0)
        self.assertGreater(n, 0)

    def test_preview_agrees_with_gate_at_the_same_version(self):
        for seed in range(25):
            rng, mandate, ledger, gate = random_history(self.model, 100 + seed)
            for _ in range(20):
                p, q = rng.randint(45, 95), rng.randint(1, 11)
                snap = Preview(self.model, ledger).snapshot()
                out, _ = gate.check("t", "seller", "offer", p, q)
                self.assertEqual(snap.answerable(p, q), out != REFUSED)

    def test_preview_is_advisory_gate_and_authorisation_still_decide(self):
        rng, mandate, ledger, gate = random_history(self.model, 9, steps=0)
        snap = Preview(self.model, ledger).snapshot()
        grid = [(p, q) for p in range(50, 96) for q in range(1, 11)]
        not_ans = [t for t in grid if not snap.answerable(*t)]
        outside = [t for t in grid if snap.answerable(*t) and not self.model.within(mandate, *t)]
        self.assertTrue(not_ans)
        out, _ = gate.check("t", "seller", "offer", *not_ans[0])
        self.assertEqual(out, REFUSED, "terms outside the preview must still be refused by the gate")
        if outside:
            snap2 = Preview(self.model, ledger).snapshot()
            t = next(x for x in outside if snap2.answerable(*x))
            out, _ = gate.check("t", "seller", "offer", *t)
            self.assertEqual(out, NO, "previewed terms outside the mandate must still be rejected")
        # in a session every seller action is checked: checks == rounds - no-action rounds
        run = run_config(CFG, self.model, self.scen, "E_preview")
        for ep in run["epochs"]:
            for s in ep["sessions"]:
                rounds_with_action = s["rounds"] - s["no_action"]
                self.assertEqual(s["checks"], rounds_with_action)

    def test_candidate_rule_follows_documented_order(self):
        self.assertNotIn("import random", inspect.getsource(agents_mod))
        pub = CFG["seller_public"]
        for seed in range(30):
            rng, mandate, ledger, gate = random_history(self.model, 500 + seed, steps=rng_steps(seed))
            seller = PreviewSeller(pub)
            qmax = rng.randint(2, 10)
            seller.start(-(-qmax // 2), qmax)
            seller.set_preview(Preview(self.model, ledger).snapshot)
            for t in range(CFG["round_limit"]):
                bid = (rng.randint(50, 95), qmax)
                snap = Preview(self.model, ledger).snapshot()
                a = max(pub["ask_min"], pub["ask_start"] - pub["ask_step"] * t)
                if bid[0] >= a and snap.answerable(*bid):
                    expect = ("accept", bid)
                else:
                    order = [(p, q) for p in range(a, pub["ask_min"] - 1, -1) for q in range(seller.q, seller.qmin - 1, -1)]
                    first = next((x for x in order if snap.answerable(*x)), None)
                    expect = ("propose", first if first else (a, seller.q))
                got = seller.decide(t, bid)
                self.assertEqual(got, expect)
                self.assertEqual(seller.decide(t, bid), got, "decision not deterministic")
                self.assertEqual(seller.last_preview_version, ledger.version)


def rng_steps(seed):
    return 5 + seed % 20


if __name__ == "__main__":
    unittest.main(verbosity=2)
