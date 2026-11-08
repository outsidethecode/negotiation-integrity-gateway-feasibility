"""Scripted agents. None of these is an LLM.

ConventionalSeller is configuration E's strategy (and, behind the direct
checker, the scripted stand-in used to exercise configuration C's interface).
It never reads the mandate: it learns only through check outcomes.
"""
import math
from .gate import YES, NO, REFUSED, decide_refusal


class ConventionalSeller:
    """Time-dependent concession on a public price schedule (in the style of
    Faratin et al.). Accept-check a bid when it is at least the current ask;
    otherwise propose the current ask. One mandate-dependent check per round."""

    name = "conventional"

    def __init__(self, pub):
        self.ask_start, self.ask_step = pub["ask_start"], pub["ask_step"]
        self.ask_min, self.cap = pub["ask_min"], pub["quantity_cap"]

    def start(self, qmin, qmax):
        self.qmin, self.q = qmin, min(qmax, self.cap)

    def ask(self, t):
        return max(self.ask_min, self.ask_start - self.ask_step * t)

    def decide(self, t, bid):
        a = self.ask(t)
        if bid is not None and bid[0] >= a:
            return ("accept", bid)
        return ("propose", (a, self.q))

    def observe(self, action, outcome):
        if action[0] == "propose" and outcome == NO and self.q > self.qmin:
            self.q -= 1


class CooperativeBuyer:
    """Buyer with private valuation v and quantity range [m, d]; bids rise
    from v - discount towards v; accepts seller offers near its current bid.
    Parameters are drawn independently of the seller's mandate."""

    kind = "cooperative"

    def __init__(self, v, d, discount, tol):
        self.v, self.d, self.m = v, d, math.ceil(d / 2)
        self.discount, self.tol = discount, tol

    def start(self, rounds):
        self.rounds = rounds

    def bid(self, t, ledger=None, seller=None, model=None, uses_gate=False):
        b0 = self.v - self.discount
        inc = self.discount / max(1, self.rounds - 2)
        return (min(self.v, int(round(b0 + inc * t))), self.d)

    def accepts(self, t, price, qty):
        cur = self.bid(t)[0]
        return price <= self.v and price <= cur + self.tol and self.m <= qty <= self.d


class AdaptiveProber:
    """Adversarial counterparty with the pooled view (it colludes with the
    seller's LLM and every other counterparty, as the threat model assumes).
    It knows the generator prior, the gate code and the seller's public
    strategy. Each round it chooses the bid that maximises the expected
    maximum window probability under its posterior, skipping checks it can
    predict the gate would refuse. It never accepts the seller's offers."""

    kind = "prober"

    def __init__(self, generator_prior_name, ask_ceiling=95):
        self.prior_name = generator_prior_name
        self.ask_ceiling = ask_ceiling
        self.m, self.d = 1, 10

    def start(self, rounds):
        self.rounds = rounds

    def _expected_window(self, model, states, price, qty):
        prior = model.priors[self.prior_name]
        tot = sum(prior[s] for s in states)
        exp = 0.0
        for a in (True, False):
            sub = [s for s in states if model.within(s, price, qty) == a]
            if not sub:
                continue
            pa = sum(prior[s] for s in sub) / tot
            exp += pa * float(model.max_window(self.prior_name, sub))
        return exp

    def bid(self, t, ledger, seller, model, uses_gate):
        states = ledger.consistent
        ask = seller.ask(t)
        candidates = [("low", (model.floors[0], self.d), ("propose", ask, seller.q))]
        for p in range(ask, self.ask_ceiling + 1):
            candidates.append(("probe", (p, 1), ("accept", p, 1)))
        for q in range(2, model.limits[-1] + 1):
            candidates.append(("probe", (ask, q), ("accept", ask, q)))
        best, best_val = None, -1.0
        for label, bid, (_, p, q) in candidates:
            if uses_gate and decide_refusal(model, states, p, q):
                continue  # the coalition can predict refusals, which carry no information
            val = self._expected_window(model, states, p, q)
            if val > best_val:
                best, best_val = bid, val
        return best if best is not None else (model.floors[0], self.d)

    def accepts(self, t, price, qty):
        return False


class PreviewSeller(ConventionalSeller):
    """E-preview (revision of 9 Oct 2026). Same public concession schedule as
    ConventionalSeller, plus the answerability preview (see preview.py).

    Candidate rule (documented, deterministic, no randomness): accept-check the
    buyer's bid if it is at or above the current public ask and the preview says
    the gate would answer it; otherwise scan prices from the current public ask
    downwards to the public minimum and, at each price, quantities from the
    current quantity down to the buyer's minimum, and propose the first pair
    the preview marks answerable. Inputs: public ask schedule, buyer's public
    quantity range, preview snapshot. The mandate is never read."""

    name = "preview"

    def set_preview(self, provider):
        self.preview_provider = provider  # returns a PreviewSnapshot bound to a ledger version

    def decide(self, t, bid):
        snap = self.preview_provider()
        self.last_preview_version = snap.version
        a = self.ask(t)
        if bid is not None and bid[0] >= a and snap.answerable(*bid):
            return ("accept", bid)
        for p in range(a, self.ask_min - 1, -1):
            for q in range(self.q, self.qmin - 1, -1):
                if snap.answerable(p, q):
                    return ("propose", (p, q))
        return ("propose", (a, self.q))  # nothing answerable: this check will be refused
