"""Trusted components: mandate store, pooled-history ledger, disclosure gate
(configurations D and E) and direct checker (configuration C).

Only MandateStore holds the mandate. Gate.decide_refusal never receives it.
"""
import time
from dataclasses import dataclass, field

YES, NO, REFUSED = "yes", "no", "refused"


class MandateStore:
    """Trusted store. The canary is a unique marker used to detect leakage
    into LLM inputs; it is never meant to leave this object."""

    def __init__(self, floor, limit, canary):
        self._floor, self._limit, self.canary = floor, limit, canary

    def within(self, price, qty):
        return price >= self._floor and qty <= self._limit

    def reveal_for_audit(self):
        """Test oracle only: used by the audit and replay, never by agents."""
        return (self._floor, self._limit)


@dataclass
class Ledger:
    """Pooled adversarial history for one mandate epoch."""
    model: object
    events: list = field(default_factory=list)
    version: int = 0
    consistent: list = None

    def __post_init__(self):
        if self.consistent is None:
            self.consistent = list(self.model.states)

    def record(self, event):
        event = dict(event, version=self.version)
        self.events.append(event)
        if event.get("outcome") in (YES, NO):
            p, q, a = event["price"], event["qty"], event["outcome"] == YES
            self.consistent = [s for s in self.consistent if self.model.within(s, p, q) == a]
        self.version += 1
        return event


def decide_refusal(model, consistent, price, qty):
    """Simulatable refusal rule: depends only on the consistent set (the pooled
    history), the query and the declared model, never on the actual mandate."""
    answers = {model.within(s, price, qty) for s in consistent}
    for a in answers:
        sub = [s for s in consistent if model.within(s, price, qty) == a]
        for name in model.priors:
            if model.max_window(name, sub) > model.tau:
                return True
    return False


class Gate:
    """Disclosure gate. Answers a binding check only if every possible answer
    keeps every declared prior within the threshold."""

    def __init__(self, model, store, ledger):
        self.model, self.store, self.ledger = model, store, ledger
        self.timings = []

    def check(self, session, actor, kind, price, qty):
        t0 = time.perf_counter()
        refuse = decide_refusal(self.model, self.ledger.consistent, price, qty)
        outcome = REFUSED if refuse else (YES if self.store.within(price, qty) else NO)
        self.timings.append(time.perf_counter() - t0)
        ev = self.ledger.record({"type": "check", "session": session, "actor": actor, "kind": kind,
                                 "price": price, "qty": qty, "outcome": outcome})
        return outcome, ev


class DirectChecker:
    """Configuration C: trusted authorisation checker that reads the mandate
    and returns every outcome, with no cumulative disclosure control.
    The ledger still records outcomes so inference can be measured."""

    def __init__(self, model, store, ledger):
        self.model, self.store, self.ledger = model, store, ledger
        self.timings = []

    def check(self, session, actor, kind, price, qty):
        t0 = time.perf_counter()
        outcome = YES if self.store.within(price, qty) else NO
        self.timings.append(time.perf_counter() - t0)
        ev = self.ledger.record({"type": "check", "session": session, "actor": actor, "kind": kind,
                                 "price": price, "qty": qty, "outcome": outcome})
        return outcome, ev
