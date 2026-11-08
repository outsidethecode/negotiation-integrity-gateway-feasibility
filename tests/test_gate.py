"""Exact finite-model checks of the gate (rational arithmetic, no sampling
approximations in the bound itself). Passing these tests is evidence about
this implementation on these cases, not a general proof of correctness."""
import os
import random
import sys
import unittest
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from feas.model import build_model, load_config  # noqa: E402
from feas.gate import Ledger, Gate, MandateStore, decide_refusal, YES, NO, REFUSED  # noqa: E402

CFG = load_config(os.path.join(ROOT, "config", os.environ.get("FEAS_POLICY", "base") + ".json"))


def random_queries(rng, model, n):
    return [(rng.randint(model.floors[0] - 5, model.floors[-1] + 5), rng.randint(1, model.limits[-1] + 1))
            for _ in range(n)]


class TestSatisfiability(unittest.TestCase):
    def test_declared_policy_is_satisfiable(self):
        self.assertTrue(build_model(CFG).satisfiability_report()["satisfiable"])

    def test_low_threshold_detected_as_unsatisfiable(self):
        # uniform prior already gives 5/40 = 1/8 to every 5-level floor window
        self.assertFalse(build_model(CFG, tau=Fraction(1, 10)).satisfiability_report()["satisfiable"])

    def test_concentrated_prior_detected_as_unsatisfiable(self):
        m = build_model(CFG)
        conc = {s: (1000 if 70 <= s[0] <= 72 else 1) for s in m.states}
        m2 = build_model(CFG, priors_override={"uniform": m.priors["uniform"], "concentrated": conc})
        self.assertFalse(m2.satisfiability_report()["satisfiable"])

    def test_full_support_required(self):
        m = build_model(CFG)
        bad = dict(m.priors["uniform"]); bad[m.states[0]] = 0
        with self.assertRaises(ValueError):
            build_model(CFG, priors_override={"uniform": m.priors["uniform"], "bad": bad})


class TestGateProperties(unittest.TestCase):
    TRIALS, STEPS = 60, 40

    def setUp(self):
        self.model = build_model(CFG)

    def _walk(self, seed):
        rng = random.Random(seed)
        mandate = rng.choice(self.model.states)
        store = MandateStore(*mandate, canary="X")
        ledger = Ledger(self.model)
        gate = Gate(self.model, store, ledger)
        return rng, mandate, ledger, gate

    def test_refusal_does_not_depend_on_actual_mandate(self):
        for seed in range(self.TRIALS):
            rng, mandate, ledger, gate = self._walk(seed)
            for p, q in random_queries(rng, self.model, self.STEPS):
                # every remaining consistent mandate must yield the same refusal decision
                decisions = set()
                for s in ledger.consistent[:: max(1, len(ledger.consistent) // 25)] + [mandate]:
                    alt_store = MandateStore(*s, canary="X")
                    alt_gate = Gate(self.model, alt_store, Ledger(self.model, consistent=list(ledger.consistent)))
                    out, _ = alt_gate.check("t", "seller", "offer", p, q)
                    decisions.add(out == REFUSED)
                self.assertEqual(len(decisions), 1, f"seed {seed}: refusal depended on the actual mandate")
                gate.check("t", "seller", "offer", p, q)

    def test_every_released_answer_preserves_bound(self):
        for seed in range(self.TRIALS):
            rng, mandate, ledger, gate = self._walk(1000 + seed)
            for p, q in random_queries(rng, self.model, self.STEPS):
                out, _ = gate.check("t", "seller", "offer", p, q)
                for name in self.model.priors:
                    self.assertLessEqual(self.model.max_window(name, ledger.consistent), self.model.tau)
                self.assertIn(mandate, ledger.consistent)

    def test_refusals_do_not_change_consistent_set(self):
        rng, mandate, ledger, gate = self._walk(7)
        for p, q in random_queries(rng, self.model, 200):
            before = list(ledger.consistent)
            out, _ = gate.check("t", "seller", "offer", p, q)
            if out == REFUSED:
                self.assertEqual(before, ledger.consistent)

    def test_answers_are_truthful(self):
        rng, mandate, ledger, gate = self._walk(11)
        for p, q in random_queries(rng, self.model, 200):
            out, _ = gate.check("t", "seller", "offer", p, q)
            if out != REFUSED:
                self.assertEqual(out == YES, self.model.within(mandate, p, q))


if __name__ == "__main__":
    unittest.main(verbosity=2)
