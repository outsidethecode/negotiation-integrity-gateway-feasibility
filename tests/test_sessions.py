"""Session-level checks: no out-of-mandate commitment, deterministic runs and
exact replay of recorded decisions."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from feas.model import build_model, load_config  # noqa: E402
from feas.session import build_scenarios, run_config  # noqa: E402
from feas.replay import replay_config  # noqa: E402

CFG = load_config(os.path.join(ROOT, "config", os.environ.get("FEAS_POLICY", "base") + ".json"))


class TestSessions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model(CFG)
        cls.scen = build_scenarios(CFG, cls.model)
        cls.runs = {c: run_config(CFG, cls.model, cls.scen, c) for c in ("C_script", "E", "E_preview")}

    def test_no_out_of_mandate_commitment(self):
        for c, run in self.runs.items():
            for ep in run["epochs"]:
                for s in ep["sessions"]:
                    self.assertTrue(s["in_mandate"], f"{c} {s['session']} committed outside the mandate")

    def test_gate_configuration_never_exceeds_threshold(self):
        for c in ("E", "E_preview"):
            for ep in self.runs[c]["epochs"]:
                for s in ep["sessions"]:
                    self.assertFalse(s["exceeds_threshold"], f"{c} {s['session']}")

    def test_runs_are_deterministic(self):
        again = run_config(CFG, self.model, self.scen, "E")
        strip = lambda r: [[{k: v for k, v in s.items() if k != "gate_seconds"} for s in ep["sessions"]] for ep in r["epochs"]]
        self.assertEqual(strip(again), strip(self.runs["E"]))

    def test_replay_reproduces_decisions(self):
        for c, run in self.runs.items():
            mismatches = replay_config(self.model, self.scen, run, uses_gate=(c != "C_script"))
            self.assertEqual(mismatches, [], f"{c}: replay mismatches {mismatches[:3]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
