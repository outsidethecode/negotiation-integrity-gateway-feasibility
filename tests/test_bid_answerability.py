"""Offline tests of the current-bid answerability correction ("Dpb").
NO LLM IS CALLED: deterministic fake backends drive the harness. Passing these
tests is evidence about this implementation on these cases, not a proof, and
says nothing about whether the correction changes a model's behaviour."""
import json
import os
import random
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from feas.model import build_model, load_config  # noqa: E402
from feas.session import build_scenarios  # noqa: E402
from feas.gate import Ledger, Gate, MandateStore, decide_refusal, YES, NO, REFUSED  # noqa: E402
from feas.preview import Preview, PreviewSnapshot  # noqa: E402
from feas import llm_agent as L  # noqa: E402
from feas.llm_run import run_llm_config  # noqa: E402
from feas.bid_answerability import (BidAwareLLMSeller, run_bid_aware_config, bid_line, NOTE_DPB,  # noqa: E402
                                    ADDED_SENTENCE)
from feas.fake_policies import schedule_policy  # noqa: E402
from feas.fake_policies_followup import line_reading_policy, always_accept_policy, parse_bid_line, LINE  # noqa: E402

CFG = load_config(os.path.join(ROOT, "config", os.environ.get("FEAS_POLICY", "base") + ".json"))
OLD_FRESH = os.path.join(ROOT, "results", "llm_plan", "scenarios_fresh_SEALED_TRUSTED_not_for_llm.json")


def sender(policy):
    be = L.FakeBackend(policy)
    return be, L.TrustedSender(be, L.SpendLedger(1e9), 0.10, 0.50, 1024, sleep=lambda s: None)


def old_fresh():
    scen = json.load(open(OLD_FRESH))
    for ep in scen:
        ep["mandate"] = tuple(ep["mandate"])
    return scen


def replay_until(model, events, version):
    led = Ledger(model)
    for ev in events:
        if ev["version"] >= version:
            break
        led.record({k: v for k, v in ev.items() if k not in ("version", "preview_version")})
    return led.consistent


class TestOldInterfacePreserved(unittest.TestCase):
    def test_original_run_files_unchanged(self):
        import run_llm
        self.assertEqual(run_llm.verify_frozen(), [], "a file of the original frozen run changed")

    def test_original_prompts_byte_identical(self):
        frozen = json.load(open(os.path.join(ROOT, "config", "llm_prompts_frozen.json")))
        _, s = sender(schedule_policy)
        for i in ("C", "D", "Dp"):
            self.assertEqual(L.LLMSeller(i, s, 8, CFG["seller_public"]).system, frozen["system_prompts"][i])

    def test_dpb_system_prompt_adds_one_sentence_only(self):
        _, s = sender(schedule_policy)
        dp = L.LLMSeller("Dp", s, 8, CFG["seller_public"]).system
        dpb = BidAwareLLMSeller(s, 8, CFG["seller_public"]).system
        self.assertEqual(dpb, dp.replace(L.NOTES["Dp"], NOTE_DPB))
        self.assertTrue(NOTE_DPB.startswith(L.NOTES["Dp"]))
        for word in ("approved", "safe", "accept now", "permitted to accept"):
            self.assertNotIn(word, ADDED_SENTENCE.lower())
        self.assertIsNone(L.MANDATE_LIKE.search(dpb))


class TestDpbHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model(CFG)
        cls.dev = build_scenarios(CFG, cls.model)[:3]

    def test_user_prompt_is_dp_prompt_plus_one_line(self):
        _, s1 = sender(schedule_policy)
        _, s2 = sender(schedule_policy)
        _, dp_calls = run_llm_config(CFG, self.model, self.dev, "Dp", s1)
        _, dpb_calls = run_bid_aware_config(CFG, self.model, self.dev, s2)
        self.assertEqual(len(dp_calls), len(dpb_calls))
        for a, b in zip(dp_calls, dpb_calls):
            lines_b = b["user"].split("\n")
            self.assertEqual(len(LINE.findall(b["user"])), 1)
            removed = "\n".join(x for x in lines_b if not LINE.match(x))
            self.assertEqual(removed, a["user"], "Dpb differs from Dp other than by the bid line")
            self.assertTrue(LINE.match(lines_b[-2]), "bid line must sit just before the final instruction")

    def test_bound_to_round_bid_and_history_version(self):
        _, s = sender(line_reading_policy)
        run, calls = run_bid_aware_config(CFG, self.model, self.dev, s)
        events = {ep["epoch"]: ep["events"] for ep in run["epochs"]}
        n = 0
        for c in calls:
            ep = int(c["session"][1:].split("s")[0])
            evs = events[ep]
            bid_ev = [e for e in evs if e["type"] == "bid" and e["session"] == c["session"] and e["round"] == c["round"]][0]
            info = parse_bid_line(c["user"])
            self.assertEqual(list(info["bid"]), [bid_ev["price"], bid_ev["qty"]])
            self.assertEqual(info["version"], bid_ev["version"] + 1)  # the bid is recorded, then the snapshot is taken
            self.assertEqual(info["version"], c["bid_line"]["version"])
            self.assertEqual(info["version"], c["preview_version"])
            consistent = replay_until(self.model, evs, info["version"])
            self.assertEqual(info["answerable"], not decide_refusal(self.model, consistent, *info["bid"]))
            act = [e for e in evs if e["type"] == "check" and e["session"] == c["session"]
                   and e.get("preview_version") == info["version"]]
            for e in act:
                self.assertEqual(e["version"], info["version"], "check ran at a different history version")
            n += 1
        self.assertGreater(n, 0)
        for ep in run["epochs"]:
            for s_ in ep["sessions"]:
                self.assertEqual(s_["stale_preview"], 0)

    def test_stale_preview_is_detected_and_visible(self):
        frozen_snaps = {}

        def stale_factory(ledger):
            def provider():
                key = id(ledger)
                if key not in frozen_snaps:  # first snapshot of the epoch is reused for ever: stale
                    frozen_snaps[key] = Preview(self.model, ledger).snapshot()
                return frozen_snaps[key]
            return provider
        _, s = sender(line_reading_policy)
        run, calls = run_bid_aware_config(CFG, self.model, self.dev, s, provider_factory=stale_factory)
        stale = sum(x["stale_preview"] for ep in run["epochs"] for x in ep["sessions"])
        self.assertGreater(stale, 0, "a stale snapshot was not detected")
        versions = {parse_bid_line(c["user"])["version"] for c in calls}
        self.assertLess(len(versions), len(calls), "stale line should show an old version")

    def test_independent_of_actual_mandate_consistent_with_history(self):
        """Rerun whole epochs with alternative actual mandates consistent with the
        epoch's full recorded history, preferring ones that disagree with the actual
        mandate about a bid shown to the model. Every prompt must be identical."""
        powered = 0
        # old fresh epoch 2 (now a diagnostic case) has alternatives that disagree about a bid that was
        # answerable when shown, under both policies; development epoch 0 adds a generic case
        for ep in [old_fresh()[2], self.dev[0]]:
            _, s = sender(line_reading_policy)
            base_run, base_calls = run_bid_aware_config(CFG, self.model, [ep], s)
            led = Ledger(self.model)
            for ev in base_run["epochs"][0]["events"]:
                led.record({k: v for k, v in ev.items() if k not in ("version", "preview_version")})
            actual = tuple(ep["mandate"])
            shown = [(tuple(c["bid_line"]["bid"]), c["bid_line"]["accept_check_answerable"]) for c in base_calls]
            alts = [m for m in led.consistent if m != actual]
            # only a bid that was answerable when shown can reveal a leak through the line
            disagree = [m for m in alts if any(ans and self.model.within(m, *b) != self.model.within(actual, *b)
                                               for b, ans in shown)]
            powered += bool(disagree)
            for alt in disagree[:: max(1, len(disagree) // 4)][:4] + alts[:: max(1, len(alts) // 3)][:3]:
                _, s2 = sender(line_reading_policy)
                _, alt_calls = run_bid_aware_config(CFG, self.model, [dict(ep, mandate=alt)], s2)
                self.assertEqual([c["user"] for c in alt_calls], [c["user"] for c in base_calls])
            excluded = [m for m in self.model.states if m not in set(led.consistent)]
            _, s3 = sender(line_reading_policy)
            _, ctrl = run_bid_aware_config(CFG, self.model, [dict(ep, mandate=excluded[len(excluded) // 2])], s3)
            self.assertNotEqual([c["user"] for c in ctrl], [c["user"] for c in base_calls])
        self.assertGreater(powered, 0, "no epoch had a consistent alternative disagreeing about an answerable bid")


class TestDpbSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model(CFG)

    def histories(self, n=60, seed=0):
        rng = random.Random(seed)
        for i in range(n):
            led = Ledger(self.model)
            mandate = rng.choice(self.model.states)
            gate = Gate(self.model, MandateStore(*mandate, canary="X"), led)
            for _ in range(rng.randint(0, 25)):
                gate.check("t", "seller", "offer", rng.randint(50, 95), rng.randint(1, 10))
            yield rng, led

    def test_answerable_bid_check_can_return_yes_or_no(self):
        """Same history, same bid, same line ('true'); the truthful answer is yes
        for one consistent mandate and no for another."""
        found = 0
        for rng, led in self.histories():
            snap = Preview(self.model, led).snapshot()
            for _ in range(40):
                bid = (rng.randint(50, 95), rng.randint(1, 10))
                if not snap.answerable(*bid):
                    continue
                yes = [m for m in led.consistent if self.model.within(m, *bid)]
                no = [m for m in led.consistent if not self.model.within(m, *bid)]
                if not (yes and no):
                    continue
                outs, lines = [], set()
                for m in (yes[0], no[0]):
                    l2 = Ledger(self.model, consistent=list(led.consistent), version=led.version)
                    gate = Gate(self.model, MandateStore(*m, canary="X"), l2)  # this world's actual mandate is m
                    lines.add(bid_line(Preview(self.model, l2).snapshot(), bid))
                    out, _ = gate.check("t", "seller", "accept", *bid)
                    outs.append(out)
                self.assertEqual(lines, {bid_line(snap, bid)}, "the line depended on the actual mandate")
                self.assertEqual(outs, [YES, NO])
                found += 1
                break
        self.assertGreater(found, 5)

    def test_line_value_same_for_every_consistent_mandate_and_matches_gate(self):
        for rng, led in self.histories(30, seed=1):
            snap = Preview(self.model, led).snapshot()
            for _ in range(20):
                bid = (rng.randint(50, 95), rng.randint(1, 10))
                outs, lines = set(), set()
                sample = led.consistent[:: max(1, len(led.consistent) // 8)]
                sample += [m for m in led.consistent if self.model.within(m, *bid)][:2]
                sample += [m for m in led.consistent if not self.model.within(m, *bid)][:2]
                for m in sample:
                    l2 = Ledger(self.model, consistent=list(led.consistent), version=led.version)
                    gate = Gate(self.model, MandateStore(*m, canary="X"), l2)  # store first: this world's mandate
                    lines.add(bid_line(Preview(self.model, l2).snapshot(), bid))
                    out, _ = gate.check("t", "seller", "accept", *bid)
                    outs.add(out == REFUSED)
                self.assertEqual(len(outs), 1, "refusal depended on the actual mandate")
                self.assertEqual(len(lines), 1, "the bid line depended on the actual mandate")
                self.assertEqual(snap.answerable(*bid), not outs.pop())

    def test_unanswerable_bid_with_answerable_alternative_offers(self):
        found = 0
        for rng, led in self.histories(80, seed=2):
            snap = Preview(self.model, led).snapshot()
            for _ in range(40):
                d = rng.randint(2, 10)
                bid = (rng.randint(50, 95), d)
                if snap.answerable(*bid):
                    continue
                m_ = -(-d // 2)
                alts = [(p, q) for q in range(m_, d + 1) for p in range(50, 96) if snap.answerable(p, q)]
                if not alts:
                    continue
                self.assertIn("accept_check_answerable: false", bid_line(snap, bid))
                mandate = led.consistent[0]
                l2 = Ledger(self.model, consistent=list(led.consistent), version=led.version)
                g = Gate(self.model, MandateStore(*mandate, canary="X"), l2)
                self.assertEqual(g.check("t", "seller", "accept", *bid)[0], REFUSED)
                self.assertNotEqual(g.check("t", "seller", "offer", *alts[0])[0], REFUSED)
                found += 1
                break
        self.assertGreater(found, 5)

    def test_harness_never_suppresses_or_substitutes_the_models_action(self):
        dev = build_scenarios(CFG, self.model)[:2]
        _, s = sender(always_accept_policy)
        run, calls = run_bid_aware_config(CFG, self.model, dev, s)
        checks = [e for ep in run["epochs"] for e in ep["events"] if e["type"] == "check"]
        self.assertEqual(len(checks), len(calls))
        false_lines = 0
        for c, e in zip(calls, checks):
            self.assertEqual(e["kind"], "accept")
            self.assertEqual([e["price"], e["qty"]], c["bid_line"]["bid"])
            if not c["bid_line"]["accept_check_answerable"]:
                false_lines += 1
                self.assertEqual(e["outcome"], REFUSED)
        self.assertGreater(false_lines, 0)

    def test_full_fake_run_deterministic_checks_on_old_fresh_cases(self):
        """The previously examined fresh cases are now development cases for this correction."""
        from feas.replay import replay_config
        scen = old_fresh()
        _, s = sender(line_reading_policy)
        run, calls = run_bid_aware_config(CFG, self.model, scen, s)
        for ep in run["epochs"]:
            for x in ep["sessions"]:
                self.assertTrue(x["in_mandate"])
                self.assertFalse(x["exceeds_threshold"])
                self.assertEqual(x["stale_preview"], 0)
        self.assertEqual(replay_config(self.model, scen, run, uses_gate=True), [])
        for c in calls:
            for ep in scen:
                self.assertNotIn(ep["canary"], c["user"])
            self.assertIsNone(L.MANDATE_LIKE.search(c["user"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
