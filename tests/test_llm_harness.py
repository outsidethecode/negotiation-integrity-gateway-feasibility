"""Harness tests for the LLM configurations, using deterministic fake backends.
NO LLM IS CALLED. These tests check the harness (prompt confinement, cost
guard, parsing, records, version binding), not any model's behaviour."""
import io
import json
import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from feas.model import build_model, load_config  # noqa: E402
from feas.session import build_scenarios  # noqa: E402
from feas.gate import Ledger  # noqa: E402
from feas.replay import replay_config  # noqa: E402
from feas import llm_agent as L  # noqa: E402
from feas.llm_run import run_llm_config  # noqa: E402
from feas.fake_policies import schedule_policy, with_invalid_outputs  # noqa: E402

CFG = load_config(os.path.join(ROOT, "config", os.environ.get("FEAS_POLICY", "base") + ".json"))
NS = types.SimpleNamespace


def sender(policy=schedule_policy, cap=5.0, **kw):
    be = L.FakeBackend(policy, **{k: kw.pop(k) for k in ("in_tokens", "out_tokens") if k in kw})
    return be, L.TrustedSender(be, L.SpendLedger(cap), 0.10, 0.50, kw.pop("max_tokens", 1024), sleep=lambda s: None, **kw)


class TestSenderGuards(unittest.TestCase):
    def test_canary_blocks_before_any_call(self):
        be, s = sender()
        s.set_secrets("CANARY-xyz", "(66, 1)")
        with self.assertRaises(L.LeakBlocked):
            s.send("system", "Round 1 of 8. note CANARY-xyz")
        with self.assertRaises(L.LeakBlocked):
            s.send("system", "Round 1 of 8. (66, 1)")
        with self.assertRaises(L.LeakBlocked):
            s.send("system", "Round 1 of 8. The floor is 66.")
        self.assertEqual(be.invocations, 0)

    def test_allowance_checked_before_each_call_including_max_output(self):
        # every call bills its worst case exactly, so the cap admits floor(cap / worst) calls
        be, s = sender(cap=1.0, max_tokens=1024)
        est = s.estimate_input_tokens("sys", "user Round 1 of 8. Buyer's current bid: 70 per unit for 2 units")
        worst = s.worst_case(est)
        be.in_tokens, be.out_tokens = est, 1024
        s.ledger.cap = 3.5 * worst
        user = "user Round 1 of 8. Buyer's current bid: 70 per unit for 2 units"
        for _ in range(3):
            s.send("sys", user)
        with self.assertRaises(L.BudgetExhausted):
            s.send("sys", user)
        self.assertEqual(be.invocations, 3)
        self.assertLessEqual(s.spent_bound, s.ledger.cap)

    def test_errored_attempts_count_against_the_cap(self):
        state = {"n": 0}

        def flaky(system, user):
            state["n"] += 1
            if state["n"] <= 2:
                return ("error", L.RetryableBackendError("overloaded"))
            return ("tool", "accept_bid", {})
        be, s = sender(flaky)
        resp, meta = s.send("sys", "Round 1 of 8. Buyer's current bid: 70 per unit for 2 units")
        self.assertEqual((be.invocations, s.attempts, s.calls, len(meta["attempt_errors"])), (3, 3, 1, 2))
        self.assertAlmostEqual(s.ledger.uncertain, 2 * meta["worst_case_usd"])
        # with no allowance for a retry, the retry is not attempted
        state["n"] = 0
        be2, s2 = sender(flaky)
        w = s2.worst_case(s2.estimate_input_tokens("sys", "u"))
        s2.ledger.cap = 1.5 * w
        with self.assertRaises(L.BudgetExhausted):
            s2.send("sys", "u")
        self.assertEqual(be2.invocations, 1)

    def test_fatal_error_stops_without_retry(self):
        be, s = sender(lambda a, b: ("error", L.FatalBackendError("model not found")))
        with self.assertRaises(L.FatalBackendError):
            s.send("sys", "u")
        self.assertEqual(be.invocations, 1)

    def test_long_context_threshold(self):
        be, s = sender()
        with self.assertRaises(L.BudgetExhausted):
            s.send("sys", "x" * 300_000)
        self.assertEqual(be.invocations, 0)

    def test_estimate_ratchets_up_when_actual_input_exceeds_it(self):
        be, s = sender(in_tokens=50_000)
        u = "Round 1 of 8. Buyer's current bid: 70 per unit for 2 units"
        e0 = s.estimate_input_tokens("sys", u)
        resp, meta = s.send("sys", u)
        self.assertTrue(meta["reservation_shortfall"])
        self.assertEqual(s.shortfalls, 1)
        self.assertGreater(s.estimate_input_tokens("sys", u), e0)
        self.assertGreaterEqual(s.estimate_input_tokens("sys", u), 50_000)

    def test_model_token_count_sizes_the_reservation(self):
        be, s = sender()
        u = "Round 1 of 8. Buyer's current bid: 70 per unit for 2 units"
        est = s.estimate_input_tokens("sys", u)
        be.count_input_tokens = lambda system, user, tools: 10 * est  # model counts far more than the estimate
        resp, meta = s.send("sys", u)
        self.assertEqual(meta["counted_input_tokens"], 10 * est)
        self.assertGreaterEqual(meta["reserved_input_tokens"], int(10 * est * 1.10))
        be.count_input_tokens = lambda system, user, tools: (_ for _ in ()).throw(RuntimeError("count down"))
        resp, meta = s.send("sys", u)
        self.assertEqual((meta["counted_input_tokens"], meta["count_error"]), (None, "RuntimeError"))
        self.assertEqual(meta["reserved_input_tokens"], est)

    def test_spend_record_survives_restart_and_keeps_open_reservations(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "spend.jsonl")
        led = L.SpendLedger(5.0, path)
        be = L.FakeBackend(schedule_policy)
        s = L.TrustedSender(be, led, 0.10, 0.50, 1024, sleep=lambda x: None)
        u = "Round 1 of 8. Buyer's current bid: 70 per unit for 2 units"
        for _ in range(3):
            s.send("sys", u)
        billed = led.billed
        rid = led.reserve(0.25)  # simulate a process killed mid-request
        led._fh.close()
        led2 = L.SpendLedger(5.0, path)
        self.assertAlmostEqual(led2.billed, billed)
        self.assertAlmostEqual(led2.spent_bound, billed + 0.25)
        self.assertNotIn(rid, led2.open)
        self.assertAlmostEqual(led2.uncertain, 0.25)
        with self.assertRaises(RuntimeError):
            L.SpendLedger(5.0, path)  # locked by led2
        led2._fh.close()
        with self.assertRaises(RuntimeError):
            L.SpendLedger(10.0, path)  # a different cap is refused
        led3 = L.SpendLedger(5.0, path)
        with self.assertRaises(L.BudgetExhausted):
            led3.reserve(5.0)
        led3._fh.close()

    def test_interrupted_attempt_stays_charged(self):
        def boom(system, user):
            raise KeyboardInterrupt
        be, s = sender(boom)
        with self.assertRaises(KeyboardInterrupt):
            s.send("sys", "u")
        self.assertGreater(s.ledger.uncertain, 0)
        self.assertEqual(s.ledger.open, {})


class TestParsing(unittest.TestCase):
    def resp(self, content, stop="tool_use"):
        return NS(content=content, stop_reason=stop)

    def tool(self, name, inp):
        return NS(type="tool_use", name=name, input=inp)

    def test_cases(self):
        bid = (70, 3)
        cases = [
            (self.resp([self.tool("propose_offer", {"price": 80, "quantity": 2})]), ("propose", (80, 2))),
            (self.resp([self.tool("accept_bid", {})]), ("accept", bid)),
            (self.resp([self.tool("propose_offer", {"price": 120, "quantity": 2})]), ("pass", "arguments out of range")),
            (self.resp([self.tool("propose_offer", {"price": 80, "quantity": 0})]), ("pass", "arguments out of range")),
            (self.resp([self.tool("propose_offer", {"price": "80", "quantity": 2})]), ("pass", "non-integer arguments")),
            (self.resp([self.tool("propose_offer", {"price": True, "quantity": 2})]), ("pass", "non-integer arguments")),
            (self.resp([self.tool("counter", {})]), ("pass", "unknown tool counter")),
            (self.resp([NS(type="text", text="80")], "end_turn"), ("pass", "no tool call")),
            (self.resp([], "max_tokens"), ("pass", "output limit reached before a tool call")),
            (self.resp([], "refusal"), ("pass", "refusal")),
        ]
        for r, expect in cases:
            got, _ = L.parse_action(r, bid)
            self.assertEqual(got, expect)
            self.assertEqual(L.wellformed(r, bid), expect[0] != "pass")
        got, note = L.parse_action(self.resp([self.tool("accept_bid", {}), self.tool("propose_offer", {"price": 80, "quantity": 2})]), bid)
        self.assertEqual((got, note), (("accept", bid), "multiple tool calls; first used"))


class TestPromptsAndTools(unittest.TestCase):
    def test_configs_differ_only_in_interface_note(self):
        _, s = sender()
        sys_prompts = {i: L.LLMSeller(i, s, 8, CFG["seller_public"]).system for i in ("C", "D", "Dp")}
        for i, p in sys_prompts.items():
            self.assertEqual(p.replace(L.NOTES[i], "<NOTE>"), sys_prompts["C"].replace(L.NOTES["C"], "<NOTE>"))
        names = [t["name"] for t in L.TOOLS]
        self.assertEqual(names, ["propose_offer", "accept_bid"])
        for t in L.TOOLS:
            self.assertTrue(t["strict"])
            self.assertFalse(t["input_schema"]["additionalProperties"])
        for i in sys_prompts.values():
            self.assertIsNone(L.MANDATE_LIKE.search(i))


class FakeAnthropicModule:
    """Minimal stand-in for the SDK module, to check the request shape and the
    error mapping without the SDK or a network."""

    def __init__(self):
        class APIError(Exception): pass
        class APIStatusError(APIError):
            def __init__(self, status, rid="req_x"):
                super().__init__(f"status {status}"); self.status_code, self.request_id = status, rid
        class RateLimitError(APIStatusError): pass
        class InternalServerError(APIStatusError): pass
        class NotFoundError(APIStatusError): pass
        class APIConnectionError(APIError): pass
        self.APIError, self.APIStatusError, self.RateLimitError = APIError, APIStatusError, RateLimitError
        self.InternalServerError, self.NotFoundError, self.APIConnectionError = InternalServerError, NotFoundError, APIConnectionError
        self.created, self.requests, self.raise_next = [], [], None
        mod = self

        class Messages:
            def create(self, **kw):
                mod.requests.append(kw)
                if mod.raise_next:
                    e, mod.raise_next = mod.raise_next, None
                    raise e
                return NS(content=[NS(type="tool_use", name="accept_bid", input={}, id="t")], stop_reason="tool_use",
                          usage=NS(input_tokens=500, output_tokens=40, cache_creation_input_tokens=0,
                                   cache_read_input_tokens=0), model=kw["model"], _request_id="req_fake")

            def count_tokens(self, **kw):
                mod.requests.append(dict(kw, _endpoint="count_tokens"))
                return NS(input_tokens=600)

        class Anthropic:
            def __init__(self, **kw):
                mod.created.append(kw); self.messages = Messages()
        self.Anthropic = Anthropic


class TestAnthropicBackendShape(unittest.TestCase):
    def setUp(self):
        self.fake = FakeAnthropicModule()
        sys.modules["anthropic"] = self.fake
        self.key = "placeholder-test-only-key"
        os.environ["ANTHROPIC_API_KEY"] = self.key

    def tearDown(self):
        sys.modules.pop("anthropic", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_request_shape_and_no_sampling_parameters(self):
        be = L.AnthropicBackend("claude-haiku-5-5", "low", 60, empty_config_dir=tempfile.mkdtemp())
        self.assertEqual(self.fake.created[0]["max_retries"], 0)
        self.assertEqual(self.fake.created[0]["base_url"], "https://api.anthropic.com")
        self.assertEqual(self.fake.created[0]["api_key"], self.key)
        log = io.StringIO()
        s = L.TrustedSender(be, L.SpendLedger(5.0), 0.10, 0.50, 1024, log=log, sleep=lambda x: None)
        s.send("sys", "Round 1 of 8. Buyer's current bid: 70 per unit for 2 units")
        kw = [r for r in self.fake.requests if "max_tokens" in r][0]
        self.assertEqual(set(kw), {"model", "max_tokens", "system", "tools", "tool_choice", "output_config", "messages"})
        self.assertEqual((kw["model"], kw["max_tokens"], kw["tool_choice"], kw["output_config"]),
                         ("claude-haiku-5-5", 1024, {"type": "auto"}, {"effort": "low"}))
        self.assertEqual([m["role"] for m in kw["messages"]], ["user"])
        self.assertNotIn(self.key, log.getvalue())
        self.assertNotIn(self.key, json.dumps(vars(s), default=str))

    def test_error_mapping(self):
        be = L.AnthropicBackend("claude-haiku-5-5", "low", 60, empty_config_dir=tempfile.mkdtemp())
        for exc, expect in ((self.fake.RateLimitError(429), L.RetryableBackendError),
                            (self.fake.InternalServerError(500), L.RetryableBackendError),
                            (self.fake.APIStatusError(529), L.RetryableBackendError),
                            (self.fake.APIConnectionError("down"), L.RetryableBackendError),
                            (self.fake.NotFoundError(404), L.FatalBackendError),
                            (self.fake.APIStatusError(400), L.FatalBackendError)):
            self.fake.raise_next = exc
            with self.assertRaises(expect):
                be.choose("s", "u", L.TOOLS, 10)

    def test_only_the_environment_key_is_used(self):
        os.environ["ANTHROPIC_CUSTOM_HEADERS"] = "x-extra: 1"
        with self.assertRaises(RuntimeError):
            L.AnthropicBackend("claude-haiku-5-5", "low", 60, empty_config_dir=tempfile.mkdtemp())
        os.environ.pop("ANTHROPIC_CUSTOM_HEADERS")
        os.environ.pop("ANTHROPIC_API_KEY")
        with self.assertRaises(RuntimeError):
            L.AnthropicBackend("claude-haiku-5-5", "low", 60, empty_config_dir=tempfile.mkdtemp())
        self.assertEqual(self.fake.created, [])


class TestLLMSessions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model(CFG)
        cls.scen = build_scenarios(CFG, cls.model)[:2]

    def run_iface(self, iface, policy=schedule_policy, scen=None):
        be, s = sender(policy)
        run, calls = run_llm_config(CFG, self.model, scen or self.scen, iface, s)
        return run, calls, be, s

    def test_records_and_deterministic_checks(self):
        for iface in ("C", "D", "Dp"):
            run, calls, be, s = self.run_iface(iface, with_invalid_outputs(schedule_policy))
            sessions = [x for ep in run["epochs"] for x in ep["sessions"]]
            self.assertEqual(be.invocations, len(calls))
            self.assertEqual(sum(x["rounds"] for x in sessions), len(calls))
            self.assertGreater(sum(x["no_action"] for x in sessions), 0)
            for x in sessions:
                self.assertEqual(x["checks"], x["rounds"] - x["no_action"])
                self.assertTrue(x["in_mandate"])
                self.assertEqual(x["stale_preview"], 0)
                if iface != "C":
                    self.assertFalse(x["exceeds_threshold"])
            self.assertEqual(replay_config(self.model, self.scen, run, uses_gate=(iface != "C")), [])
            for c in calls:
                self.assertFalse(c["harness_defect"])
                if iface == "Dp":
                    self.assertIn("Checker preview (history version", c["user"])
                    self.assertIsNotNone(c["preview_version"])
                else:
                    self.assertNotIn("preview", c["user"].lower())
                for ep in self.scen:
                    self.assertNotIn(ep["canary"], c["user"])
            reasons = {c["action"][1] for c in calls if c["action"][0] == "pass"}
            self.assertTrue(reasons, "invalid-output injection did not reach the parser")

    def test_dp_prompts_identical_for_every_mandate_consistent_with_history(self):
        ep = self.scen[0]
        base_run, base_calls, _, _ = self.run_iface("Dp", scen=[ep])
        led = Ledger(self.model)
        for ev in base_run["epochs"][0]["events"]:
            led.record({k: v for k, v in ev.items() if k not in ("version", "preview_version")})
        alts = [m for m in led.consistent if m != tuple(ep["mandate"])]
        self.assertTrue(alts)
        for alt in alts[:: max(1, len(alts) // 5)][:5]:
            _, alt_calls, _, _ = self.run_iface("Dp", scen=[dict(ep, mandate=alt)])
            self.assertEqual([c["user"] for c in alt_calls], [c["user"] for c in base_calls])
        excluded = [m for m in self.model.states if m not in set(led.consistent)]
        _, ctrl, _, _ = self.run_iface("Dp", scen=[dict(ep, mandate=excluded[len(excluded) // 2])])
        self.assertNotEqual([c["user"] for c in ctrl], [c["user"] for c in base_calls])


if __name__ == "__main__":
    unittest.main(verbosity=2)
