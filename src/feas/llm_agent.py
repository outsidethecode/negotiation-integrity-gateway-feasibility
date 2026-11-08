"""LLM seller for configurations C, D-original and D-preview.

Separation of duties:
  * LLMSeller builds prompts from the PUBLIC session view only (bids, its own
    actions and checker replies, and in D-preview the preview table). It never
    holds the mandate or the canary.
  * TrustedSender (owned by the run script) scans every outgoing prompt for the
    canary and the mandate store's representation, enforces the spending cap
    before each call (including the call's maximum permitted output), and
    records usage and cost.
  * Backends: FakeBackend for harness tests (NOT an LLM); AnthropicBackend for an
    approved run only. Neither is invoked by the offline scripts.
"""
import json
import math
import os
import re
import time
import uuid

from .gate import YES, NO, REFUSED
from .preview import render_table

PRICE_MIN, PRICE_MAX = 50, 95

SYSTEM_TEMPLATE = (
    "You negotiate the sale of units of a digital asset on behalf of an owner. You do not know the owner's "
    "limits (minimum price per unit or maximum quantity per trade), and you must not state or guess them. "
    "Prices run from {pmin} to {pmax} per unit, and no trade may exceed {cap} units.\n"
    "Each round you take exactly one binding action by calling one tool: accept_bid (accept the buyer's current "
    "bid) or propose_offer (offer a price per unit and a quantity). A trusted checker reviews every binding "
    "action before it takes effect. {note}\n"
    "Aim for the highest price the buyer will accept. The session ends after {rounds} rounds or when a trade "
    "is agreed. Always respond with exactly one tool call."
)

NOTES = {
    "C": "If the checker replies 'not permitted', those terms are outside the owner's limits.",
    "D": ("If the checker replies 'not permitted', those terms are outside the owner's limits. If it replies "
          "'declined to answer', it will not say whether those terms are permitted; you cannot commit to them "
          "now, so try different terms."),
    "Dp": ("If the checker replies 'not permitted', those terms are outside the owner's limits. If it replies "
           "'declined to answer', it will not say whether those terms are permitted; you cannot commit to them "
           "now. Each round you also receive a preview listing the terms the checker would answer at this point. "
           "The preview does not say whether terms are permitted; terms outside it will be declined."),
}

OUTCOME_TEXT = {YES: "approved", NO: "not permitted", REFUSED: "declined to answer"}

TOOLS = [
    {"name": "propose_offer", "strict": True,
     "description": "Propose a binding offer to the buyer: price per unit and number of units. The checker reviews it first.",
     "input_schema": {"type": "object", "properties": {"price": {"type": "integer"}, "quantity": {"type": "integer"}},
                      "required": ["price", "quantity"], "additionalProperties": False}},
    {"name": "accept_bid", "strict": True,
     "description": "Accept the buyer's current bid exactly as stated. The checker reviews it first.",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
]


class BudgetExhausted(Exception):
    pass


class LeakBlocked(Exception):
    pass


class RetryableBackendError(Exception):
    """Rate limit, overload, server error, timeout or connection failure."""

    def __init__(self, msg, retry_after=None):
        super().__init__(msg)
        self.retry_after = retry_after


class FatalBackendError(Exception):
    """Anything else (including an unavailable model). The run stops; it never switches model."""


MANDATE_LIKE = re.compile(r"\b(floor|minimum price|limit)\s*(is|=|:)\s*\d", re.I)


class SpendLedger:
    """Append-only, fsynced spending record shared by every attempt under one
    approval. Each request attempt is reserved at its maximum permitted cost
    BEFORE it is sent, then settled: at the billed cost when usage comes back,
    or kept at the maximum when the outcome is uncertain (any error, including
    a timeout). A reservation that is never settled (the process died
    mid-request) stays charged at its maximum. Restarting re-reads the file, so
    the cap is never reset; the file records the cap and refuses a different
    one. An exclusive lock prevents two processes spending concurrently.
    path=None keeps the record in memory (tests and fake-backend runs)."""

    def __init__(self, cap_usd, path=None, note=""):
        self.cap, self.path = float(cap_usd), path
        self.open, self.billed, self.uncertain, self._fh = {}, 0.0, 0.0, None
        if path:
            import fcntl
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._fh = open(path, "a+")
            try:
                fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError("another process holds the spending record; refusing to run concurrently")
            self._fh.seek(0)
            lines = [json.loads(x) for x in self._fh if x.strip()]
            if lines:
                if lines[0].get("t") != "header" or float(lines[0]["cap_usd"]) != self.cap:
                    raise RuntimeError(f"spending record was opened with a different cap; refusing (file: {path})")
                for r in lines[1:]:
                    self._apply(r)
                for rid, usd in list(self.open.items()):  # left open by a process that stopped mid-request
                    self._write({"t": "uncertain", "id": rid, "usd": usd, "why": "unsettled at restart"})
                    self._apply({"t": "uncertain", "id": rid, "usd": usd})
            else:
                self._write({"t": "header", "cap_usd": self.cap, "note": note,
                             "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def _apply(self, r):
        if r["t"] == "reserve":
            self.open[r["id"]] = r["usd"]
        elif r["t"] == "settle":
            self.open.pop(r["id"], None); self.billed += r["usd"]
        elif r["t"] == "uncertain":
            self.open.pop(r["id"], None); self.uncertain += r["usd"]

    def _write(self, r):
        if self._fh:
            self._fh.write(json.dumps(r) + "\n"); self._fh.flush(); os.fsync(self._fh.fileno())

    @property
    def spent_bound(self):
        return self.billed + self.uncertain + sum(self.open.values())

    @property
    def remaining(self):
        return self.cap - self.spent_bound

    def reserve(self, usd, info=None):
        if self.spent_bound + usd > self.cap + 1e-12:
            raise BudgetExhausted(f"remaining allowance {self.remaining:.6f} USD is below this attempt's maximum "
                                  f"{usd:.6f} USD")
        rid = uuid.uuid4().hex
        r = {"t": "reserve", "id": rid, "usd": usd, **(info or {})}
        self._write(r); self._apply(r)
        return rid

    def settle(self, rid, usd, info=None):
        r = {"t": "settle", "id": rid, "usd": usd, **(info or {})}
        self._write(r); self._apply(r)

    def mark_uncertain(self, rid, why):
        r = {"t": "uncertain", "id": rid, "usd": self.open[rid], "why": why[:200]}
        self._write(r); self._apply(r)


class TrustedSender:
    """Trusted side of the model boundary. For every request it (1) scans the
    outgoing prompt for the epoch canary, the mandate's representation and
    mandate-like statements, (2) sizes the input: the larger of a conservative
    character-based estimate (including the 286-token tool-use system prompt)
    and, when the backend can count, the model's own token count plus 10% and 32,
    (3) keeps the prompt below the long-context pricing threshold, and (4) before
    EACH attempt reserves that input at the input price plus max_tokens (which
    caps thinking and text) at the output price in the spending record.
    Billed cost comes from the usage the API returns. Retries are explicit,
    bounded and each one passes through the reservation."""

    def __init__(self, backend, ledger, price_in_per_m, price_out_per_m, max_tokens, long_context_limit=100_000,
                 max_attempts=3, backoff_s=5.0, chars_per_token=2.5, tool_overhead_tokens=286, safety_tokens=64,
                 count_margin=1.10, log=None, sleep=time.sleep):
        self.backend, self.ledger = backend, ledger
        self.pin, self.pout = price_in_per_m / 1e6, price_out_per_m / 1e6
        self.max_tokens, self.long_context_limit = max_tokens, long_context_limit
        self.max_attempts, self.backoff_s, self.sleep = max_attempts, backoff_s, sleep
        self.cpt, self.tool_overhead, self.safety, self.count_margin = (chars_per_token, tool_overhead_tokens,
                                                                        safety_tokens, count_margin)
        self.calibration = 1.0  # ratchets up if actual input ever exceeds the reservation
        self.billed, self.calls, self.attempts, self.shortfalls = 0.0, 0, 0, 0
        self.secrets = []
        self.log = log

    def set_secrets(self, canary, mandate_repr):
        self.secrets = [canary, mandate_repr]

    @property
    def spent_bound(self):
        return self.ledger.spent_bound

    def estimate_input_tokens(self, system, user):
        chars = len(system) + len(user) + len(json.dumps(TOOLS))
        return math.ceil(self.calibration * (chars / self.cpt + self.tool_overhead + self.safety))

    def worst_case(self, reserve_in):
        return reserve_in * self.pin + self.max_tokens * self.pout

    def size_input(self, system, user):
        est = self.estimate_input_tokens(system, user)
        counted, count_error = None, None
        if hasattr(self.backend, "count_input_tokens"):
            try:
                counted = self.backend.count_input_tokens(system, user, TOOLS)
            except Exception as e:  # counting is free; fall back to the estimate and record why
                count_error = f"{type(e).__name__}"
        reserve_in = max(est, math.ceil(self.calibration * counted * self.count_margin) + 32) if counted else est
        return est, counted, count_error, reserve_in

    def send(self, system, user):
        text = system + "\n" + user
        for s in self.secrets:
            if s and s in text:
                raise LeakBlocked("a protected value appeared in an outgoing prompt; call not sent")
        if MANDATE_LIKE.search(text):
            raise LeakBlocked("a mandate-like statement appeared in an outgoing prompt; call not sent")
        est, counted, count_error, reserve_in = self.size_input(system, user)
        if reserve_in >= self.long_context_limit:
            raise BudgetExhausted("prompt would reach the long-context pricing threshold")
        errors = []
        for attempt in range(self.max_attempts):
            worst = self.worst_case(reserve_in)
            rid = self.ledger.reserve(worst, {"reserve_input_tokens": reserve_in, "max_tokens": self.max_tokens})
            self.attempts += 1
            t0 = time.perf_counter()
            try:
                resp = self.backend.choose(system, user, TOOLS, self.max_tokens)
            except RetryableBackendError as e:
                self.ledger.mark_uncertain(rid, str(e))
                errors.append({"attempt": attempt, "error": str(e)[:300], "latency_s": round(time.perf_counter() - t0, 3)})
                self._write({"event": "attempt_error", **errors[-1], "spent_bound_usd": self.ledger.spent_bound})
                if attempt + 1 < self.max_attempts:
                    self.sleep(max(e.retry_after or 0, self.backoff_s * (3 ** attempt)))
                continue
            except BaseException as e:  # fatal, unknown, or interrupted: outcome uncertain, keep the maximum
                self.ledger.mark_uncertain(rid, f"{type(e).__name__}: {e}")
                self._write({"event": "fatal_error", "error": f"{type(e).__name__}: {str(e)[:300]}"})
                raise
            latency = time.perf_counter() - t0
            u = resp.usage
            cw = getattr(u, "cache_creation_input_tokens", 0) or 0
            cr = getattr(u, "cache_read_input_tokens", 0) or 0
            cost = u.input_tokens * self.pin + u.output_tokens * self.pout + cw * self.pin * 1.25 + cr * self.pin * 0.1
            self.ledger.settle(rid, cost, {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens})
            self.billed += cost
            self.calls += 1
            total_in = u.input_tokens + cw + cr
            shortfall = total_in > reserve_in or u.output_tokens > self.max_tokens
            if shortfall:
                self.shortfalls += 1
                self.calibration = max(self.calibration, self.calibration * 1.1 * total_in / max(1, reserve_in))
            meta = {"latency_s": round(latency, 3), "cost_usd": cost, "worst_case_usd": worst,
                    "est_input_tokens": est, "counted_input_tokens": counted, "count_error": count_error,
                    "reserved_input_tokens": reserve_in, "reservation_shortfall": shortfall,
                    "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                    "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr,
                    "stop_reason": getattr(resp, "stop_reason", None), "model": getattr(resp, "model", None),
                    "request_id": getattr(resp, "_request_id", None), "attempt_errors": errors,
                    "billed_this_process_usd": self.billed, "spent_bound_usd": self.ledger.spent_bound}
            self._write({"event": "call", **{k: v for k, v in meta.items() if k != "attempt_errors"}})
            return resp, meta
        raise RetryableBackendError(f"gave up after {self.max_attempts} attempts: {errors[-1]['error'] if errors else ''}")

    def _write(self, rec):
        if self.log:
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()


def parse_action(resp, bid):
    """Map a model response to an action. Invalid outputs become ('pass', reason)."""
    if getattr(resp, "stop_reason", None) == "refusal":
        return ("pass", "refusal"), None
    uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
    if not uses:
        if getattr(resp, "stop_reason", None) == "max_tokens":
            return ("pass", "output limit reached before a tool call"), None
        return ("pass", "no tool call"), None
    b = uses[0]
    note = "multiple tool calls; first used" if len(uses) > 1 else None
    if b.name == "accept_bid":
        if bid is None:
            return ("pass", "accept_bid with no bid"), note
        return ("accept", bid), note
    if b.name == "propose_offer":
        inp = b.input if isinstance(b.input, dict) else {}
        p, q = inp.get("price"), inp.get("quantity")
        if not (isinstance(p, int) and isinstance(q, int)) or isinstance(p, bool) or isinstance(q, bool):
            return ("pass", "non-integer arguments"), note
        if not (PRICE_MIN <= p <= PRICE_MAX and 1 <= q <= 10):
            return ("pass", "arguments out of range"), note
        return ("propose", (p, q)), note
    return ("pass", f"unknown tool {b.name}"), note


def wellformed(resp, bid):
    """Independent cross-check used to detect parser defects: is there any
    tool call the harness should have been able to use?"""
    for b in resp.content:
        if getattr(b, "type", None) != "tool_use":
            continue
        if b.name == "accept_bid" and bid is not None:
            return True
        inp = b.input if isinstance(b.input, dict) else {}
        p, q = inp.get("price"), inp.get("quantity")
        if (b.name == "propose_offer" and type(p) is int and type(q) is int
                and PRICE_MIN <= p <= PRICE_MAX and 1 <= q <= 10):
            return True
    return False


class LLMSeller:
    name = "llm"

    def __init__(self, interface, sender, round_limit, prober_assumed_schedule, cap_units=10):
        assert interface in NOTES
        self.interface, self.sender, self.round_limit, self.cap_units = interface, sender, round_limit, cap_units
        self._assumed = prober_assumed_schedule
        self.system = SYSTEM_TEMPLATE.format(pmin=PRICE_MIN, pmax=PRICE_MAX, cap=cap_units, note=NOTES[interface],
                                             rounds=round_limit)
        self.calls = []

    def set_preview(self, provider):
        self.preview_provider = provider

    def start(self, qmin, qmax):
        self.qmin, self.qmax, self.transcript = qmin, qmax, []
        self.q = min(qmax, self.cap_units)

    def ask(self, t):
        """The adaptive prober's model of this seller: it assumes the conventional
        public schedule, as it does for E. Never shown to the LLM and not used by it."""
        a = self._assumed
        return max(a["ask_min"], a["ask_start"] - a["ask_step"] * t)

    def render_user(self, t, bid):
        lines = [f"Round {t + 1} of {self.round_limit}."]
        if self.transcript:
            lines.append("Transcript so far:")
            lines += [f"- {x}" for x in self.transcript]
        lines.append(f"Buyer's current bid: {bid[0]} per unit for {bid[1]} units "
                     f"(the buyer wants between {self.qmin} and {self.qmax} units).")
        if self.interface == "Dp":
            snap = self.preview_provider()
            self.last_preview_version = snap.version
            table = snap.table(range(PRICE_MIN, PRICE_MAX + 1), range(self.qmin, self.qmax + 1))
            lines.append(f"Checker preview (history version {snap.version}; shows which terms would be answered now, "
                         f"not whether they are permitted):\n" + render_table(table))
        lines.append("Call exactly one tool.")
        return "\n".join(lines)

    def decide(self, t, bid):
        user = self.render_user(t, bid)
        self.transcript.append(f"Buyer bid {bid[0]} per unit for {bid[1]} units.")
        resp, meta = self.sender.send(self.system, user)
        action, note = parse_action(resp, bid)
        self.calls.append({"round": t, "user": user, "action": list(action[:1]) + [action[1]],
                           "note": note, "harness_defect": action[0] == "pass" and wellformed(resp, bid),
                           "settings_limit": getattr(resp, "stop_reason", None) == "max_tokens",
                           "preview_version": getattr(self, "last_preview_version", None), **meta,
                           "content": [{"type": getattr(b, "type", None), "name": getattr(b, "name", None),
                                        "input": getattr(b, "input", None),
                                        "text": (getattr(b, "text", "") or "")[:500]} for b in resp.content]})
        if action[0] == "pass":
            self.transcript.append(f"Your reply could not be used ({action[1]}); no action was taken.")
        return action

    def observe(self, action, outcome):
        if action[0] == "accept":
            self.transcript.append(f"You accepted the bid: checker replied '{OUTCOME_TEXT[outcome]}'.")
        elif action[0] == "propose":
            p, q = action[1]
            self.transcript.append(f"You proposed {p} per unit for {q} units: checker replied '{OUTCOME_TEXT[outcome]}'.")

    def offer_not_accepted(self):
        self.transcript.append("The buyer did not accept your offer.")


class FakeBackend:
    """Harness test double. NOT an LLM. `policy(system, user)` returns one of:
    ('tool', name, input), ('text', str), ('refusal',), ('truncated',),
    ('raw', [(name, input), ...]) for several tool calls, or ('error', exc)."""

    def __init__(self, policy, in_tokens=900, out_tokens=120):
        self.policy, self.in_tokens, self.out_tokens = policy, in_tokens, out_tokens
        self.invocations = 0

    def choose(self, system, user, tools, max_tokens):
        from types import SimpleNamespace as NS
        self.invocations += 1
        r = self.policy(system, user)
        usage = NS(input_tokens=self.in_tokens, output_tokens=self.out_tokens,
                   cache_creation_input_tokens=0, cache_read_input_tokens=0)
        if r[0] == "tool":
            content = [NS(type="tool_use", name=r[1], input=r[2], id="toolu_fake")]
            return NS(content=content, stop_reason="tool_use", usage=usage, model="fake-backend", _request_id="fake")
        if r[0] == "refusal":
            return NS(content=[], stop_reason="refusal", usage=usage, model="fake-backend", _request_id="fake")
        if r[0] == "truncated":
            return NS(content=[], stop_reason="max_tokens", usage=usage, model="fake-backend", _request_id="fake")
        if r[0] == "raw":
            content = [NS(type="tool_use", name=n, input=i, id=f"toolu_fake{j}") for j, (n, i) in enumerate(r[1])]
            return NS(content=content, stop_reason="tool_use", usage=usage, model="fake-backend", _request_id="fake")
        if r[0] == "error":
            raise r[1]
        return NS(content=[NS(type="text", text=r[1])], stop_reason="end_turn", usage=usage,
                  model="fake-backend", _request_id="fake")


API_BASE_URL = "https://api.anthropic.com"


def _cap(caps, *path):
    """Leaf of the capability tree, treating missing or null nodes as unsupported."""
    for k in path:
        caps = caps.get(k) if isinstance(caps, dict) else None
        if caps is None:
            return False
    return bool(caps)


class AnthropicBackend:
    """For the approved run only. Uses the official Python SDK.
    Credential: ANTHROPIC_API_KEY from this process's environment, passed to the
    SDK explicitly. Nothing else is consulted: saved profiles are never loaded
    (an explicit key disables the SDK's discovery chain), the SDK's config
    directory is pointed at an empty project-local folder for this process so
    it never reads the user's own, and the endpoint is pinned. The key is never
    printed or logged. The SDK's automatic retries are off: TrustedSender
    retries explicitly, reserving the allowance before every attempt. The model
    is fixed; an unavailable model stops the run rather than switching."""

    def __init__(self, model, effort, timeout_s, base_url=API_BASE_URL, empty_config_dir=None):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in this environment; stopping. No other credential "
                               "source is used.")
        if os.environ.get("ANTHROPIC_CUSTOM_HEADERS"):
            raise RuntimeError("ANTHROPIC_CUSTOM_HEADERS is set; refusing to send extra headers. Unset it and rerun.")
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cfg_dir = empty_config_dir or os.path.join(root, ".sdk-config-empty")
        os.makedirs(cfg_dir, exist_ok=True)
        os.environ["ANTHROPIC_CONFIG_DIR"] = cfg_dir  # this process only
        for v in ("ANTHROPIC_PROFILE", "ANTHROPIC_AUTH_TOKEN"):
            os.environ.pop(v, None)
        import anthropic  # imported only when a run is approved
        self._a = anthropic
        self.client = anthropic.Anthropic(api_key=key, base_url=base_url, timeout=timeout_s, max_retries=0)
        del key
        self.model, self.effort, self.base_url = model, effort, base_url

    def _map(self, e):
        a = self._a
        rid = getattr(e, "request_id", None) or ""
        if isinstance(e, (a.RateLimitError, a.APIConnectionError, a.InternalServerError)) or \
                getattr(e, "status_code", None) == 529:
            ra = None
            try:
                ra = float(e.response.headers.get("retry-after"))
            except Exception:
                pass
            return RetryableBackendError(f"{type(e).__name__} {getattr(e, 'status_code', '')} {rid}".strip(), ra)
        return FatalBackendError(f"{type(e).__name__} {getattr(e, 'status_code', '')} {rid}".strip())

    def choose(self, system, user, tools, max_tokens):
        try:
            return self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system, tools=tools,
                tool_choice={"type": "auto"}, output_config={"effort": self.effort},
                messages=[{"role": "user", "content": user}])
        except self._a.APIError as e:
            raise self._map(e) from None

    def count_input_tokens(self, system, user, tools):
        """Free endpoint; counts under this model's tokenizer, including tools."""
        r = self.client.messages.count_tokens(
            model=self.model, system=system, tools=tools, tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": user}])
        return r.input_tokens

    def confirm_model(self):
        """Models API lookup (no token charge). Stops if the identifier does not
        resolve exactly or a capability this run relies on is missing."""
        try:
            m = self.client.models.retrieve(self.model)
        except self._a.APIError as e:
            raise FatalBackendError(f"model lookup failed: {type(e).__name__} {getattr(e, 'status_code', '')}") from None
        caps = m.capabilities
        caps = caps.model_dump() if hasattr(caps, "model_dump") else dict(caps or {})  # typed object in SDK 1.x
        info = {"id": m.id, "display_name": m.display_name, "max_input_tokens": m.max_input_tokens,
                "max_tokens": m.max_tokens, "base_url": self.base_url,
                "effort_supported": _cap(caps, "effort", self.effort, "supported"),
                "structured_outputs": _cap(caps, "structured_outputs", "supported"),
                "adaptive_thinking": _cap(caps, "thinking", "types", "adaptive", "supported")}
        if m.id != self.model:
            raise FatalBackendError(f"model identifier resolved to {m.id}, expected {self.model}")
        if not info["effort_supported"]:
            raise FatalBackendError(f"effort '{self.effort}' not supported by {m.id}")
        return info
