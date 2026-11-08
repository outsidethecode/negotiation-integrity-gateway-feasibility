"""Replay a recorded run: recompute every check outcome and every post-session
inference measurement from the logged events and the trusted mandate record,
and report any mismatch."""
from .gate import Ledger, decide_refusal, YES, NO, REFUSED


def replay_config(model, scenarios, run, uses_gate):
    mismatches = []
    mandates = {ep["epoch"]: ep["mandate"] for ep in scenarios}
    for ep in run["epochs"]:
        if "events" not in ep:
            continue
        r, k = mandates[ep["epoch"]]
        ledger = Ledger(model)
        recorded_windows = {s["session"]: s["worst_window_after"] for s in ep["sessions"]}
        for ev in ep["events"]:
            if ev["type"] == "check":
                p, q = ev["price"], ev["qty"]
                if uses_gate and decide_refusal(model, ledger.consistent, p, q):
                    expect = REFUSED
                else:
                    expect = YES if (p >= r and q <= k) else NO
                if expect != ev["outcome"]:
                    mismatches.append((ep["epoch"], ev["version"], ev["outcome"], expect))
                ledger.record({k2: v for k2, v in ev.items() if k2 != "version"} | {"outcome": expect})
            else:
                ledger.record({k2: v for k2, v in ev.items() if k2 != "version"})
            if ev["type"] == "session_end":
                worst, _ = model.worst_window(ledger.consistent)
                if str(worst) != recorded_windows[ev["session"]]:
                    mismatches.append((ep["epoch"], ev["session"], "window", recorded_windows[ev["session"]], str(worst)))
    return mismatches
