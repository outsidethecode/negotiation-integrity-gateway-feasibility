"""Deterministic stand-ins used ONLY to test the follow-up harness. They are
not LLMs and their outcomes are not LLM results. Each is a pure function of
the prompt text."""
import re

from .fake_policies import BID, ROUND, parse_preview, schedule_policy

LINE = re.compile(r"^accept_check_answerable: (true|false) \(current bid (\d+) per unit for (\d+) units; "
                  r"history version (\d+)\)$", re.M)


def parse_bid_line(user):
    m = LINE.search(user)
    if not m:
        return None
    return {"answerable": m.group(1) == "true", "bid": (int(m.group(2)), int(m.group(3))), "version": int(m.group(4))}


def line_reading_policy(system, user):
    """Uses the bid line: accept only when the line says true and the bid meets
    a falling target; otherwise behave like schedule_policy (which proposes
    from the preview). Never accepts when the line says false."""
    info = parse_bid_line(user)
    bp, bq = map(int, BID.search(user).groups())
    n = int(ROUND.search(user).group(1))
    target = max(50, 95 - 4 * (n - 1))
    if info and info["answerable"] and bp >= target:
        return ("tool", "accept_bid", {})
    r = schedule_policy(system, user)
    if r[1] == "accept_bid" and info and not info["answerable"]:
        table = parse_preview(user) or {}
        for p in range(target, 49, -1):
            for q in sorted(table, reverse=True):
                if any(a <= p <= b for a, b in table[q]):
                    return ("tool", "propose_offer", {"price": p, "quantity": q})
        return ("tool", "propose_offer", {"price": target, "quantity": bq})
    return r


def always_accept_policy(system, user):
    """Ignores the bid line and always calls accept_bid (to show the harness
    neither suppresses nor substitutes the model's action)."""
    return ("tool", "accept_bid", {})
