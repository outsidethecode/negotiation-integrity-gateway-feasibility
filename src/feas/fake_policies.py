"""Deterministic stand-ins for the model, used ONLY to test the harness.
They are not LLMs and their outcomes are not LLM results. Each is a pure
function of the prompt text, so identical prompts give identical actions."""
import re

BID = re.compile(r"Buyer's current bid: (\d+) per unit for (\d+) units")
ROUND = re.compile(r"Round (\d+) of (\d+)")
ROW = re.compile(r"^  (\d+) units: (.*)$", re.M)


def parse_preview(user):
    if "Checker preview" not in user:
        return None
    table = {}
    for q, txt in ROW.findall(user.split("Checker preview", 1)[1]):
        rng = []
        if txt.strip() != "none":
            for part in txt.split(", "):
                a, _, b = part.partition("–")
                rng.append((int(a), int(b or a)))
        table[int(q)] = rng
    return table


def schedule_policy(system, user):
    """Concede 4 per round from 95; accept a bid at or above the target; with a
    preview, propose the highest answerable price at or below the target (largest
    quantity first)."""
    bp, bq = map(int, BID.search(user).groups())
    n = int(ROUND.search(user).group(1))
    target = max(50, 95 - 4 * (n - 1))
    table = parse_preview(user)
    if table is None:
        if bp >= target:
            return ("tool", "accept_bid", {})
        return ("tool", "propose_offer", {"price": target, "quantity": bq})
    ok = lambda p, q: any(a <= p <= b for a, b in table.get(q, []))
    if bp >= target and ok(bp, bq):
        return ("tool", "accept_bid", {})
    for p in range(target, 49, -1):
        for q in sorted(table, reverse=True):
            if ok(p, q):
                return ("tool", "propose_offer", {"price": p, "quantity": q})
    return ("tool", "propose_offer", {"price": target, "quantity": bq})


INVALID_CYCLE = [
    ("text", "I would offer 80."),
    ("refusal",),
    ("truncated",),
    ("tool", "propose_offer", {"price": 120, "quantity": 2}),
    ("tool", "propose_offer", {"price": "80", "quantity": 2}),
    ("tool", "propose_offer", {"price": True, "quantity": 2}),
    ("tool", "counter_offer", {"price": 80}),
]


def with_invalid_outputs(policy, every=3):
    """Replace every `every`-th reply (by round number) with an unusable one."""
    def wrapped(system, user):
        n = int(ROUND.search(user).group(1))
        if n % every == 0:
            return INVALID_CYCLE[(n + len(user)) % len(INVALID_CYCLE)]
        return policy(system, user)
    return wrapped
