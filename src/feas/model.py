"""Declared finite model: mandate domain, priors, windows and threshold.

Everything here is public (known to the pooled adversary). The actual
mandate is never stored in this module.
"""
from fractions import Fraction
import json


class Model:
    def __init__(self, floors, limits, priors, floor_width, limit_width, tau, protect=("floor", "limit")):
        self.protect = tuple(protect)
        self.floors = list(floors)
        self.limits = list(limits)
        self.states = [(r, k) for r in self.floors for k in self.limits]
        self.priors = priors  # name -> {state: positive int weight}
        self.floor_width = floor_width
        self.limit_width = limit_width
        self.tau = Fraction(tau)
        for name, prior in priors.items():
            missing = [s for s in self.states if prior.get(s, 0) <= 0]
            if missing:
                raise ValueError(f"prior {name} lacks full support ({len(missing)} states)")

    @staticmethod
    def within(state, price, qty):
        """Base-scope predicate: may `qty` units be committed at `price`?"""
        r, k = state
        return price >= r and qty <= k

    def max_window(self, prior_name, states):
        """Highest posterior probability of any declared window, given the
        consistent set `states` (exact rational arithmetic)."""
        prior = self.priors[prior_name]
        total = 0
        by_floor = {r: 0 for r in self.floors}
        by_limit = {k: 0 for k in self.limits}
        for s in states:
            w = prior[s]
            total += w
            by_floor[s[0]] += w
            by_limit[s[1]] += w
        if total == 0:
            raise ZeroDivisionError("consistent set has zero mass")
        best = 0
        if "floor" in self.protect:
            fl = [by_floor[r] for r in self.floors]
            for i in range(len(fl) - self.floor_width + 1):
                best = max(best, sum(fl[i:i + self.floor_width]))
        if "limit" in self.protect:
            lm = [by_limit[k] for k in self.limits]
            for i in range(len(lm) - self.limit_width + 1):
                best = max(best, sum(lm[i:i + self.limit_width]))
        return Fraction(best, total)

    def worst_window(self, states):
        """Max over priors of max_window; returns (value, prior_name)."""
        vals = [(self.max_window(n, states), n) for n in self.priors]
        return max(vals)

    def satisfies(self, states):
        return all(self.max_window(n, states) <= self.tau for n in self.priors)

    def satisfiability_report(self):
        vals = {n: self.max_window(n, self.states) for n in self.priors}
        return {"satisfiable": all(v <= self.tau for v in vals.values()),
                "initial_max_window": {n: str(v) for n, v in vals.items()},
                "threshold": str(self.tau)}


def build_model(cfg, tau=None, priors_override=None):
    floors = range(cfg["floors"]["min"], cfg["floors"]["max"] + 1)
    limits = range(cfg["limits"]["min"], cfg["limits"]["max"] + 1)
    priors = priors_override or {
        "uniform": {(r, k): 1 for r in floors for k in limits},
        "market": {(r, k): (1 + max(0, 20 - abs(r - 70))) * (15 - k) for r in floors for k in limits},
    }
    return Model(floors, limits, priors, cfg["windows"]["floor_width"], cfg["windows"]["limit_width"],
                 tau if tau is not None else Fraction(cfg["threshold"]),
                 protect=cfg.get("protect", ["floor", "limit"]))


def load_config(path):
    with open(path) as f:
        return json.load(f)
