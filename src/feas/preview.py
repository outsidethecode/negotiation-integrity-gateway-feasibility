"""Answerability preview (proposed interface refinement, 9 October 2026).

Information boundary
--------------------
The preview reports whether the gate would ANSWER a check, not whether the
actual mandate would accept the terms. It is computed only from
  * the declared model (domain, priors, windows, threshold), and
  * the consistent set of the pooled history at a stated ledger version.
It holds no reference to the mandate store, does not filter by true values and
does not rank by private surplus. It is advisory: the gate check and the
authorisation check still run before any commitment, and a preview is valid
only for the ledger version it was computed at.
"""
from .gate import decide_refusal


class Preview:
    def __init__(self, model, ledger):
        self._model = model
        self._ledger = ledger  # read only: consistent set and version

    def snapshot(self):
        """Bind a preview to the current history version."""
        return PreviewSnapshot(self._model, list(self._ledger.consistent), self._ledger.version)


class PreviewSnapshot:
    def __init__(self, model, consistent, version):
        self._model, self._consistent, self.version = model, consistent, version
        self._cache = {}

    def answerable(self, price, qty):
        key = (price, qty)
        if key not in self._cache:
            self._cache[key] = not decide_refusal(self._model, self._consistent, price, qty)
        return self._cache[key]

    def table(self, prices, quantities):
        """Answerable price ranges per quantity, for display to an agent."""
        out = {}
        for q in quantities:
            ranges, start, prev = [], None, None
            for p in prices:
                ok = self.answerable(p, q)
                if ok and start is None:
                    start = p
                if not ok and start is not None:
                    ranges.append((start, prev)); start = None
                prev = p
            if start is not None:
                ranges.append((start, prev))
            out[q] = ranges
        return out


def render_table(table):
    lines = []
    for q, ranges in table.items():
        txt = ", ".join(f"{a}–{b}" if a != b else f"{a}" for a, b in ranges) or "none"
        lines.append(f"  {q} units: {txt}")
    return "\n".join(lines)
