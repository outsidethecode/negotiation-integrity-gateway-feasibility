# Negotiation Integrity Gateway — Empirical Feasibility Study

**Status:** Independent, preliminary feasibility study (October 2026). No general security or negotiation advantage is claimed. This public-facing source subset omits the proposal documents, raw experiment logs and sealed scenarios.

## Research question

Can an AI agent negotiating on behalf of an owner produce useful agreements while an external boundary constrains what its responses and commitments reveal about private owner instructions (such as floor price and quantity limit)? What happens to task usefulness when an agent must operate through a policy-limited interface?

This is not a general prompt-injection benchmark and does not establish prevention or detection of coordination among separate businesses.

## Experiment design

A synthetic single-issue negotiation with a finite space of 40 candidate floor prices and 10 quantity limits. Scripted cooperative buyers and an adaptive probing buyer interact with either (a) a conventional LLM seller, (b) the same LLM behind a disclosure-aware gate, (c) a gated LLM with an answerability preview, or (d) scripted baselines. Two declared policies were assessed; floor-only results are a separate diagnostic and must not be mixed with floor-and-quantity results.

The evaluated agent was a single model (`claude-haiku-5-5`) at low effort, with structured tool interactions. The original private research archive preserves the frozen plans, full call logs and detailed reporting; this source subset does not.

## Headline findings (October 2026)

| Comparison | Observed result | Interpretation |
|---|---|---|
| Experiment 1, base policy, 11 feasible cases | Ungated LLM: **9/10/9** agreements across 3 repetitions; gated LLM: **1/1/1**; gated with preview: **4/7/7**; scripted preview baseline: **7**. | The gate maintained the declared inference bound in tested conditions, but task usefulness was impaired. The preview recovered some usefulness, not a general LLM advantage. |
| Experiment 2, base policy, 8 feasible cases | Gated preview LLM **0/0/0**; with answerability indicator **1/0/0**; scripted reference **3**. | The extra indicator reduced one kind of refused action but did not reliably improve agreement. |
| All experiments | **4,452** real model calls across smoke/main phases; archived API-usage-based cost approximately **US$1.3274**. | Recorded experimental work, not a proof of production efficacy. |

The follow-up suggests a particular failure mode: an agent's early checks can consume its future options, leaving legitimate agreements unreachable under the specified disclosure policy. We treat this as an observed phenomenon in a limited testbed and a direction for future research.

## Reproducibility

**Offline reproducibility:** Requires Python 3.13; selected tests use the standard library and fake model backends, without external APIs. Run these commands from the repository root:

```sh
python3 -I tests/test_gate.py
python3 -I tests/test_preview.py
python3 -I tests/test_llm_harness.py
```

The headline numbers below are transcribed from archived private run reports. They are **not** independently reproducible from this public subset because the original sealed prompts, private run logs and trusted scenarios are intentionally excluded. The test suite verifies mechanisms and harness behaviour, not the historical paid LLM experiment results.

The complete private project also includes optional paid-API experiment tooling and frozen scenarios that are deliberately not part of this public-facing package.

## Limitations and negative results

One low-effort model, a small synthetic finite domain, scripted counterparties, short probing interactions and structured tools. No full free-text adversarial evaluation, no production workload, no multi-agent coordination detection, no empirical evidence of commercial demand, and no broad model comparison. Three repetitions on the same cases do not yield independent task samples. A scripted negotiation strategy was competitive with or better than the tested LLM strategies. The answerability indicator's measured reduction in attempted unanswerable acceptances did not translate into improved outcomes.

## Authorship and development process

The project was initiated and directed by Dr Zakwan Jaroucheh as independent exploratory research. AI coding assistants were used during implementation. The author's research contributions include problem framing, evaluation direction and interpretation; the detailed breakdown of code and experimental work will be documented after review. No claim of funding, endorsement or formal validation by ARIA or Anthropic is made.

## Related materials

Separate academic publications and patent applications demonstrate adjacent work in identity verification, cryptographic custody and trust boundaries. They are not evidence that this gateway is effective outside its tested domain.

This curated release omits proposal submissions, budgets, internal notes, raw trusted scenarios, third-party sources and Git history.

## Licensing

No open-source licence has been selected for this package. Unless and until a licence is added by the author, the source is provided for inspection and evaluation only; no general redistribution or reuse rights are granted. Do not describe it as an open-source licensed project.
