# Methods and assumptions

The feasibility study investigates negotiation between a single synthetic seller and scripted buyer counterparties. A private seller mandate contains a price floor and per-trade quantity constraint. The externally imposed policy gate attempts to prevent unauthorised commitments and excessive disclosure of private mandate information under a **specified finite model**.

The base policy and the `floor_only` diagnostic protect different properties; do not combine the results. The experiments compare a plain LLM seller, a disclosure-aware gated seller, a gated seller with preview, and scripted baseline agents. A follow-up tests an answerability indicator.

The main usefulness outcome is completed cooperative agreements among **feasible** negotiation cases. Other checks examine attempted prohibited disclosures, refusal decisions and declared inference-bound violations. Repeated trials on the same scenarios do not count as new independent task samples.

The October 2026 reports cover one low-effort model, synthetic cases and short interactions. The source subset here enables inspection and offline mechanism/harness testing. The original sealed scenarios, API calls, model settings and run metadata remain in a separate private archive and are not publicly reproducible from this package.

An external deterministic gate can enforce a correctly specified policy under complete mediation; that mechanism is established security engineering. The empirical question is how such boundaries interact with **model behaviour and task usefulness**.
