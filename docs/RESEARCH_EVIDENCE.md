# Evidence boundaries and limitations

This review candidate contains implementation and synthetic offline evaluation mechanisms. It is not the original full experiment archive.

The separate private audit of 8 Nov 2026 verified 322 files against a manifest and 34 targeted tests. Two historical LLM experiments involved 4,452 recorded API calls with approx. $1.3274 usage-estimated cost. The complete data and trusted scenarios have been withheld pending a confidentiality and disclosure review.

**Experiment 1**: The base-policy LLM with gate only completed 1/1/1 of 11 feasible cases; the preview variant 4/7/7; the scripted preview 7; the ungated LLM 9/10/9 but exceeded the declared inference-window threshold.

**Experiment 2**: Preview-only 0/0/0 of 8 feasible cases; answerability indicator 1/0/0; scripted preview 3. The answerability indicator reduced one type of bad attempted action but did not reliably improve useful agreements.

These are historical archived results, not freshly rerun with this subset. Repetitions on identical scenarios are not independent samples. No general prompt-injection resistance, multi-agent collusion detection, real-world commercial performance or statistical generalisation is established. A deterministic gate is established security engineering, not a claimed novel safety guarantee.

See the README for project description and the research question.
