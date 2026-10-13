# Question sets

IDs of the questions and tasks that are subsets of a public benchmark. The other selection benchmarks are used in full
as described in Table 2 of the report (IFEval 541 prompts, GPQA Diamond 198 questions, LiveBench public release of
2026-06-25 with 682 questions, AutomationBench 600 tasks, LiveCodeBench v6 from August 2024 to May 2025 with 454
questions, Terminal-Bench 2.1 89 tasks, τ³-bench 375 tasks of tau2-bench v1.0.1).

| File | Set | Source and how it was drawn |
|---|---|---|
| `hle_selection_1700.txt` | selection | Humanity's Last Exam, `cais/hle` on Hugging Face, test split, revision `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`. The 2,158 text-only questions (rows with an empty image field) are put in one order, `random.Random(20261010).sample(sorted(ids), len(ids))`; these are the first 1,700. |
| `hle_validation_457.txt` | validation | 457 of the remaining 458 questions of the same order (Section 3.1 of the report). |
| `livebench_selection_plot_unscrambling_40.txt` | selection | LiveBench release LiveBench-2024-07-26, plot unscrambling questions of the holdout. |
| `livebench_validation_508.txt` | validation | LiveBench release LiveBench-2024-07-26: questions not among the 682 selection questions and not similar to any of them, other than the 40 above. |
| `tau3_long_context_45.txt` | selection | τ³-bench (tau2-bench v1.0.1): the 45 long banking conversations. Figure 1 leaves them out; they are used to choose the long-context model (Section 5.2 of the report). |
| `telecom_validation_100.txt` | validation | tau2-bench v1.0.1 telecom, `tasks_full.json` (2,285 tasks) minus the 114 tasks of the `base` split; `random.Random(20261012).sample(sorted(pool), 200)`; these are the first 100 of that order. |

Question and answer text is not included; it is available from each benchmark's publisher.
