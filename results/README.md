# Per-question results

One row per question (or task) and arm, with the score and the cost. The scripts in [`../analysis`](../analysis)
recompute the numbers in the report's figures and tables from these files. Question IDs are the benchmarks' own; the question lists of the
subsets are in [`../question_sets`](../question_sets).

- Scores are between 0 and 1 (1 = correct or passed); some LiveBench tasks give partial credit. The report shows them
  multiplied by 100.
- Costs are in USD, from the provider's billing, including reasoning tokens. Auto's cost includes the Jev call and,
  after a handoff, both answers.
- In Auto rows, `selected_model` is the model Jev's class was sent to, and `handed_off` is 1 when the answer came
  from Kimi K3 after a handoff (HLE only).
- No question or answer text and no model responses are included.

Arms and models use the full model names: `deepseek-v4.1-flash` (DS 4.1), `deepseek-v4-flash-0731` (DS 0731),
`kimi-k3` (Kimi K3), `glm-5.3-flash` (GLM-Flash), `glm-5.3` (GLM-5.3), `qwen3.8-27b` (Qwen); the Auto arm is `auto`.

## Selection set

### `selection/single_models.csv`

Each model's result on every selection question (the 9 panels of Figure 1), with 3 runs per question on GPQA.

| Column | Definition |
|---|---|
| `benchmark` | `ifeval`, `gpqa` (GPQA Diamond), `livebench`, `automationbench`, `livecodebench` (LiveCodeBench v6), `terminal_bench_2_1`, `tau3` (τ³-bench), `hle` (the first 1,700 HLE questions), `livebench_plot_unscrambling` (the 40 plot-unscrambling questions of the LiveBench holdout) |
| `question_id` | The benchmark's ID: IFEval key, GPQA record ID, LiveBench question ID, AutomationBench `domain.task`, LiveCodeBench question ID, Terminal-Bench task name, τ³ `domain\|task ID`, HLE ID |
| `run` | GPQA only: run number, 0 to 2; empty for the other benchmarks |
| `category` | LiveBench category, AutomationBench domain, τ³ domain, HLE category; empty for the other benchmarks |
| `task` | LiveBench task; empty for the other benchmarks |
| `model` | The model that answered |
| `score` | Official grade, 0 to 1 (LiveBench partial credit); on HLE, empty answers and answers without a verdict are 0 (Section 4.3 of the report) |
| `cost_usd` | Cost of the question (of the run, on GPQA); empty for τ³, see `tau3_cost.csv` |

DS 0731 did not run the plot-unscrambling questions. DS 4.1 has 372 τ³ rows: the 3 long banking conversations it
did not finish are left out. Figure 1 leaves out the 45 long banking conversations listed in
`question_sets/tau3_long_context_45.txt`.

### `selection/tau3_cost.csv`

τ³ costs are given as each model's mean per task.

| Column | Definition |
|---|---|
| `model` | The model |
| `cost_usd_per_task` | Mean cost per task over the model's τ³ tasks, including the simulated customer (Section 3.4) |

### `selection/routing.csv`

Jev's class for every selection question (Figure 2).

| Column | Definition |
|---|---|
| `benchmark` | As in `single_models.csv` |
| `question_id` | As in `single_models.csv` |
| `expected_class` | The class labelled from the class definitions (Section 6.1); empty for IFEval, whose prompts span several direct-answer classes that all use DS 4.1 |
| `expected_model` | The primary model of the expected class (Table 1) |
| `jev_class` | The class Jev chose |
| `selected_model` | The primary model of Jev's class |

A question counts as routed to the model of its class when `selected_model` equals `expected_model`.

## Validation sets

### `validation/hle.csv`

The 457 HLE validation questions for Auto and five single models (Figure 3), and one more row.

| Column | Definition |
|---|---|
| `question_id` | HLE ID |
| `subject` | HLE category |
| `split` | `validation` for the 457 questions in `question_sets/hle_validation_457.txt`; `other` for Kimi K3's result on the one remaining HLE question, used only for its accuracy on all 2,158 questions |
| `arm` | `auto` or a model |
| `selected_model` | Auto: the model Jev's class was sent to; single models: the model itself |
| `handed_off` | Auto: 1 when DS 4.1 gave no answer and the answer came from Kimi K3, otherwise 0; empty for single models |
| `score` | 1 if the judge marked the answer correct, otherwise 0; empty answers and answers without a verdict are 0 (Section 4.3) |
| `cost_usd` | Cost of the question |

### `validation/livebench_holdout.csv`

The 508 LiveBench holdout questions for Auto and five single models (Figure 4).

| Column | Definition |
|---|---|
| `question_id` | LiveBench question ID, as in `question_sets/livebench_validation_508.txt` |
| `category` | LiveBench category |
| `task` | LiveBench task |
| `arm` | `auto` or a model |
| `selected_model` | Auto: the model Jev's class was sent to; single models: the model itself |
| `score` | Official grade, 0 to 1 (some tasks give partial credit) |
| `cost_usd` | Cost of the question |

### `validation/telecom.csv`

The 100 τ²-bench telecom tasks for Auto and six single models, one trial per task (Figure 5).

| Column | Definition |
|---|---|
| `task_id` | tau2-bench telecom task ID, as in `question_sets/telecom_validation_100.txt` |
| `arm` | `auto` or a model |
| `selected_model` | Auto: the model Jev's class was sent to; single models: the model itself |
| `passed` | 1 if the task passed, otherwise 0 |

### `validation/telecom_cost.csv`

Telecom costs are per-arm totals, because several tasks can share one session. Only the side of the model under
test is counted; the user simulator is the same for every arm and is not included. Figure 5's cost per task is
`total_cost_usd / tasks`.

| Column | Definition |
|---|---|
| `arm` | `auto` or a model |
| `tasks` | Number of tasks |
| `total_cost_usd` | Total cost of the arm over these tasks |
