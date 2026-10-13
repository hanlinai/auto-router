# Jev-Based Multi-Model Routing

Auto routes each conversation to one of six open-weight models. For each new conversation, Jev, a decision model
from TypeSafe AI, picks one of 16 task classes, and a fixed policy sends the conversation to the model chosen for
that class: the cheapest model whose score on that kind of task is not clearly below the best model's. This
repository holds the technical report, the routing decision used in every Auto run, the question sets, the
per-question results and the scripts that recompute the report's numbers.

**Read the report: [REPORT.md](REPORT.md).** Measurements were taken in October 2026 with models served by engy.ai.

## Contents

| Path | What it is |
|---|---|
| [`REPORT.md`](REPORT.md) | Technical report: design, selection rule, implementation checks, and results on three validation sets (HLE, LiveBench holdout, τ²-bench telecom) |
| [`figures/`](figures) | The report's figures and table images |
| [`router/`](router) | Reference implementation of the routing decision: Jev classification with the published prompt, the policy, sessions, the long-context chain and the handoff rule, with unit tests; see its README |
| [`question_sets/`](question_sets) | IDs of the questions and tasks drawn from public benchmarks, with how they were drawn; see its README |
| [`results/`](results) | Per-question results of every arm on the selection and validation sets (question ID, selected model, score and cost); see its README |
| [`analysis/`](analysis) | Scripts that recompute the numbers in the report's figures and tables, and the confidence intervals, from `results/`; see its README |

## Running Auto

```bash
cd router
pip install -r requirements.txt
python -m pytest tests            # unit tests, no network access
```

`router/README.md` shows how to put the decision in front of an OpenAI-compatible endpoint. Running it needs a Jev
API key from TypeSafe AI and access to the six models named in `router/policy.json`.

## Recomputing the numbers

```bash
cd analysis
pip install -r requirements.txt
python selection.py              # Figures 1 and 2
python validation.py             # Figures 3 to 7, Table 5 and the numbers of Sections 6.3 and 7
```

`analysis/README.md` lists what each script prints and the numbers in the report that need data not included here.

## Repeating the measurements

Each benchmark was run with its publisher's evaluation code and the settings in Section 3.2 of the report (reasoning
effort at the maximum, sampling, judge models and user simulator). The selection benchmarks are used in full except
for the subsets listed in `question_sets/`; the validation sets are listed there too. Each question was answered
once, so a repeated run will differ by the run-to-run variation discussed in Section 4.2, and results obtained from
another provider or another serving configuration can differ further.

Question and answer text, model responses and request logs are not included.

## License

Apache License 2.0; see [LICENSE](LICENSE).
