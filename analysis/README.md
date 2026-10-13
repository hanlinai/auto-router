# Analysis scripts

These scripts recompute the numbers in the report's figures and tables, and the confidence intervals, from the per-question results in
[`../results`](../results) and the question lists in [`../question_sets`](../question_sets). They need no network
access.

```bash
cd analysis
pip install -r requirements.txt
python selection.py
python validation.py
```

The scripts were run with Python 3.12 and numpy 2.5.

| Script | Prints or writes |
|---|---|
| `selection.py` | Figure 1 (accuracy and cost per question on each benchmark), Figure 2 (routing on the selection set, with the main confusions quoted in Section 6.1) and the GPQA single-run accuracy of each model with its run-to-run range (Section 4.2) |
| `validation.py` | Figures 3 to 6, Table 5, the route counts and the handed-off comparison of Section 7.1, Kimi K3's HLE accuracy on all 2,158 questions and on the selection and validation questions, Auto's routing and cost multiples on each validation set (Sections 6.3, 7.1 to 7.3), the telecom tasks passed by every arm, the equal-weight average with its 95% interval and the cost share of every arm (Figure 7), the two-set averages without telecom, and the Pareto line (Section 7.4) |
| `common.py` | Shared helpers: reading the result files, the bootstrap and number formatting |

Figure 1 uses the official weighting for LiveBench (mean within each task, then within each category, then
across categories), the mean of the 3 runs for GPQA, and the τ³ tasks other than the 45 long banking conversations
listed in `question_sets/tau3_long_context_45.txt`. On the validation sets every score is the per-question mean.

## Confidence intervals

Every interval is a 95% percentile bootstrap interval from 5,000 resamples with seed 20261010. Differences between
Auto and another arm are paired: the per-question difference is resampled by question, so both arms always use the
same questions. Each comparison on one set starts a fresh generator with the seed, so any interval can be reproduced
on its own. For the equal-weight average of several sets, each set is resampled separately by question with one
generator, in the order HLE, LiveBench holdout, telecom, and the resampled set means are averaged; this gives the
intervals of the averaged differences and, applied to each arm's own scores, the error bars of Figure 7. Costs have
no intervals.

## Numbers not recomputed here

Some numbers in the report depend on data that these files do not contain:

- **Table 3** (basis for each class): it pools per-class subsets of the selection benchmarks, and some of its bases
  come from earlier tests on other benchmarks and from public leaderboards.
- **Table 4 and the simulated-Auto estimate in Section 6.3**: simulated Auto takes Kimi K3's answer where DS 4.1
  gave no answer, which needs a per-answer "no answer" flag. The same flag underlies the counts of empty answers in
  Sections 5.2 and 7.1.
- **Section 4.3 judge checks**: they compare judge verdicts with each other and with exact-match scoring, which
  needs the model responses.
- **Section 3.3 difficulty levels of LiveCodeBench and Terminal-Bench 2.1**: they need the benchmarks' difficulty
  labels. The GPQA groups of Section 3.3 and Table 3 follow `expected_class` in `results/selection/routing.csv`.
- **Section 2.1 Jev cost shares**: the Jev call is included in Auto's cost per question but is not listed
  separately.
- **Jev's classes and sessions in the validation runs** (Section 4.4 coverage, the 88.2% in Section 7.1, the 75
  telecom sessions in Section 6.3): the result files give the model each question was sent to, not the class or
  the session.
