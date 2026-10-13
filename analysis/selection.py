"""Selection set (in-sample): Figures 1 and 2 of the report and the GPQA run-to-run spread of Section 4.2.
usage: python selection.py"""
from collections import Counter, defaultdict

import numpy as np

from common import MODELS, NAME, read, read_ids, table

ROWS = ["ifeval", "gpqa", "livebench", "automationbench", "livecodebench", "terminal_bench_2_1", "tau3", "hle",
        "livebench_plot_unscrambling"]
LABEL = {"ifeval": "IFEval", "gpqa": "GPQA Diamond", "livebench": "LiveBench", "automationbench": "AutomationBench",
         "livecodebench": "LiveCodeBench v6", "terminal_bench_2_1": "Terminal-Bench 2.1", "tau3": "τ³-bench (330 tasks)",
         "hle": "HLE (first 1,700 questions)", "livebench_plot_unscrambling": "LiveBench plot unscrambling (40 questions)"}


def accuracy(rows, bench, long45):
    """Score as defined in Figure 1: GPQA is the mean of its 3 runs; LiveBench uses the official weighting (mean
    within each task, then within each category, then across categories); τ³ leaves out the 45 long conversations."""
    rs = [r for r in rows if r["benchmark"] == bench]
    if bench == "tau3":
        rs = [r for r in rs if r["question_id"] not in long45]
    if bench == "livebench":
        cat = defaultdict(lambda: defaultdict(list))
        for r in rs:
            cat[r["category"]][r["task"]].append(float(r["score"]))
        return 100 * np.mean([np.mean([np.mean(v) for v in tasks.values()]) for tasks in cat.values()])
    return 100 * np.mean([float(r["score"]) for r in rs])


def main():
    rows = read("selection/single_models.csv")
    long45 = set(read_ids("tau3_long_context_45.txt"))
    per = defaultdict(list)
    for r in rows:
        per[(r["benchmark"], r["model"])].append(r)
    tau_cost = {r["model"]: float(r["cost_usd_per_task"]) for r in read("selection/tau3_cost.csv")}

    t3, t4 = [], []
    for b in ROWS:
        acc, cost = [], []
        for m in MODELS:
            rs = per.get((b, m))
            if not rs:
                acc.append("—"); cost.append("—"); continue
            acc.append(f"{accuracy(rs, b, long45):.1f}")
            c = tau_cost[m] if b == "tau3" else np.mean([float(r["cost_usd"]) for r in rs])
            cost.append(f"{c:.3f}" if c >= 0.01 else f"{c:.4f}")
        t3.append([LABEL[b]] + acc); t4.append([LABEL[b].replace(" (330 tasks)", "")] + cost)
    print("Figure 1. Accuracy (%) per benchmark\n")
    print(table(["Benchmark"] + [NAME[m] for m in MODELS], t3))
    print("\nFigure 1. Cost per question (USD); τ³ is the mean over all its tasks\n")
    print(table(["Benchmark"] + [NAME[m] for m in MODELS], t4))

    routing = read("selection/routing.csv")
    tot, ok, conf = Counter(), Counter(), defaultdict(Counter)
    for r in routing:
        tot[r["benchmark"]] += 1
        if r["selected_model"] == r["expected_model"]:
            ok[r["benchmark"]] += 1
        else:
            conf[r["benchmark"]][f"{r['expected_class'] or 'ifeval'} → {r['jev_class']}"] += 1
    print("\nFigure 2. Routing on the selection set (questions routed to the model of their class)\n")
    t6 = [[LABEL.get(b, b).replace(" (330 tasks)", ""), tot[b], ok[b],
           "; ".join(f"{n} {k}" for k, n in conf[b].most_common(4)) or "none"] for b in ROWS if b in tot]
    t6.append(["Total", sum(tot.values()), f"{sum(ok.values())} ({100 * sum(ok.values()) / sum(tot.values()):.1f}%)",
               f"{sum(tot.values()) - sum(ok.values())} questions"])
    print(table(["Benchmark", "Questions", "Routed to the model of their class", "Main confusions"], t6))

    print("\nSection 4.2. GPQA single-run accuracy per model (3 runs)\n")
    spreads = []
    for m in MODELS:
        runs = defaultdict(list)
        for r in per[("gpqa", m)]:
            runs[r["run"]].append(float(r["score"]))
        acc = sorted(100 * np.mean(v) for v in runs.values())
        spreads.append(acc[-1] - acc[0])
        print(f"- {NAME[m]}: " + ", ".join(f"{a:.1f}" for a in acc) + f" (range {acc[-1] - acc[0]:.1f})")
    print(f"- range across models: {min(spreads):.1f} to {max(spreads):.1f} points")


if __name__ == "__main__":
    main()
