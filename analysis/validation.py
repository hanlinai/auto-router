"""Validation sets: Figures 3 to 7 and Table 5 of the report with the numbers quoted in Sections 6.3 and 7.
usage: python validation.py"""
from collections import Counter

import numpy as np

from common import DS, D7, G5, GF, K3, NAME, QW, averaged_ci, by_arm, cost_text, diff_text, paired_ci, read, read_ids, table

ARMS = ["auto", K3, DS, G5, GF, QW]


def per_set_table(title, metric, score, cost, arms, unit="Cost per question"):
    a = np.array(score["auto"])
    rows = []
    for m in arms:
        x = np.array(score[m])
        rows.append([NAME[m], f"{100 * x.mean():.1f}", cost_text(float(np.mean(cost[m]))),
                     "none" if m == "auto" else diff_text(paired_ci(a - x))])
    print(f"\n{title}\n")
    print(table(["Arm", metric, f"{unit} (USD)", "Auto − arm (95% CI)"], rows))


def main():
    # ---------------------------------------------------------------- HLE (Figure 3, Table 5, Section 7.1)
    hle_rows = read("validation/hle.csv")
    val = [r for r in hle_rows if r["split"] == "validation"]
    qs = read_ids("hle_validation_457.txt")
    sc, co = by_arm(val), by_arm(val, value="cost_usd")
    H = {m: [sc[m][q] for q in qs] for m in ARMS}
    HC = {m: [co[m][q] for q in qs] for m in ARMS}
    per_set_table("Figure 3. HLE validation set (457 questions)", "Accuracy", H, HC, ARMS)
    r_k3, r_ds = np.mean(HC["auto"]) / np.mean(HC[K3]), np.mean(HC["auto"]) / np.mean(HC[DS])
    print(f"\nAuto's cost per question: {r_k3:.0%} of Kimi K3's, {r_ds:.0f} times DS 4.1's")
    auto = {r["question_id"]: r for r in val if r["arm"] == "auto"}
    route = {}
    for q in qs:
        m, h = auto[q]["selected_model"], auto[q]["handed_off"] == "1"
        route[q] = ("Sent to DS 4.1, handed off to Kimi K3" if h else "Sent to DS 4.1, its answer kept" if m == DS
                    else "Sent to Kimi K3" if m == K3 else "Classified as research agent, sent to GLM-Flash" if m == GF else m)
    rows = []
    for g in ("Sent to DS 4.1, its answer kept", "Sent to DS 4.1, handed off to Kimi K3", "Sent to Kimi K3",
              "Classified as research agent, sent to GLM-Flash"):
        idx = [i for i, q in enumerate(qs) if route[q] == g]
        a, k, d = (np.array(H[m])[idx] for m in ("auto", K3, DS))
        rows.append([g, len(idx), f"{100 * a.mean():.1f}", f"{100 * k.mean():.1f}", f"{100 * d.mean():.1f}",
                     diff_text(paired_ci(a - k)) if len(idx) > 2 else "none"])
    print("\nTable 5. HLE validation set split by route\n")
    print(table(["Route", "Questions", "Auto", "Kimi K3", "DS 4.1", "Auto − Kimi K3 (95% CI)"], rows))
    first = Counter(auto[q]["selected_model"] for q in qs)
    print(f"\nFirst model: {dict((NAME[m], n) for m, n in first.items())}; handed off: {sum(auto[q]['handed_off'] == '1' for q in qs)}")
    idx = [i for i, q in enumerate(qs) if route[q] == "Sent to DS 4.1, handed off to Kimi K3"]
    print("Handed-off questions, Auto − DS 4.1:", diff_text(paired_ci(np.array(H["auto"])[idx] - np.array(H[DS])[idx])))
    sel = read("selection/single_models.csv")
    k3_sel = [float(r["score"]) for r in sel if r["benchmark"] == "hle" and r["model"] == K3]
    ds_sel = [float(r["score"]) for r in sel if r["benchmark"] == "hle" and r["model"] == DS]
    k3_other = [float(r["score"]) for r in hle_rows if r["split"] == "other" and r["arm"] == K3]
    n_all = len(k3_sel) + len(qs) + len(k3_other)
    print(f"Kimi K3 on all {n_all:,} questions: {100 * (sum(k3_sel) + sum(H[K3]) + sum(k3_other)) / n_all:.1f}%; "
          f"first 1,700 vs 457: Kimi K3 {100 * np.mean(k3_sel):.1f} → {100 * np.mean(H[K3]):.1f}, "
          f"DS 4.1 {100 * np.mean(ds_sel):.1f} → {100 * np.mean(H[DS]):.1f}")

    # ---------------------------------------------------------------- LiveBench holdout (Figure 4, Section 7.2)
    lb = read("validation/livebench_holdout.csv")
    lqs = read_ids("livebench_validation_508.txt")
    sc, co = by_arm(lb), by_arm(lb, value="cost_usd")
    L = {m: [sc[m][q] for q in lqs] for m in ARMS}
    LC = {m: [co[m][q] for q in lqs] for m in ARMS}
    per_set_table("Figure 4. LiveBench holdout (508 questions)", "Score", L, LC, ARMS)
    lauto = Counter(r["selected_model"] for r in lb if r["arm"] == "auto")
    print(f"\nAuto's routing: {dict((NAME[m], n) for m, n in lauto.items())}; cost per question "
          f"{np.mean(LC['auto']) / np.mean(LC[K3]):.0%} of Kimi K3's, {np.mean(LC['auto']) / np.mean(LC[DS]):.1f} times DS 4.1's")
    cats = Counter(next(r["category"] for r in lb if r["question_id"] == q) for q in lqs)
    print("Categories:", dict(cats))

    # ---------------------------------------------------------------- telecom (Figure 5, Section 7.3)
    tel = read("validation/telecom.csv")
    tids = read_ids("telecom_validation_100.txt")
    passed = {}
    for r in tel:
        passed.setdefault(r["arm"], {})[r["task_id"]] = float(r["passed"])
    tcost = {r["arm"]: float(r["total_cost_usd"]) / int(r["tasks"]) for r in read("validation/telecom_cost.csv")}
    tarms = ["auto", K3, DS, G5, GF, D7, QW]
    T = {m: [passed[m][t] for t in tids] for m in tarms}
    TC = {m: [tcost[m]] * len(tids) for m in tarms}
    per_set_table("Figure 5. τ²-bench telecom (100 tasks; side of the model under test only)", "Pass rate", T, TC, tarms,
                  unit="Cost per task")
    allpass = sum(all(T[m][i] == 1 for m in tarms) for i in range(len(tids)))
    allfail = sum(all(T[m][i] == 0 for m in tarms) for i in range(len(tids)))
    print(f"\nAuto's cost per task: {tcost['auto'] / tcost[K3]:.0%} of Kimi K3's, {tcost['auto'] / tcost[DS]:.0f} times DS 4.1's; "
          f"all arms pass {allpass} tasks, all arms fail {allfail}")

    # ---------------------------------------------------------------- Figures 6 and 7, Section 7.4
    sets = {"HLE (457 questions)": (H, HC), "LiveBench holdout (508 questions)": (L, LC), "τ²-bench telecom (100 tasks)": (T, TC)}
    rows = []
    for name, (S, C) in sets.items():
        a = np.array(S["auto"])
        rows.append([name, f"{100 * np.mean(S[K3]):.1f}", f"{100 * np.mean(S[DS]):.1f}",
                     f"{100 * a.mean():.1f} ({np.mean(C['auto']) / np.mean(C[K3]):.0%})",
                     diff_text(paired_ci(a - np.array(S[K3]))), diff_text(paired_ci(a - np.array(S[DS])))])
    pts = []
    for m in ARMS:
        score, lo, hi = averaged_ci([np.array(S[m]) for S, _ in sets.values()])
        rel = float(np.mean([np.mean(C[m]) / np.mean(C[K3]) for _, C in sets.values()]))
        pts.append({"arm": m, "name": NAME[m], "score": score, "ci_lo": lo, "ci_hi": hi, "cost_rel": rel})
    P = {p["arm"]: p for p in pts}
    d3 = {b: averaged_ci([np.array(S["auto"]) - np.array(S[b]) for S, _ in sets.values()]) for b in (K3, DS)}
    rows.append(["Equal-weight average", f"{P[K3]['score']:.1f}", f"{P[DS]['score']:.1f}",
                 f"{P['auto']['score']:.1f} ({P['auto']['cost_rel']:.0%})", diff_text(d3[K3]), diff_text(d3[DS])])
    print("\nFigure 6 and Section 7.4. Summary of the three validation sets (in parentheses: cost per question as a share of Kimi K3's)\n")
    print(table(["Validation set", "Kimi K3", "DS 4.1", "Auto", "Auto − Kimi K3", "Auto − DS 4.1"], rows))
    print("\nFigure 7. Equal-weight average and cost share of every arm:")
    for p in pts:
        print(f"- {p['name']}: {p['score']:.1f} (95% CI {p['ci_lo']:.1f} to {p['ci_hi']:.1f}), cost {p['cost_rel']:.0%} of Kimi K3")
    two = list(sets.values())[:2]
    for b in (K3, DS):
        print(f"Without telecom (HLE and LiveBench holdout, equal weight), Auto − {NAME[b]}:",
              diff_text(averaged_ci([np.array(S["auto"]) - np.array(S[b]) for S, _ in two])))
    front = [p["name"] for p in pts if not any(q["cost_rel"] < p["cost_rel"] and q["score"] > p["score"] for q in pts)]
    print("Pareto line:", ", ".join(front))
    for name, (S, C) in sets.items():
        dom = [NAME[m] for m in S if m != "auto" and np.mean(C[m]) < np.mean(C["auto"]) and np.mean(S[m]) > np.mean(S["auto"])]
        print(f"{name}: single models both cheaper and higher-scoring than Auto: {', '.join(dom) or 'none'}")


if __name__ == "__main__":
    main()
