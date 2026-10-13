"""Shared helpers: reading the result files, the paired bootstrap and number formatting used in the report."""
import csv
import os
from collections import defaultdict

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
QUESTION_SETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "question_sets")
SEED = 20261010           # bootstrap seed used throughout the report
RESAMPLES = 5000

K3, DS, D7, GF, G5, QW = "kimi-k3", "deepseek-v4.1-flash", "deepseek-v4-flash-0731", "glm-5.3-flash", "glm-5.3", "qwen3.8-27b"
MODELS = [DS, D7, K3, GF, G5, QW]                      # column order of the Figure 1 output
NAME = {"auto": "Auto", K3: "Kimi K3", DS: "DS 4.1", D7: "DS 0731", GF: "GLM-Flash", G5: "GLM-5.3", QW: "Qwen"}


def read(path):
    with open(os.path.join(RESULTS, path), newline="") as f:
        return list(csv.DictReader(f))


def read_ids(name):
    with open(os.path.join(QUESTION_SETS, name)) as f:
        return [line.strip() for line in f if line.strip()]


def by_arm(rows, key="question_id", arm="arm", value="score"):
    """rows -> {arm: {question: float}}"""
    out = defaultdict(dict)
    for r in rows:
        if r[value] != "":
            out[r[arm]][r[key]] = float(r[value])
    return out


def paired_ci(diff):
    """Mean of a per-question difference with its 95% percentile bootstrap interval, in points (x100).
    A fresh generator per comparison, so every interval can be reproduced on its own."""
    d = np.asarray(diff, float)
    idx = np.random.default_rng(SEED).integers(0, len(d), size=(RESAMPLES, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return 100 * d.mean(), 100 * lo, 100 * hi


def averaged_ci(per_set):
    """Equal-weight mean over several sets of per-question values (differences or scores), each set resampled by
    question with one generator in set order; returns (mean, low, high) in points."""
    rng = np.random.default_rng(SEED)
    boot, point = 0.0, 0.0
    for v in per_set:
        v = np.asarray(v, float)
        point += v.mean() / len(per_set)
        boot = boot + v[rng.integers(0, len(v), size=(RESAMPLES, len(v)))].mean(1) / len(per_set)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return 100 * point, 100 * lo, 100 * hi


def signed(v):
    s = f"{abs(v):.1f}"
    if s == "0.0":
        return "0.0"
    return ("+" if v > 0 else "−") + s


def diff_text(t):
    m, lo, hi = t
    if signed(lo) == signed(hi) == "0.0":
        return "0.0"
    return f"{signed(m)} ({signed(lo)} to {signed(hi)})"


def cost_text(x):
    """Validation tables: three decimals from 0.01 USD, two significant digits below."""
    if x >= 0.01:
        return f"{x:.3f}"
    return f"{x:.{1 - int(np.floor(np.log10(x)))}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
