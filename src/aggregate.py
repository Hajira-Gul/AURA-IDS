import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEY = ["macro_f1_true", "binary_f1", "attack_detection_rate",
       "attack_family_recall", "binary_fpr", "unknown_auroc",
       "expl_spearman", "expl_top10_overlap"]
BUDGETS = ["0.01", "0.05", "0.1"]
TEST_METRICS = ["macro_f1_true", "binary_f1", "attack_detection_rate"]
rng = np.random.RandomState(0)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m, adj, run = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(x, n=10000):
    x = np.asarray(x)
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


df = pd.concat([pd.read_csv(f) for f in glob.glob("results/*.csv")], ignore_index=True)
df = df[df["metric"].isin(KEY)]

# 1. mean over evaluation windows for each method, seed and metric
ps = df.groupby(["method", "seed", "metric"])["value"].mean().reset_index()

# 2. mean +/- std over seeds, with bootstrap 95% CI
rows = []
for (method, metric), g in ps.groupby(["method", "metric"]):
    v = g["value"].dropna().values
    if len(v) == 0:
        continue
    lo, hi = boot_ci(v)
    rows.append(dict(method=method, metric=metric, n_seeds=len(v),
                     mean=v.mean(), std=v.std(ddof=1) if len(v) > 1 else np.nan,
                     ci_low=lo, ci_high=hi))
summary = pd.DataFrame(rows)
summary.to_csv("results/summary_long.csv", index=False)

wide = summary.assign(txt=lambda d: d["mean"].round(3).astype(str) + " ± " +
                      d["std"].round(3).astype(str)).pivot(
    index="method", columns="metric", values="txt")
wide.to_csv("results/summary_mean_std.csv")
pd.set_option("display.width", 250, "display.max_columns", 20)
print(wide)

short = summary.groupby("method")["n_seeds"].min()
print("\nMethods with fewer than 10 seeds:", short[short < 10].to_dict() or "none")

# 3. paired Wilcoxon: full vs every other method at the same budget, Holm-corrected
out = []
for b in BUDGETS:
    ref = f"full_b{b}"
    if ref not in ps["method"].values:
        continue
    others = [m for m in ps["method"].unique()
              if m != ref and (m.endswith(f"_b{b}") or "_b" not in m)]
    for metric in TEST_METRICS:
        rows_b = []
        a = ps[(ps.method == ref) & (ps.metric == metric)].set_index("seed")["value"]
        for m in others:
            c = ps[(ps.method == m) & (ps.metric == metric)].set_index("seed")["value"]
            both = a.index.intersection(c.index)
            diff = (a[both] - c[both]).dropna()
            if len(diff) < 6 or (diff == 0).all():
                p = np.nan
            else:
                p = wilcoxon(diff).pvalue
            rows_b.append(dict(budget=b, metric=metric, compared_to=m,
                               mean_diff=diff.mean(), n=len(diff), p_raw=p))
        t = pd.DataFrame(rows_b)
        ok = t["p_raw"].notna()
        t.loc[ok, "p_holm"] = holm(t.loc[ok, "p_raw"])
        out.append(t)
tests = pd.concat(out, ignore_index=True)
tests.to_csv("results/paired_tests.csv", index=False)
print("\nPaired tests (full vs others; mean_diff > 0 means full is higher):")
print(tests.round(4).to_string())