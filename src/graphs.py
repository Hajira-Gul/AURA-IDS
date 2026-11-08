import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
files = [f for f in glob.glob("results/*.csv")
         if not os.path.basename(f).startswith(("summary", "paired", "label"))]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
ps = df.groupby(["method", "seed", "metric"])["value"].mean().reset_index()
B = "0.05"


def stat(method, metric):
    g = ps[(ps.method == method) & (ps.metric == metric)]["value"].dropna()
    if len(g) == 0:
        return np.nan, 0.0
    return g.mean(), (g.std(ddof=1) if len(g) > 1 else 0.0)


def present(ms, metric="macro_f1_true"):
    return [m for m in ms if not np.isnan(stat(m, metric)[0])]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=200)
    plt.close(fig)
    print("saved", name)


# 1-2. per-window curves (mean +/- std over seeds)
curve = present(["static_mlp", f"naive_ft_mlp_b{B}", f"periodic_mlp_b{B}",
                 f"river_hat_b{B}", f"full_b{B}", f"no_unknown_b{B}",
                 f"unknown_only_b{B}"])
for metric, name in [("macro_f1_true", "fig1_window_macro_f1"),
                     ("attack_detection_rate", "fig2_window_detection")]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for mname in curve:
        g = df[(df.method == mname) & (df.metric == metric)].groupby("window")["value"]
        mu, sd = g.mean(), g.std().fillna(0)
        ax.plot(mu.index, mu.values, marker="o", ms=3, label=mname)
        ax.fill_between(mu.index, mu - sd, mu + sd, alpha=0.12)
    ax.set_xlabel("window (7-19 = evaluation stream)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per window, label budget {B} (mean ± std over seeds)")
    ax.legend(fontsize=7)
    save(fig, name)

# 3. main comparison bars
main = present(["static_mlp", "static_lightgbm", f"naive_ft_mlp_b{B}", f"periodic_mlp_b{B}",
                f"river_hat_b{B}", f"ssf_like_b{B}", f"unknown_only_b{B}",
                f"no_unknown_b{B}", f"full_b{B}"])
metrics = ["macro_f1_true", "binary_f1", "attack_detection_rate"]
fig, ax = plt.subplots(figsize=(11, 5))
x, w = np.arange(len(main)), 0.27
for i, metric in enumerate(metrics):
    mus, sds = zip(*[stat(m, metric) for m in main])
    ax.bar(x + (i - 1) * w, mus, w, yerr=sds, capsize=2, label=metric)
ax.set_xticks(x)
ax.set_xticklabels(main, rotation=30, ha="right", fontsize=8)
ax.set_title(f"Method comparison, budget {B} (mean ± std over seeds)")
ax.legend()
save(fig, "fig3_method_comparison")

# 4. ablations (dots + error bars, so a non-zero axis is not misleading)
abl = present([f"{v}_b{B}" for v in ["full", "no_unknown", "no_replay", "random_select",
               "ks_only", "adwin_only", "periodic_trig", "ssf_like", "unknown_only"]])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric in zip(axes, ["macro_f1_true", "binary_f1"]):
    mus, sds = zip(*[stat(a, metric) for a in abl])
    ax.errorbar(range(len(abl)), mus, yerr=sds, fmt="o", capsize=3)
    ax.axhline(stat(f"full_b{B}", metric)[0], color="red", ls="--", lw=1)
    ax.set_xticks(range(len(abl)))
    ax.set_xticklabels([a.replace(f"_b{B}", "") for a in abl], rotation=40, ha="right")
    ax.set_title(f"Ablations: {metric} (red = full)")
save(fig, "fig4_ablations")

# 5. effect of label budget
budgets = ["0.01", "0.05", "0.1"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric in zip(axes, ["macro_f1_true", "binary_f1"]):
    for base in ["periodic_mlp", "naive_ft_mlp", "full", "no_unknown", "periodic_trig"]:
        pts = [(float(b), *stat(f"{base}_b{b}", metric)) for b in budgets]
        pts = [p for p in pts if not np.isnan(p[1])]
        if pts:
            bx, mu, sd = zip(*pts)
            ax.errorbar(bx, mu, yerr=sd, marker="o", capsize=3, label=base)
    ax.axhline(stat("static_mlp", metric)[0], color="gray", ls=":", label="static_mlp")
    ax.set_xscale("log")
    ax.set_xlabel("label budget per window")
    ax.set_title(metric)
    ax.legend(fontsize=7)
save(fig, "fig5_label_budget")

# 6. labels used vs performance, plus label_cost.csv
n_eval = int((pd.read_parquet("data/processed/features.parquet",
                              columns=["split"])["split"] != "train").sum())
lab = (df[df.metric == "labels_used"].groupby(["method", "seed"])["value"].sum()
       .groupby("method").mean())
adp = (df[df.metric == "adapted"].groupby(["method", "seed"])["value"].sum()
       .groupby("method").mean())


def labels_for(m):
    if m.startswith(("static_", "unknown_only")):
        return 0.0
    if m.startswith(("naive_ft", "periodic_mlp", "river")):
        return float(m.split("_b")[-1]) * n_eval   # baselines label this fraction of every window
    return float(lab.get(m, np.nan))


rows = []
for m in sorted(ps["method"].unique()):
    rows.append(dict(method=m, labels=labels_for(m), adaptations=float(adp.get(m, np.nan)),
                     macro_f1_true=stat(m, "macro_f1_true")[0],
                     binary_f1=stat(m, "binary_f1")[0],
                     attack_detection_rate=stat(m, "attack_detection_rate")[0]))
tab = pd.DataFrame(rows)
tab.to_csv("results/label_cost.csv", index=False)
print(tab.round(3).to_string())
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric in zip(axes, ["binary_f1", "macro_f1_true"]):
    for _, r in tab.dropna(subset=["labels", metric]).iterrows():
        ax.scatter(r["labels"], r[metric], s=22)
        ax.annotate(r["method"], (r["labels"], r[metric]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("labels used over the stream")
    ax.set_title(metric)
save(fig, "fig6_labels_vs_performance")

# 7. explanation stability
ex = present([f"{v}_b{B}" for v in ["full", "no_unknown", "no_replay", "random_select",
              "ks_only", "adwin_only", "periodic_trig", "ssf_like"]], "expl_spearman")
fig, ax = plt.subplots(figsize=(9, 4.5))
x, w = np.arange(len(ex)), 0.38
for i, metric in enumerate(["expl_spearman", "expl_top10_overlap"]):
    mus, sds = zip(*[stat(m, metric) for m in ex])
    ax.bar(x + (i - 0.5) * w, mus, w, yerr=sds, capsize=2, label=metric)
ax.set_xticks(x)
ax.set_xticklabels([m.replace(f"_b{B}", "") for m in ex], rotation=30, ha="right")
ax.set_title("Explanation stability before vs after adaptation")
ax.legend()
save(fig, "fig7_explanation_stability")