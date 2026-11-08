import json
import os
import sys
import numpy as np
import pandas as pd
from river import tree
from data import load_config
from run_baselines import window_metrics

INIT_TRAIN = 20000   # subsample of the training windows (river is slow per sample)


def to_dicts(X, feats):
    return [dict(zip(feats, row)) for row in X]


def main():
    cfg = load_config()
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    seeds = [int(s) for s in sys.argv[2:]] or [0]

    df = pd.read_parquet(cfg["features_path"])
    info = json.load(open(cfg["features_path"].replace(".parquet", "_info.json")))
    feats = info["features"]
    train = df[df["split"] == "train"]
    eval_windows = sorted(df.loc[df["split"] != "train", "window_id"].unique())

    rows = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        model = tree.HoeffdingAdaptiveTreeClassifier(seed=seed)

        # initial training on a shuffled subsample of the training windows
        sub = train.iloc[rng.permutation(len(train))[:INIT_TRAIN]]
        for x, y in zip(to_dicts(sub[feats].values, feats), sub["family"].values):
            model.learn_one(x, y)
        print(f"seed {seed} | budget {budget} | initial training done")

        for w in eval_windows:
            d = df[df["window_id"] == w]
            X, y_str = d[feats].values, d["family"].values
            xs = to_dicts(X, feats)

            # 1. TEST first
            y_pred = np.array([model.predict_one(x) or "BENIGN" for x in xs])
            for metric, value in window_metrics(y_str, y_pred).items():
                rows.append(dict(method=f"river_hat_b{budget}", protocol="A",
                                 dataset="CIC-IDS2017", seed=seed,
                                 window=int(w), metric=metric, value=value))

            # 2. then TRAIN on a random labelled fraction of this window
            n_lab = max(1, int(budget * len(X)))
            for i in rng.choice(len(X), n_lab, replace=False):
                model.learn_one(xs[i], y_str[i])
            print(f"  window {w} done")

    os.makedirs("results", exist_ok=True)
    out = pd.DataFrame(rows)
    path = f"results/river_baseline_b{budget}.csv"
    out.to_csv(path, index=False)
    print(f"\nsaved {path}")

    keep = ["macro_f1_true", "binary_f1", "attack_detection_rate",
            "attack_family_recall", "binary_fpr"]
    print("\nMean over evaluation windows 7-19:")
    print(out[out["metric"].isin(keep)].groupby(["method", "metric"])["value"]
          .mean().unstack().round(4))


if __name__ == "__main__":
    main()