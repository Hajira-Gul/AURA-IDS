import copy
import json
import os
import sys
import numpy as np
import pandas as pd
from data import load_config
from models import MLPNet
from run_baselines import window_metrics

PERIOD = 3          # periodic adaptation every 3 windows
FT_EPOCHS = 3       # epochs for naive fine-tuning
PER_EPOCHS = 5      # epochs for periodic adaptation
REPLAY_TRAIN = 20000


def main():
    cfg = load_config()
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    seeds = [int(s) for s in sys.argv[2:]] or [0]

    df = pd.read_parquet(cfg["features_path"])
    info = json.load(open(cfg["features_path"].replace(".parquet", "_info.json")))
    feats = info["features"]
    classes = np.array(sorted(df["family"].unique()))     # full label space
    enc = lambda y: np.searchsorted(classes, y)

    train = df[df["split"] == "train"]
    X_tr, y_tr = train[feats].values, enc(train["family"].values)
    eval_windows = sorted(df.loc[df["split"] != "train", "window_id"].unique())
    m = cfg["mlp"]

    rows = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        base = MLPNet(len(feats), len(classes), tuple(m["hidden"]), m["lr"],
                      m["epochs"], m["batch_size"], seed).fit(X_tr, y_tr)
        models = {"naive_ft": copy.deepcopy(base), "periodic": copy.deepcopy(base)}
        buf_X, buf_y = [], []
        print(f"seed {seed} | budget {budget} | base model trained")

        for i, w in enumerate(eval_windows):
            d = df[df["window_id"] == w]
            X, y_str = d[feats].values, d["family"].values

            # 1. TEST first: predict the window before seeing any labels
            for name, mod in models.items():
                y_pred = classes[mod.predict_proba(X).argmax(axis=1)]
                for metric, value in window_metrics(y_str, y_pred).items():
                    rows.append(dict(method=f"{name}_mlp_b{budget}", protocol="A",
                                     dataset="CIC-IDS2017", seed=seed,
                                     window=int(w), metric=metric, value=value))

            # 2. then TRAIN: a random fraction of this window gets labelled
            n_lab = max(1, int(budget * len(X)))
            idx = rng.choice(len(X), n_lab, replace=False)
            Xl, yl = X[idx], enc(y_str[idx])
            buf_X.append(Xl)
            buf_y.append(yl)

            nf = models["naive_ft"]                     # no replay, so it forgets
            nf.epochs = FT_EPOCHS
            nf.fit(Xl, yl)

            if (i + 1) % PERIOD == 0:                    # fixed schedule, with replay
                r = rng.choice(len(X_tr), min(REPLAY_TRAIN, len(X_tr)), replace=False)
                Xp = np.vstack([X_tr[r]] + buf_X)
                yp = np.concatenate([y_tr[r]] + buf_y)
                pm = models["periodic"]
                pm.epochs = PER_EPOCHS
                pm.fit(Xp, yp)

    os.makedirs("results", exist_ok=True)
    out = pd.DataFrame(rows)
    path = f"results/adaptive_baselines_b{budget}.csv"
    out.to_csv(path, index=False)
    print(f"\nsaved {path}")

    sel = out[out["metric"].isin(["macro_f1_true", "binary_f1", "attack_detection_rate", "attack_family_recall", "binary_fpr"])]
    print("\nMean over evaluation windows 7-19:")
    print(sel.groupby(["method", "metric"])["value"].mean().unstack().round(4))
    print("\nMacro-F1 per window:")
    print(out[out["metric"] == "macro_f1_true"].pivot_table(
        index="window", columns="method", values="value").round(3))


if __name__ == "__main__":
    main()