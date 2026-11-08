import json
import os
import sys
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from data import load_config
from models import StaticModel
from metrics import evaluate

METHODS = ["logreg", "lightgbm", "mlp"]


def window_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    union = sorted(set(y_true) | set(y_pred))
    m = evaluate(y_true, y_pred, union)
    out = {k: m[k] for k in ["macro_f1", "macro_precision", "fpr", "accuracy"]}

    # macro-F1 over the classes that are truly present, so wrong guesses
    # do not add extra zero-F1 classes
    present = sorted(set(y_true))
    out["macro_f1_true"] = float(f1_score(y_true, y_pred, labels=present,
                                          average="macro", zero_division=0))

    atk = y_true != "BENIGN"
    flagged = y_pred != "BENIGN"
    tp = int((flagged & atk).sum())
    fp = int((flagged & ~atk).sum())
    fn = int((~flagged & atk).sum())
    out["attack_detection_rate"] = tp / atk.sum() if atk.any() else np.nan
    out["binary_fpr"] = fp / max((~atk).sum(), 1)
    out["binary_f1"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    # attacks that were flagged AND given the correct family
    out["attack_family_recall"] = float((y_pred[atk] == y_true[atk]).mean()) if atk.any() else np.nan
    return out

def main():
    cfg = load_config()
    seeds = [int(s) for s in sys.argv[1:]] or [0]
    df = pd.read_parquet(cfg["features_path"])
    info = json.load(open(cfg["features_path"].replace(".parquet", "_info.json")))
    feats = info["features"]

    train = df[df["split"] == "train"]
    X_tr, y_tr = train[feats].values, train["family"].values
    eval_windows = sorted(df.loc[df["split"] != "train", "window_id"].unique())

    rows = []
    for seed in seeds:
        for name in METHODS:
            print(f"seed {seed} | training {name} ...")
            model = StaticModel(name, cfg, seed).fit(X_tr, y_tr)
            for w in eval_windows:
                d = df[df["window_id"] == w]
                y_pred = model.predict(d[feats].values)
                res = window_metrics(d["family"].values, y_pred)
                for metric, value in res.items():
                    rows.append(dict(method=f"static_{name}", protocol="A",
                                     dataset="CIC-IDS2017", seed=seed,
                                     window=int(w), metric=metric, value=value))

    os.makedirs("results", exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/static_baselines.csv", index=False)
    print("\nsaved results/static_baselines.csv")

    piv = out[out["metric"].isin(["macro_f1_true", "binary_f1", "attack_detection_rate", "attack_family_recall", "binary_fpr"])]
    print("\nMean over evaluation windows 7-19:")
    print(piv.groupby(["method", "metric"])["value"].mean().unstack().round(4))
    print("\nMacro-F1 per window:")
    print(out[out["metric"] == "macro_f1_true"].pivot_table(
        index="window", columns="method", values="value").round(3))


if __name__ == "__main__":
    main()