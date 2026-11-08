import os
import numpy as np
import pandas as pd
import yaml

# Chronological order of the capture days (file order is the only time info we have)
DAY_FILES = [
    "Monday-WorkingHours",
    "Tuesday-WorkingHours",
    "Wednesday-workingHours",
    "Thursday-WorkingHours-Morning-WebAttacks",
    "Thursday-WorkingHours-Afternoon-Infilteration",
    "Friday-WorkingHours-Morning",
    "Friday-WorkingHours-Afternoon-PortScan",
    "Friday-WorkingHours-Afternoon-DDos",
]
META = ["label", "window_id", "day", "split"]  # never used as model features


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw(cfg):
    frames = []
    for name in DAY_FILES:
        path = os.path.join(cfg["raw_dir"], f"{name}.pcap_ISCX.csv")
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
        df.columns = df.columns.str.strip()
        df["day"] = name.split("-")[0]
        frames.append(df)
        print(f"loaded {name}: {len(df):,} rows")
    df = pd.concat(frames, ignore_index=True)  # keeps day order, then row order
    return df


def clean(df, cfg):
    df = df.rename(columns={"Label": "label"})
    # fix garbled characters in labels, e.g. "Web Attack � Brute Force"
    df["label"] = (df["label"].astype(str).str.strip()
                   .str.replace(r"[^\x00-\x7F]+", "-", regex=True))
    # drop duplicate column and (optionally) destination port
    drop = [c for c in ["Fwd Header Length.1"] if c in df.columns]
    if cfg["drop_destination_port"] and "Destination Port" in df.columns:
        drop.append("Destination Port")
    df = df.drop(columns=drop)

    feats = [c for c in df.columns if c not in META]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)

    n0 = len(df)
    df = df.dropna(subset=feats)
    print(f"dropped NaN/inf rows: {n0 - len(df):,}")
    n1 = len(df)
    df = df.drop_duplicates(subset=feats + ["label"], keep="first")
    print(f"dropped duplicate rows: {n1 - len(df):,}")

    df[feats] = df[feats].astype("float32")
    return df.reset_index(drop=True), feats


def add_windows(df, cfg):
    n, k = len(df), cfg["n_windows"]
    df["window_id"] = np.minimum(np.arange(n) * k // n, k - 1)  # 0 .. k-1
    t, v = cfg["train_windows"], cfg["val_windows"]
    df["split"] = np.where(df["window_id"] < t, "train",
                  np.where(df["window_id"] < t + v, "val", "stream"))
    return df


def subsample(df, cfg):
    """Cap big classes per window; rare classes (under the cap) are kept in full."""
    rng = np.random.RandomState(cfg["prep_seed"])
    parts = []
    for (w, lab), g in df.groupby(["window_id", "label"], sort=False):
        cap = cfg["max_benign_per_window"] if lab == "BENIGN" else cfg["max_big_attack_per_window"]
        if len(g) > cap:
            g = g.iloc[rng.choice(len(g), cap, replace=False)]
        parts.append(g)
    out = pd.concat(parts).sort_values(["window_id"], kind="stable")
    return out.reset_index(drop=True)


def main():
    cfg = load_config()
    df = load_raw(cfg)
    df, feats = clean(df, cfg)
    print(f"after cleaning: {len(df):,} rows, {len(feats)} features")
    df = add_windows(df, cfg)
    df = subsample(df, cfg)
    os.makedirs(os.path.dirname(cfg["processed_path"]), exist_ok=True)
    df.to_parquet(cfg["processed_path"], index=False)
    print(f"saved {len(df):,} rows to {cfg['processed_path']}")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nClass counts per window:")
    print(pd.crosstab(df["window_id"], df["label"]))
    print("\nSplit sizes:")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()