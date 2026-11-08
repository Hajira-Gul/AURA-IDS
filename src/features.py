import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from data import load_config, META

META_ALL = META + ["family"]


def main():
    cfg = load_config()
    skew_thr = cfg.get("skew_threshold", 3)
    clip = cfg.get("clip_value", 10)
    out_path = cfg.get("features_path", "data/processed/features.parquet")
    info_path = out_path.replace(".parquet", "_info.json")

    df = pd.read_parquet(cfg["processed_path"])
    # merge the three tiny web-attack labels into one family
    df["family"] = np.where(df["label"].str.startswith("Web Attack"),
                            "Web Attack", df["label"])
    feats = [c for c in df.columns if c not in META_ALL]
    is_train = (df["split"] == "train").values
    print(f"start: {len(feats)} features, {is_train.sum():,} train rows")

    # everything below is learned from the TRAIN windows only (no leakage)
    # 1. constant features
    train = df[is_train]
    std = train[feats].std()
    const = std.index[std == 0].tolist()
    feats = [f for f in feats if f not in const]
    print(f"after removing constant features: {len(feats)} (dropped {len(const)})")

    # 2. log transform for heavy-tailed, non-negative features
    skew = train[feats].skew()
    mn = train[feats].min()
    log_cols = [f for f in feats if mn[f] >= 0 and skew[f] > skew_thr]
    df[log_cols] = np.log1p(df[log_cols].clip(lower=0))
    print(f"log1p applied to {len(log_cols)} features")

    # 3. correlation filter
    train = df[is_train]
    corr = train[feats].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    corr_drop = [c for c in upper.columns if (upper[c] > cfg["corr_threshold"]).any()]
    feats = [f for f in feats if f not in corr_drop]
    print(f"after correlation filter (>{cfg['corr_threshold']}): {len(feats)} "
          f"(dropped {len(corr_drop)})")

    # 4. mutual information, only if still above the target size
    top_mi = None
    if len(feats) > cfg["n_features"]:
        sub = train.sample(min(50000, len(train)), random_state=cfg["prep_seed"])
        mi = mutual_info_classif(sub[feats], sub["family"],
                                 random_state=cfg["prep_seed"])
        mi = pd.Series(mi, index=feats).sort_values(ascending=False)
        keep = set(mi.index[:cfg["n_features"]])
        feats = [f for f in feats if f in keep]
        top_mi = mi.head(10).round(4).to_dict()
        print(f"after mutual information: {len(feats)}")
        print("top-10 MI features:", top_mi)

    # 5. standardise (fit on train only) and clip extreme values
    scaler = StandardScaler().fit(df.loc[is_train, feats])
    Z = scaler.transform(df[feats])
    for name in ["train", "val", "stream"]:
        m = (df["split"] == name).values
        print(f"share of values beyond +/-{clip} sd in {name}: "
              f"{(np.abs(Z[m]) > clip).mean():.4%}")
    Z = np.clip(Z, -clip, clip).astype("float32")
    assert not np.isnan(Z).any(), "NaN found after scaling"

    out = pd.DataFrame(Z, columns=feats)
    for c in META_ALL:
        out[c] = df[c].values
    out.to_parquet(out_path, index=False)

    info = {"features": feats, "log_cols": [c for c in log_cols if c in feats],
            "constant_dropped": const, "corr_dropped": corr_drop,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "families": sorted(df["family"].unique().tolist())}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nsaved {out_path} and {info_path}")

    pd.set_option("display.width", 200)
    print("\nFamily counts per split:")
    print(pd.crosstab(out["family"], out["split"]))


if __name__ == "__main__":
    main()