import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from river import drift as rdrift
from data import load_config
from models import MLPNet
from run_baselines import window_metrics

warnings.filterwarnings("ignore")

# Each variant switches components on/off. These double as ablations and baselines.
VARIANTS = {
    "full":          dict(unknown=True,  drift="both",     select="unc",  replay=True),
    "no_unknown":    dict(unknown=False, drift="both",     select="unc",  replay=True),
    "periodic_trig": dict(unknown=True,  drift="periodic", select="unc",  replay=True),
    "random_select": dict(unknown=True,  drift="both",     select="rand", replay=True),
    "no_replay":     dict(unknown=True,  drift="both",     select="unc",  replay=False),
    "ks_only":       dict(unknown=True,  drift="ks",       select="unc",  replay=True),
    "adwin_only":    dict(unknown=True,  drift="adwin",    select="unc",  replay=True),
    "unknown_only":  dict(unknown=True,  drift="never",    select="unc",  replay=True),
    "ssf_like":      dict(unknown=False, drift="ks",       select="unc",  replay=True),
}
PERIOD = 3            # window period for the fixed-schedule variant
EPOCHS = 5            # fine-tuning epochs per adaptation
REPLAY_TRAIN = 20000  # train samples kept in the replay pool
MONITOR_FRAC = 0.005  # labelled monitoring sample per window (feeds ADWIN)
RANDOM_FRAC = 0.3     # share of each label budget chosen at random
KS_D = 0.2            # minimum KS statistic for a feature to count as drifted
KS_N = 2000           # rows per KS sample


def strat_sample(y, per_class, rng):
    idx = []
    for c in np.unique(y):
        w = np.where(y == c)[0]
        idx.append(rng.choice(w, min(per_class, len(w)), replace=False))
    return np.concatenate(idx)


class CentroidDetector:
    """Unknown-attack detector: distance to class centroids in the hidden layer."""

    def fit(self, model, X, y, rng, q_radius=0.95, q_tau=0.99):
        idx = rng.permutation(len(X))
        cut = int(0.7 * len(X))
        fit_i, cal_i = idx[:cut], idx[cut:]
        E = model.embed(X)
        Ef, yf = E[fit_i], y[fit_i]
        mus, rads, cnts = [], [], []
        for c in np.unique(yf):
            e = Ef[yf == c]
            mu = e.mean(0)
            mus.append(mu)
            rads.append(np.quantile(np.linalg.norm(e - mu, axis=1), q_radius))
            cnts.append(len(e))
        rads, cnts = np.array(rads), np.array(cnts)
        big = cnts >= 20
        fallback = np.median(rads[big]) if big.any() else 1.0
        self.mu = np.array(mus)
        self.rad = np.maximum(np.where(big, rads, fallback), 1e-6)
        # threshold calibrated on held-out known samples
        self.tau = np.quantile(self._score(E[cal_i]), q_tau)

    def _score(self, E):
        return (cdist(E, self.mu) / self.rad).min(axis=1)

    def flag(self, model, X):
        s = self._score(model.embed(X))
        return s, s > self.tau


def ks_alarm(ref, cur, alpha, d_min, frac_thr):
    hits = 0
    for j in range(ref.shape[1]):
        stat, p = ks_2samp(ref[:, j], cur[:, j])
        if p < alpha and stat > d_min:
            hits += 1
    frac = hits / ref.shape[1]
    return frac, frac >= frac_thr


def adwin_fired(a):
    f = getattr(a, "drift_detected", None)
    return bool(a.change_detected if f is None else f)


def select(proba, flagged, n_lab, random_frac, rng):
    n = len(proba)
    n_lab = min(n_lab, n)
    if random_frac >= 1:
        return rng.choice(n, n_lab, replace=False)
    ent = -(proba * np.log(proba + 1e-12)).sum(1) / np.log(proba.shape[1])
    score = ent + (flagged.astype(float) if flagged is not None else 0.0)
    n_rand = int(round(random_frac * n_lab))
    top = np.argsort(-score)[:n_lab - n_rand]
    rest = np.setdiff1d(np.arange(n), top)
    return np.concatenate([top, rng.choice(rest, n_rand, replace=False)])


def perm_importance(model, X, y, rng, repeats=2):
    base = balanced_accuracy_score(y, model.predict_proba(X).argmax(1))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(X)), j]
            drops.append(base - balanced_accuracy_score(
                y, model.predict_proba(Xp).argmax(1)))
        imp[j] = np.mean(drops)
    return imp


def calibrate_ks(df, feats, cfg):
    """KS alarm settings from pure-benign training windows only (no stream peeking)."""
    rng = np.random.RandomState(0)
    pure = [w for w in range(cfg["train_windows"])
            if (df.loc[df["window_id"] == w, "family"] == "BENIGN").all()]

    def samp(w):
        a = df.loc[df["window_id"] == w, feats].values
        return a[rng.choice(len(a), min(KS_N, len(a)), replace=False)]

    stats = []
    for a, b in zip(pure[:-1], pure[1:]):
        if b != a + 1:
            continue
        A, B = samp(a), samp(b)
        stats.append([ks_2samp(A[:, j], B[:, j]).statistic for j in range(len(feats))])
    stats = np.array(stats)                       # pairs x features
    d_min = float(np.quantile(stats, 0.95))       # effect-size cutoff
    frac = (stats > d_min).mean(axis=1)           # benign pairs above the cutoff
    frac_thr = float(min(0.9, max(0.10, 1.5 * frac.max())))
    print(f"KS calibration on benign windows {pure}: median stat "
          f"{np.median(stats):.3f}, d_min (95th pct) {d_min:.3f}, "
          f"benign fractions above d_min {np.round(frac, 3).tolist()} "
          f"-> alarm when >= {frac_thr:.3f} of features exceed d_min")
    return d_min, frac_thr


def run_seed(seed, variant, budget, cfg, df, feats, classes, ks_thr):
    v = VARIANTS[variant]
    rng = np.random.RandomState(seed)
    enc = lambda y: np.searchsorted(classes, y)
    m = cfg["mlp"]
    train = df[df["split"] == "train"]
    X_tr, y_tr = train[feats].values, enc(train["family"].values)
    model = MLPNet(len(feats), len(classes), tuple(m["hidden"]), m["lr"],
                   m["epochs"], m["batch_size"], seed).fit(X_tr, y_tr)

    seen = set(y_tr.tolist())
    buf_X, buf_y = [], []
    r = rng.choice(len(X_tr), min(REPLAY_TRAIN, len(X_tr)), replace=False)
    rep_X, rep_y = X_tr[r], y_tr[r]

    det = None
    if v["unknown"]:
        det = CentroidDetector()
        s = strat_sample(rep_y, 8000, rng)
        det.fit(model, rep_X[s], rep_y[s], rng)

    ref = df.loc[df["window_id"] == cfg["train_windows"] - 1, feats].values
    ref = ref[rng.choice(len(ref), min(KS_N, len(ref)), replace=False)]
    adwin = rdrift.ADWIN(delta=cfg["drift"]["adwin_delta"])

    rows = []
    eval_windows = sorted(df.loc[df["split"] != "train", "window_id"].unique())
    for i, w in enumerate(eval_windows):
        d = df[df["window_id"] == w]
        X, y_str = d[feats].values, d["family"].values
        y_int = enc(y_str)

        def add(metric, value):
            rows.append(dict(method=f"{variant}_b{budget}", protocol="A",
                             dataset="CIC-IDS2017", seed=seed, window=int(w),
                             metric=metric, value=value))

        # 1. TEST first
        proba = model.predict_proba(X)
        pred = classes[proba.argmax(1)]
        flagged = None
        if det is not None:
            score, flagged = det.flag(model, X)
            pred = np.where(flagged, "UNKNOWN", pred)
            add("unknown_rate", float(flagged.mean()))
            novel = np.isin(y_int, list(seen), invert=True)
            if novel.any() and (~novel).any():
                add("unknown_auroc", float(roc_auc_score(novel, score)))
        for k, val in window_metrics(y_str, pred).items():
            add(k, val)

        # 2. drift detection
        labels_used = 0
        ks_frac, ks_flag, ad_flag = np.nan, False, False
        cur = X[rng.choice(len(X), min(KS_N, len(X)), replace=False)]
        if v["drift"] in ("both", "ks"):
            ks_frac, ks_flag = ks_alarm(ref, cur, cfg["drift"]["ks_alpha"], ks_thr[0], ks_thr[1])
        if v["drift"] in ("both", "adwin"):
            n_mon = max(1, int(MONITOR_FRAC * len(X)))
            labels_used += n_mon
            for j in rng.choice(len(X), n_mon, replace=False):
                adwin.update(float(pred[j] != y_str[j]))
                if adwin_fired(adwin):
                    ad_flag = True
        if v["drift"] == "periodic":
            trigger = (i + 1) % PERIOD == 0
        elif v["drift"] == "never":
            trigger = False
        else:
            trigger = ks_flag or ad_flag

        # 3. adapt
        adapt_sec = 0.0
        if trigger:
            n_lab = max(1, int(budget * len(X)))
            rf = RANDOM_FRAC if v["select"] == "unc" else 1.0
            idx = select(proba, flagged, n_lab, rf, rng)
            labels_used += len(idx)
            Xl, yl = X[idx], y_int[idx]
            buf_X.append(Xl)
            buf_y.append(yl)
            seen |= set(yl.tolist())
            pool_X = np.vstack([rep_X] + buf_X)
            pool_y = np.concatenate([rep_y] + buf_y)

            es = strat_sample(pool_y, 300, rng)
            eX, ey = pool_X[es], pool_y[es]
            imp_before = perm_importance(model, eX, ey, rng)

            t0 = time.perf_counter()
            Xt, yt = (pool_X, pool_y) if v["replay"] else (Xl, yl)
            model.epochs = EPOCHS
            model.fit(Xt, yt)
            if det is not None:
                s = strat_sample(pool_y, 8000, rng)
                det.fit(model, pool_X[s], pool_y[s], rng)
            adapt_sec = time.perf_counter() - t0

            imp_after = perm_importance(model, eX, ey, rng)
            rho = spearmanr(imp_before, imp_after).correlation
            top_b = set(np.argsort(-imp_before)[:10])
            top_a = set(np.argsort(-imp_after)[:10])
            add("expl_spearman", float(rho))
            add("expl_top10_overlap", len(top_b & top_a) / 10)

            adwin = rdrift.ADWIN(delta=cfg["drift"]["adwin_delta"])
            ref = cur

        add("adapted", float(trigger))
        add("adapt_seconds", adapt_sec)
        add("labels_used", float(labels_used))
        add("buffer_size", float(sum(len(b) for b in buf_y)))
        add("ks_frac", ks_frac)
        add("ks_alarm", float(ks_flag))
        add("adwin_alarm", float(ad_flag))
    return rows


def main():
    cfg = load_config()
    variant = sys.argv[1] if len(sys.argv) > 1 else "full"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    seeds = [int(s) for s in sys.argv[3:]] or [0]
    assert variant in VARIANTS, f"unknown variant; choose from {list(VARIANTS)}"

    df = pd.read_parquet(cfg["features_path"])
    feats = json.load(open(cfg["features_path"].replace(".parquet", "_info.json")))["features"]
    classes = np.array(sorted(df["family"].unique()))
    ks_thr = calibrate_ks(df, feats, cfg)

    rows = []
    for seed in seeds:
        print(f"{variant} | budget {budget} | seed {seed}")
        rows += run_seed(seed, variant, budget, cfg, df, feats, classes, ks_thr)

    os.makedirs("results", exist_ok=True)
    out = pd.DataFrame(rows)
    path = f"results/proposed_{variant}_b{budget}.csv"
    out.to_csv(path, index=False)
    print(f"\nsaved {path}")

    keep = ["macro_f1_true", "binary_f1", "attack_detection_rate",
            "attack_family_recall", "binary_fpr", "unknown_auroc",
            "expl_spearman", "expl_top10_overlap"]
    print("\nMean over evaluation windows 7-19:")
    print(out[out["metric"].isin(keep)].groupby(["method", "metric"])["value"]
          .mean().unstack().round(4).T)
    cost = (out[out["metric"].isin(["labels_used", "adapted", "adapt_seconds"])]
            .groupby(["method", "seed", "metric"])["value"].sum()
            .groupby(["method", "metric"]).mean().unstack().round(2))
    print("\nTotals per run (labels used, adaptation events, retraining seconds):")
    print(cost)

    first = out[out["seed"] == seeds[0]]
    cols = ["ks_frac", "ks_alarm", "adwin_alarm", "adapted", "unknown_rate",
            "unknown_auroc", "macro_f1_true", "attack_family_recall"]
    tab = first[first["metric"].isin(cols)].pivot_table(
        index="window", columns="metric", values="value")
    print("\nPer-window signals (first seed):")
    print(tab[[c for c in cols if c in tab.columns]].round(3))


if __name__ == "__main__":
    main()