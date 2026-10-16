"""Numerical helpers retained from the historical attribute-free implementation.
No historical selection settings or experiment outputs are imported.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,precision_score,recall_score,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
MIN_SLICE_N=50
MIN_SLICE_EVENTS=10
TAIL_LEVELS=(.8,.9,.95)
GLOBAL_AUPRC_MARGIN=.005
GLOBAL_BRIER_MARGIN=.010
MODELS=['Logistic Regression','Random Forest','XGBoost','LightGBM']
BASELINE_ARMS=['Class-balanced','Focal-style','CVaR-DRO']
FEATURES=['age','bp_diastole','bp_systolic','heart_rate','respiratory_rate','spo2','temperature_c','abdominal_pain','chest_pain','back_pain','headache','other_pain','cardiovascular_history','dm','ckd']
from xgboost import XGBClassifier
def binary(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).gt(0).astype("int8")

def transport(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["age"] = pd.to_numeric(df.age, errors="coerce")
    out["bp_diastole"] = pd.to_numeric(df.bp_diastole, errors="coerce")
    out["bp_systolic"] = pd.to_numeric(df.bp_systolic, errors="coerce")
    out["heart_rate"] = pd.to_numeric(df.heart_rate_final, errors="coerce")
    out["respiratory_rate"] = pd.to_numeric(df.respiratory, errors="coerce")
    out["spo2"] = pd.to_numeric(df.spo2, errors="coerce")
    t = pd.to_numeric(df.temperature, errors="coerce")
    out["temperature_c"] = np.where(t.gt(60), (t - 32) * 5 / 9, t)
    for c in ["abdominal_pain", "chest_pain", "back_pain", "headache", "dm", "ckd"]:
        out[c] = binary(df[c])
    other = ["arm_pain", "breast_pain", "dental_pain", "ear_pain", "eye_pain", "pelvic_pain", "unclassified"]
    out["other_pain"] = pd.concat([binary(df[c]) for c in other], axis=1).max(axis=1).astype("int8")
    out["cardiovascular_history"] = np.maximum(binary(df.htn), binary(df.chd))
    return out[FEATURES]

def profile_groups(df: pd.DataFrame) -> np.ndarray:
    z = df.copy()
    for c in z:
        z[c] = pd.to_numeric(z[c], errors="coerce").round(6).fillna(-999999)
    return pd.util.hash_pandas_object(z, index=False).astype("uint64").to_numpy()

def model_factory(name: str, seed: int) -> Pipeline:
    if name == "Logistic Regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed)),
        ])
    if name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=180, max_depth=14, min_samples_leaf=20,
            max_features="sqrt", n_jobs=-1, random_state=seed,
        )
    elif name == "XGBoost":
        model = XGBClassifier(
            n_estimators=180, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=10, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss", n_jobs=-1,
            random_state=seed,
        )
    elif name == "LightGBM":
        model = LGBMClassifier(
            n_estimators=180, learning_rate=0.05, num_leaves=31,
            min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, n_jobs=-1, random_state=seed, verbosity=-1,
        )
    else:
        raise KeyError(name)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

def baseline_weights(arm: str, y: np.ndarray, oof: np.ndarray | None) -> np.ndarray:
    if arm == "Class-balanced":
        pos = len(y) / (2 * max(y.sum(), 1))
        neg = len(y) / (2 * max((y == 0).sum(), 1))
        return np.where(y == 1, pos, neg)
    p_true = np.where(y == 1, oof, 1 - oof)
    if arm == "Focal-style":
        w = np.maximum((1 - p_true) ** 2, 0.02)
        return np.clip(w / w.mean(), 0.10, 5.0)
    if arm == "CVaR-DRO":
        loss = -np.log(np.clip(p_true, 1e-6, 1))
        cutoff = np.quantile(loss, 0.80)
        w = np.where(loss >= cutoff, 4.0, 1.0)
        return w / w.mean()
    raise KeyError(arm)

def global_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pred = p >= threshold
    tn = ((y == 0) & ~pred).sum()
    fp = ((y == 0) & pred).sum()
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "Sensitivity": recall_score(y, pred, zero_division=0),
        "Specificity": tn / max(tn + fp, 1),
        "PPV": precision_score(y, pred, zero_division=0),
    }

def tail_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    out = {}
    for q in TAIL_LEVELS:
        cutoff = np.quantile(loss, q)
        tail = loss >= cutoff
        out[f"CVaR_{int(round((1-q)*100)):02d}pct"] = float(loss[tail].mean())
    k = max(1, int(np.ceil(0.10 * len(loss))))
    worst = np.argpartition(loss, -k)[-k:]
    out["worst10_loss_share"] = float(loss[worst].sum() / loss.sum())
    out["mean_logloss"] = float(loss.mean())
    return out

def slice_metrics(y: np.ndarray, p: np.ndarray, threshold: float,
                  masks: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        m = masks[name].to_numpy(bool)
        if m.sum() < MIN_SLICE_N or y[m].sum() < MIN_SLICE_EVENTS or np.unique(y[m]).size < 2:
            continue
        row = {"slice_name": name, "n": int(m.sum()), "events": int(y[m].sum())}
        row.update(global_metrics(y[m], p[m], threshold))
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_slice_endpoint(tab: pd.DataFrame, prefix: str) -> dict[str, float]:
    if tab.empty:
        return {f"{prefix}_slice_count": 0, f"{prefix}_worst_AUPRC": np.nan,
                f"{prefix}_mean_AUPRC": np.nan, f"{prefix}_worst_Brier": np.nan,
                f"{prefix}_worst_Sensitivity": np.nan}
    return {
        f"{prefix}_slice_count": len(tab),
        f"{prefix}_worst_AUPRC": float(tab.AUPRC.min()),
        f"{prefix}_mean_AUPRC": float(tab.AUPRC.mean()),
        f"{prefix}_worst_Brier": float(tab.Brier.max()),
        f"{prefix}_worst_Sensitivity": float(tab.Sensitivity.min()),
    }

def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best = (-np.inf, 0.5)
    for t in np.linspace(0.05, 0.95, 181):
        pred = p >= t
        sens = recall_score(y, pred, zero_division=0)
        spec = ((y == 0) & ~pred).sum() / max((y == 0).sum(), 1)
        if sens + spec - 1 > best[0]:
            best = (sens + spec - 1, float(t))
    return best[1]

def basic_ranking(y: np.ndarray, p: np.ndarray, sm: pd.DataFrame) -> pd.DataFrame:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    fn = ((y == 1) & (p < 0.5)).astype(float)
    dual = {"abdominal_pain_and_hr_missing", "ckd_and_spo2_missing", "unclassified_and_vitals_missing_ge_3"}
    rows = []
    for name in sm.columns:
        m = sm[name].to_numpy(bool)
        if m.sum() < MIN_SLICE_N or y[m].sum() < MIN_SLICE_EVENTS or np.unique(y[m]).size < 2:
            continue
        rows.append({"slice_name": name, "support": int(m.sum()),
                     "loss_lift": float(loss[m].mean() - loss.mean()),
                     "fn_lift": float(fn[m].mean() - fn.mean())})
    out = pd.DataFrame(rows)
    out["composite_harm"] = out.loss_lift + out.fn_lift
    lo, hi = out.composite_harm.min(), out.composite_harm.max()
    out["harm_norm"] = (out.composite_harm - lo) / (hi - lo) if hi > lo else 0.0
    out["dual_source"] = out.slice_name.isin(dual).astype(int)
    out["support_norm"] = out.support / out.support.max()
    out["rank_score"] = 0.6 * out.harm_norm + 0.3 * out.dual_source + 0.1 * out.support_norm
    return out[~out.slice_name.eq("fever_temp")].sort_values(["rank_score", "composite_harm"], ascending=False).reset_index(drop=True)

def basic_weights(y: np.ndarray, sm: pd.DataFrame, ranking: pd.DataFrame,
                  lam: float, topk: int) -> tuple[np.ndarray, list[str]]:
    w = np.ones(len(y), dtype=float)
    selected = ranking[ranking.composite_harm.gt(0)].head(topk)
    for row in selected.itertuples(index=False):
        m = sm[row.slice_name].to_numpy(bool)
        w[m & (y == 1)] += lam * max(float(row.harm_norm), 0.1)
    return np.clip(w, 1, 5), selected.slice_name.tolist()

def summarize_baselines(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "AUROC", "AUPRC", "Brier", "Sensitivity", "Specificity", "PPV",
        "CVaR_20pct", "CVaR_10pct", "CVaR_05pct", "worst10_loss_share",
        "registry_worst_AUPRC", "registry_mean_AUPRC", "registry_worst_Brier",
        "registry_worst_Sensitivity", "selected_worst_AUPRC", "selected_mean_AUPRC",
    ]
    summary = folds.groupby(["model", "arm"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    er = summary[summary.arm.eq("ERM")].set_index("model")
    summary["delta_registry_worst_AUPRC"] = summary.apply(lambda r: r.registry_worst_AUPRC_mean - er.loc[r.model, "registry_worst_AUPRC_mean"], axis=1)
    summary["delta_CVaR_10pct"] = summary.apply(lambda r: r.CVaR_10pct_mean - er.loc[r.model, "CVaR_10pct_mean"], axis=1)
    summary["delta_global_AUPRC"] = summary.apply(lambda r: r.AUPRC_mean - er.loc[r.model, "AUPRC_mean"], axis=1)
    summary["delta_global_Brier"] = summary.apply(lambda r: r.Brier_mean - er.loc[r.model, "Brier_mean"], axis=1)
    summary["global_noninferiority_pass"] = (
        summary.delta_global_AUPRC.ge(-GLOBAL_AUPRC_MARGIN)
        & summary.delta_global_Brier.le(GLOBAL_BRIER_MARGIN)
    )
    summary["primary_success"] = (
        summary.delta_registry_worst_AUPRC.gt(0)
        & summary.delta_CVaR_10pct.lt(0)
        & summary.global_noninferiority_pass
    )
    return summary
