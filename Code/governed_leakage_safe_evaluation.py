from __future__ import annotations

import hashlib
import gc
import json
import time
from pathlib import Path
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit, logit
from scipy.optimize import brentq
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1] / "basic_slice_prem"
KAGGLE = ROOT / "Data" / "Raw" / "triage_cleaned_working.csv"
MEMBERSHIP = ROOT / "Notebooks" / "slice_membership.csv"
RIPAS = Path(os.environ["RIPAS_DATA_PATH"])
OUT = ROOT / "Governed_Leakage_Safe_Results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260828
OUTER_FOLDS = 5
RANKING_FOLDS = 3
N_BOOT = 2000
GLOBAL_AUPRC_MARGIN = 0.005
GLOBAL_BRIER_MARGIN = 0.010
MIN_SLICE_N = 50
MIN_SLICE_EVENTS = 10
FEATURES = [
    "age", "bp_diastole", "bp_systolic", "heart_rate", "respiratory_rate",
    "spo2", "temperature_c", "abdominal_pain", "chest_pain", "back_pain",
    "headache", "other_pain", "cardiovascular_history", "dm", "ckd",
]
DUAL_SUPPORTED = {
    "abdominal_pain_and_hr_missing", "ckd_and_spo2_missing",
    "unclassified_and_vitals_missing_ge_3",
}
EXCLUDED_ARTIFACT_SLICES = {"fever_temp"}
CONFIGS = [(0.0, 0), (0.5, 3), (1.0, 3), (2.0, 3), (0.5, 5), (1.0, 5), (2.0, 5)]


def binary(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).gt(0).astype("int8")


def kaggle_transport(df):
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
    others = ["arm_pain", "breast_pain", "dental_pain", "ear_pain", "eye_pain", "pelvic_pain", "unclassified"]
    out["other_pain"] = pd.concat([binary(df[c]) for c in others], axis=1).max(axis=1).astype("int8")
    out["cardiovascular_history"] = np.maximum(binary(df.htn), binary(df.chd))
    return out[FEATURES]


def ripas_transport(df):
    out = pd.DataFrame(index=df.index)
    mapping = {"age":"age", "bp_diastole":"bp_diastole", "bp_systolic":"bp_systolic",
               "heart_rate":"pulse_rate", "respiratory_rate":"respiratory_rate",
               "spo2":"spo2", "temperature_c":"temperature"}
    for target, source in mapping.items():
        out[target] = pd.to_numeric(df[source], errors="coerce")
    pain = df.site_of_pain.fillna("").astype(str).str.strip().str.upper()
    out["abdominal_pain"] = pain.str.contains("ABDOMINAL|PELVIC", regex=True).astype("int8")
    out["chest_pain"] = pain.str.contains("CHEST", regex=True).astype("int8")
    out["back_pain"] = pain.str.contains("BACK", regex=True).astype("int8")
    out["headache"] = pain.str.contains("HEADACHE", regex=True).astype("int8")
    out["other_pain"] = (~out[["abdominal_pain", "chest_pain", "back_pain", "headache"]].max(axis=1).astype(bool)).astype("int8")
    hx = df.htn_chd.fillna("NO").astype(str).str.upper().str.replace(" ", "", regex=False)
    out["cardiovascular_history"] = hx.ne("NO").astype("int8")
    out["dm"] = df.dm.fillna("NO").astype(str).str.upper().ne("NO").astype("int8")
    out["ckd"] = binary(df.ckd)
    return out[FEATURES]


def profile_groups(df):
    normalized = df.copy()
    for c in normalized:
        if pd.api.types.is_numeric_dtype(normalized[c]):
            normalized[c] = pd.to_numeric(normalized[c], errors="coerce").round(6).fillna(-999999)
        else:
            normalized[c] = normalized[c].fillna("<NA>").astype(str).str.strip().str.upper()
    return pd.util.hash_pandas_object(normalized, index=False).astype("uint64").to_numpy()


def group_fold_ids(groups, n_splits, seed):
    """Memory-bounded deterministic assignment; equal group hashes never cross folds."""
    g = np.asarray(groups, dtype="uint64")
    mixed = (g ^ np.uint64(seed)) * np.uint64(11400714819323198485)
    return (mixed % np.uint64(n_splits)).astype("int8")


def grouped_splits(groups, n_splits, seed):
    folds = group_fold_ids(groups, n_splits, seed)
    for fold in range(n_splits):
        te = np.flatnonzero(folds == fold); tr = np.flatnonzero(folds != fold)
        yield tr, te


def model_factory(name, seed):
    if name == "Logistic Regression":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("model", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed))])
    if name == "Random Forest":
        m = RandomForestClassifier(n_estimators=180, max_depth=14, min_samples_leaf=20,
                                   max_features="sqrt", n_jobs=1, random_state=seed)
    elif name == "XGBoost":
        m = XGBClassifier(n_estimators=180, max_depth=5, learning_rate=.05, subsample=.8,
                          colsample_bytree=.8, min_child_weight=10, reg_lambda=1.0,
                          objective="binary:logistic", eval_metric="logloss", n_jobs=1,
                          random_state=seed)
    elif name == "LightGBM":
        m = LGBMClassifier(n_estimators=180, learning_rate=.05, num_leaves=31,
                           min_child_samples=40, subsample=.8, colsample_bytree=.8,
                           reg_lambda=1.0, n_jobs=1, random_state=seed, verbosity=-1)
    else:
        raise KeyError(name)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", m)])


def fit_model(name, X, y, seed, weights=None):
    m = model_factory(name, seed)
    kwargs = {} if weights is None else {"model__sample_weight": weights}
    return m.fit(X, y, **kwargs)


def metric_row(y, p, threshold):
    y = np.asarray(y); p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6); pred = p >= threshold
    tn = int(((y == 0) & ~pred).sum()); fp = int(((y == 0) & pred).sum())
    return {
        "AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p), "LogLoss": log_loss(y, p),
        "Sensitivity": recall_score(y, pred, zero_division=0),
        "Specificity": tn / max(tn + fp, 1), "PPV": precision_score(y, pred, zero_division=0),
    }


def calibration_stats(y, p):
    y = np.asarray(y); p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6); z = logit(p).reshape(-1, 1)
    m = LogisticRegression(C=1e6, solver="lbfgs").fit(z, y)
    bins = pd.qcut(p, q=min(10, max(3, len(y) // 100)), duplicates="drop")
    frame = pd.DataFrame({"y": y, "p": p, "bin": bins})
    cal = frame.groupby("bin", observed=True).agg(obs=("y", "mean"), pred=("p", "mean"), n=("y", "size"))
    ece = np.average(np.abs(cal.obs - cal.pred), weights=cal.n)
    return {"calibration_intercept": float(m.intercept_[0]), "calibration_slope": float(m.coef_[0, 0]), "ECE": float(ece)}


def choose_threshold(y, p):
    best = (-1, .5)
    for t in np.linspace(.05, .95, 181):
        pred = p >= t
        sens = recall_score(y, pred, zero_division=0)
        spec = ((y == 0) & ~pred).sum() / max((y == 0).sum(), 1)
        if sens + spec - 1 > best[0]:
            best = (sens + spec - 1, float(t))
    return best[1]


def grouped_oof(name, X, y, groups, n_splits, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        m = fit_model(name, X.iloc[tr], y[tr], seed + fold)
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def rank_slices(y, p, sm):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6); y = np.asarray(y)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)); pred = p >= .5
    fn = ((y == 1) & ~pred).astype(float)
    rows = []
    for s in sm.columns:
        m = sm[s].to_numpy(bool); events = int(y[m].sum())
        if m.sum() < MIN_SLICE_N or events < MIN_SLICE_EVENTS or len(np.unique(y[m])) < 2:
            continue
        rows.append({"slice_name": s, "support": int(m.sum()), "events": events,
                     "loss_lift": float(loss[m].mean() - loss.mean()),
                     "fn_lift": float(fn[m].mean() - fn.mean())})
    r = pd.DataFrame(rows)
    r["composite_harm"] = r.loss_lift + r.fn_lift
    lo, hi = r.composite_harm.min(), r.composite_harm.max()
    r["harm_norm"] = (r.composite_harm - lo) / (hi - lo) if hi > lo else 0.0
    r["dual_source"] = r.slice_name.isin(DUAL_SUPPORTED).astype(int)
    r["support_norm"] = r.support / r.support.max()
    r["rank_score"] = .6 * r.harm_norm + .3 * r.dual_source + .1 * r.support_norm
    return r[~r.slice_name.isin(EXCLUDED_ARTIFACT_SLICES)].sort_values(["rank_score", "composite_harm"], ascending=False).reset_index(drop=True)


def positive_target_weights(y, sm, ranking, lam, topk):
    w = np.ones(len(y), dtype=float)
    if lam <= 0 or topk <= 0:
        return w, []
    selected = ranking[ranking.composite_harm > 0].head(topk)
    for row in selected.itertuples(index=False):
        member = sm[row.slice_name].to_numpy(bool)
        w[member & (y == 1)] += lam * max(float(row.harm_norm), .1)
    return np.clip(w, 1, 5), selected.slice_name.tolist()


def target_utility(y, p, sm, selected, threshold):
    vals = []
    for s in selected:
        m = sm[s].to_numpy(bool)
        if m.sum() >= MIN_SLICE_N and y[m].sum() >= MIN_SLICE_EVENTS:
            vals.append(recall_score(y[m], p[m] >= threshold, zero_division=0))
    return float(np.mean(vals)) if vals else np.nan


def select_repair(name, X, y, groups, sm, seed):
    split = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed)
    fit_idx, val_idx = next(split.split(X, y, groups))
    base_oof = grouped_oof(name, X.iloc[fit_idx].reset_index(drop=True), y[fit_idx], groups[fit_idx], RANKING_FOLDS, seed)
    ranking = rank_slices(y[fit_idx], base_oof, sm.iloc[fit_idx].reset_index(drop=True))
    threshold = choose_threshold(y[fit_idx], base_oof)
    base_model = fit_model(name, X.iloc[fit_idx], y[fit_idx], seed)
    base_p = base_model.predict_proba(X.iloc[val_idx])[:, 1]
    base_met = metric_row(y[val_idx], base_p, threshold)
    del base_model, base_p
    gc.collect()
    rows = []
    for lam, topk in CONFIGS:
        w, selected = positive_target_weights(y[fit_idx], sm.iloc[fit_idx].reset_index(drop=True), ranking, lam, topk)
        model = fit_model(name, X.iloc[fit_idx], y[fit_idx], seed + int(lam * 10) + topk, None if lam == 0 else w)
        p = model.predict_proba(X.iloc[val_idx])[:, 1]; met = metric_row(y[val_idx], p, threshold)
        util = target_utility(y[val_idx], p, sm.iloc[val_idx].reset_index(drop=True), selected, threshold)
        eligible = (met["AUPRC"] >= base_met["AUPRC"] - GLOBAL_AUPRC_MARGIN and
                    met["Brier"] <= base_met["Brier"] + GLOBAL_BRIER_MARGIN)
        rows.append({"lambda": lam, "top_k": topk, "eligible": eligible, "target_sensitivity": util,
                     "AUPRC": met["AUPRC"], "Brier": met["Brier"], "Sensitivity": met["Sensitivity"],
                     "selected_slices": "|".join(selected)})
        del model, p
        gc.collect()
    tab = pd.DataFrame(rows)
    eligible = tab[tab.eligible & tab.target_sensitivity.notna() & (tab["lambda"] > 0)]
    if eligible.empty:
        choice = tab.iloc[0]
    else:
        choice = eligible.sort_values(["target_sensitivity", "AUPRC", "Brier"], ascending=[False, False, True]).iloc[0]
    return float(choice["lambda"]), int(choice.top_k), threshold, tab


def prevalence_standardized_auprc(y, p, target_prevalence):
    observed = np.mean(y)
    wp = target_prevalence / max(observed, 1e-12)
    wn = (1 - target_prevalence) / max(1 - observed, 1e-12)
    return average_precision_score(y, p, sample_weight=np.where(y == 1, wp, wn))


def intercept_shift(y, p):
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    f = lambda a: expit(z + a).mean() - y.mean()
    return brentq(f, -20, 20)


def crossfit_recalibration(y, p, groups, seed):
    out_i = np.zeros(len(y)); out_is = np.zeros(len(y))
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(z.reshape(-1, 1), y, groups):
        a = intercept_shift(y[tr], p[tr]); out_i[te] = expit(z[te] + a)
        m = LogisticRegression(C=1e6, solver="lbfgs").fit(z[tr].reshape(-1, 1), y[tr])
        out_is[te] = m.predict_proba(z[te].reshape(-1, 1))[:, 1]
    return out_i, out_is


def cluster_bootstrap(y, p0, p1, groups, seed, target_prev, threshold):
    rng = np.random.default_rng(seed); uniq = np.unique(groups); lookup = {g: np.where(groups == g)[0] for g in uniq}
    values = {"AUPRC": [], "AUPRC_standardized": [], "Brier": [], "Sensitivity": []}
    for _ in range(N_BOOT):
        sampled = rng.choice(uniq, len(uniq), replace=True); idx = np.concatenate([lookup[g] for g in sampled]); yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        values["AUPRC"].append(average_precision_score(yy, p1[idx]) - average_precision_score(yy, p0[idx]))
        values["AUPRC_standardized"].append(prevalence_standardized_auprc(yy, p1[idx], target_prev) - prevalence_standardized_auprc(yy, p0[idx], target_prev))
        values["Brier"].append(brier_score_loss(yy, p1[idx]) - brier_score_loss(yy, p0[idx]))
        values["Sensitivity"].append(recall_score(yy, p1[idx] >= threshold, zero_division=0) - recall_score(yy, p0[idx] >= threshold, zero_division=0))
    rows = []
    for metric, vals in values.items():
        a = np.asarray(vals); rows.append({"metric": metric, "mean_delta": a.mean(), "ci_low": np.quantile(a, .025),
                                          "ci_high": np.quantile(a, .975), "p_two_sided": min(1.0, 2 * min(np.mean(a <= 0), np.mean(a >= 0))),
                                          "bootstrap_success": len(a), "cluster_unit": "exact_predictor_profile"})
    return rows


def external_slices(X):
    return {
        "abdominal_pain": X.abdominal_pain.eq(1).to_numpy(), "chest_pain": X.chest_pain.eq(1).to_numpy(),
        "back_pain": X.back_pain.eq(1).to_numpy(), "cardiovascular_history": X.cardiovascular_history.eq(1).to_numpy(),
        "diabetes": X.dm.eq(1).to_numpy(), "ckd": X.ckd.eq(1).to_numpy(),
        "tachycardia": X.heart_rate.gt(100).to_numpy(), "tachypnea": X.respiratory_rate.gt(20).to_numpy(),
        "hypoxia": X.spo2.lt(95).to_numpy(),
    }


def subgroup_rows(model, arm, y, p, threshold, ripas_raw, Xr):
    groups = {
        "sex_female": ripas_raw.gender.fillna("").astype(str).str.upper().str.startswith("F").to_numpy(),
        "sex_male": ripas_raw.gender.fillna("").astype(str).str.upper().str.startswith("M").to_numpy(),
        "age_18_39": Xr.age.between(18, 39).to_numpy(), "age_40_64": Xr.age.between(40, 64).to_numpy(),
        "age_65_plus": Xr.age.ge(65).to_numpy(), **external_slices(Xr),
    }
    rows = []
    for g, mask in groups.items():
        if mask.sum() < 30 or y[mask].sum() < 10 or len(np.unique(y[mask])) < 2:
            continue
        row = {"model": model, "arm": arm, "group": g, "n": int(mask.sum()), "events": int(y[mask].sum()), "prevalence": float(y[mask].mean())}
        row.update(metric_row(y[mask], p[mask], threshold)); row.update(calibration_stats(y[mask], p[mask])); rows.append(row)
    return rows


def main():
    start = time.time(); raw = pd.read_csv(KAGGLE, low_memory=False); X = kaggle_transport(raw); y = raw.y.to_numpy(int)
    sm = pd.read_csv(MEMBERSHIP).astype("int8"); kgroups = profile_groups(X)
    rr = pd.read_excel(RIPAS); Xr = ripas_transport(rr); yr = rr.admission.to_numpy(int); rgroups = profile_groups(Xr)
    adult = Xr.age.ge(18).to_numpy(); models = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
    outer_records = []; selection_records = []; internal_predictions = []; internal_slice_records = []
    external_records = []; subgroup_records = []; boot_records = []
    internal_pred_path = OUT / "internal_oof_predictions.csv"; external_pred_path = OUT / "ripas_frozen_predictions.csv"
    for path in [internal_pred_path, external_pred_path]:
        if path.exists(): path.unlink()
    for mi, name in enumerate(models):
        print("START", name, flush=True); pooled = {"y": [], "ERM": [], "BASiC_ASAP": [], "threshold": []}
        model_internal_frames = []; model_external_frames = []
        outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
        for fold, (tr, te) in enumerate(outer.split(X, y, kgroups), 1):
            lam, topk, selection_threshold, stab = select_repair(name, X.iloc[tr].reset_index(drop=True), y[tr], kgroups[tr], sm.iloc[tr].reset_index(drop=True), SEED + mi*100 + fold)
            stab.assign(model=name, outer_fold=fold).to_csv(OUT / f"selection_grid__{name.replace(' ', '_')}__fold{fold}.csv", index=False)
            base_oof = grouped_oof(name, X.iloc[tr].reset_index(drop=True), y[tr], kgroups[tr], RANKING_FOLDS, SEED + fold)
            ranking = rank_slices(y[tr], base_oof, sm.iloc[tr].reset_index(drop=True)); threshold = choose_threshold(y[tr], base_oof)
            w, selected = positive_target_weights(y[tr], sm.iloc[tr].reset_index(drop=True), ranking, lam, topk)
            selection_records.append({"model": name, "outer_fold": fold, "lambda": lam, "top_k": topk, "selected_slices": "|".join(selected), "threshold": threshold})
            fold_preds = {}
            for arm, ww in [("ERM", None), ("BASiC_ASAP", w if lam > 0 else None)]:
                m = fit_model(name, X.iloc[tr], y[tr], SEED + fold, ww); p = m.predict_proba(X.iloc[te])[:, 1]
                fold_preds[arm] = p
                row = {"model": name, "outer_fold": fold, "arm": arm, "n": len(te), "events": int(y[te].sum()), "lambda": lam, "top_k": topk, "threshold": threshold}
                row.update(metric_row(y[te], p, threshold)); row.update(calibration_stats(y[te], p)); outer_records.append(row)
                pooled[arm].append(p)
                if arm == "ERM": pooled["y"].append(y[te]); pooled["threshold"].append(np.repeat(threshold, len(te)))
                model_internal_frames.append(pd.DataFrame({"model": name, "outer_fold": fold, "arm": arm,
                    "source_row": te.astype(int), "profile_group": kgroups[te].astype("uint64"), "y": y[te].astype(int),
                    "probability": p.astype(float), "threshold": threshold}))
                del m
                gc.collect()
            for slice_name in selected:
                mask = sm.iloc[te][slice_name].to_numpy(bool)
                if mask.sum() < MIN_SLICE_N or y[te][mask].sum() < MIN_SLICE_EVENTS or len(np.unique(y[te][mask])) < 2:
                    continue
                for arm, p in fold_preds.items():
                    sr = {"model": name, "outer_fold": fold, "arm": arm, "slice_name": slice_name,
                          "n": int(mask.sum()), "events": int(y[te][mask].sum()), "threshold": threshold}
                    sr.update(metric_row(y[te][mask], p[mask], threshold)); internal_slice_records.append(sr)
            pd.concat(model_internal_frames, ignore_index=True).to_csv(internal_pred_path, mode="a", header=not internal_pred_path.exists(), index=False)
            model_internal_frames = []
            print(name, "fold", fold, "lambda", lam, "topk", topk, flush=True)
            gc.collect()
        yy = np.concatenate(pooled["y"]); pp0 = np.concatenate(pooled["ERM"]); pp1 = np.concatenate(pooled["BASiC_ASAP"]); tt = np.concatenate(pooled["threshold"])
        for arm, p in [("ERM", pp0), ("BASiC_ASAP", pp1)]:
            row = {"model": name, "arm": arm, "n": len(yy), "events": int(yy.sum()), "prevalence": float(yy.mean())}
            row.update(metric_row(yy, p, float(np.median(tt)))); row.update(calibration_stats(yy, p)); internal_predictions.append(row)

        full_oof = grouped_oof(name, X, y, kgroups, 5, SEED + mi); ranking = rank_slices(y, full_oof, sm)
        chosen = pd.DataFrame(selection_records); chosen = chosen[chosen.model == name]
        lam = float(chosen["lambda"].median()); topk = int(round(chosen.top_k.median())); threshold = choose_threshold(y, full_oof)
        w, selected = positive_target_weights(y, sm, ranking, lam, topk)
        preds = {}
        for arm, ww in [("ERM", None), ("BASiC_ASAP", w if lam > 0 else None)]:
            m = fit_model(name, X, y, SEED, ww); p = m.predict_proba(Xr)[:, 1]; preds[arm] = p
            model_external_frames.append(pd.DataFrame({"model": name, "arm": arm, "source_row": np.arange(len(yr), dtype=int),
                "profile_group": rgroups.astype("uint64"), "y": yr.astype(int), "adult": adult.astype(bool),
                "probability": p.astype(float), "threshold": threshold}))
            pi, pis = crossfit_recalibration(yr[adult], p[adult], rgroups[adult], SEED + mi)
            for update, pu in [("transported", p[adult]), ("intercept_only_crossfit", pi), ("intercept_slope_crossfit", pis)]:
                row = {"model": name, "arm": arm, "update": update, "cohort": "RIPAS_adult", "n": int(adult.sum()), "events": int(yr[adult].sum()),
                       "prevalence": float(yr[adult].mean()), "kaggle_prevalence": float(y.mean()), "prevalence_ratio": float(yr[adult].mean()/y.mean()),
                       "threshold": threshold, "mean_predicted_risk": float(np.mean(pu)),
                       "AUPRC_standardized_to_kaggle_prevalence": prevalence_standardized_auprc(yr[adult], pu, y.mean())}
                row.update(metric_row(yr[adult], pu, threshold)); row.update(calibration_stats(yr[adult], pu)); external_records.append(row)
            subgroup_records.extend(subgroup_rows(name, arm, yr[adult], p[adult], threshold, rr.loc[adult].reset_index(drop=True), Xr.loc[adult].reset_index(drop=True)))
            del m
            gc.collect()
        for row in cluster_bootstrap(yr[adult], preds["ERM"][adult], preds["BASiC_ASAP"][adult], rgroups[adult], SEED + mi, y.mean(), threshold):
            row.update({"model": name, "cohort": "RIPAS_adult"}); boot_records.append(row)
        ranking.assign(model=name).to_csv(OUT / f"full_training_ranking__{name.replace(' ', '_')}.csv", index=False)
        pd.concat(model_external_frames, ignore_index=True).to_csv(external_pred_path, mode="a", header=not external_pred_path.exists(), index=False)
        del model_internal_frames, model_external_frames, preds, full_oof, ranking
        gc.collect()
        print("EXTERNAL", name, flush=True)

    outer_df = pd.DataFrame(outer_records); internal_df = pd.DataFrame(internal_predictions); external_df = pd.DataFrame(external_records); boot_df = pd.DataFrame(boot_records)
    outer_df.to_csv(OUT / "internal_outer_fold_results.csv", index=False); internal_df.to_csv(OUT / "internal_pooled_oof_results.csv", index=False)
    pd.DataFrame(internal_slice_records).to_csv(OUT / "internal_target_slice_results.csv", index=False)
    pd.DataFrame(selection_records).to_csv(OUT / "governed_selection_log.csv", index=False); external_df.to_csv(OUT / "ripas_transport_and_recalibration.csv", index=False)
    pd.DataFrame(subgroup_records).to_csv(OUT / "ripas_demographic_and_clinical_subgroups.csv", index=False); boot_df.to_csv(OUT / "ripas_cluster_bootstrap_deltas.csv", index=False)
    manifest = {
        "analysis_id": hashlib.sha256(str(SEED).encode()).hexdigest()[:12], "seed": SEED, "outer_folds": OUTER_FOLDS,
        "ranking_folds": RANKING_FOLDS, "bootstrap_replicates": N_BOOT, "models": models,
        "kaggle_rows": len(raw), "kaggle_events": int(y.sum()), "kaggle_exact_duplicate_rows": int(raw.duplicated().sum()),
        "kaggle_unique_predictor_profiles": int(len(np.unique(kgroups))), "ripas_rows": len(rr), "ripas_events": int(yr.sum()),
        "ripas_exact_duplicate_rows": int(rr.duplicated().sum()), "ripas_unique_predictor_profiles": int(len(np.unique(rgroups))),
        "identifier_limitation": "Neither supplied dataset contained a patient identifier; exact-predictor-profile grouping prevents identical profiles crossing internal folds and supplies conservative external bootstrap clusters but cannot guarantee patient-level independence.",
        "external_use": "RIPAS outcomes were used only after training, repair-selection, and threshold locking on Kaggle; recalibration is separately labelled cross-fitted local updating.",
        "global_AUPRC_noninferiority_margin": GLOBAL_AUPRC_MARGIN, "global_Brier_deterioration_margin": GLOBAL_BRIER_MARGIN,
        "elapsed_seconds": time.time() - start,
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(internal_df.to_string(index=False)); print(boot_df.to_string(index=False)); print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
