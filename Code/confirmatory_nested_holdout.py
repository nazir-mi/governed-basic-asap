from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1] / "basic_slice_prem"
DATA = ROOT / "Data" / "Raw" / "triage_cleaned_working.csv"
MEMBERSHIP = ROOT / "Notebooks" / "slice_membership.csv"
REGISTRY = ROOT / "Data" / "Raw" / "approved_slice_registry_FINAL.csv"
OUT = ROOT / "Confirmatory_Results"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [2026, 42, 314, 2718, 8675309]
PRIMARY_SEED = 2026
N_BOOT = 1000
TOP_K = 10
LAMBDA_REPAIR = 1.5
WEIGHT_CAP = 5.0
EXCLUDED_ARTIFACT_SLICES = {"fever_temp"}
DUAL_SUPPORTED = {
    "abdominal_pain_and_hr_missing",
    "ckd_and_spo2_missing",
    "unclassified_and_vitals_missing_ge_3",
}

NUMERIC = [
    "age", "n_admissions", "heart_rate_final", "bp_systolic", "bp_diastole",
    "respiratory", "temperature", "spo2", "vitals_missing_count",
    "abdominal_pain", "chest_pain", "back_pain", "unclassified", "ckd", "dm",
    "headache", "htn", "cough", "chd", "asthma", "bronchitis", "dizziness",
    "glaucoma", "hepatitis", "paralysis", "pneumonia", "sicklecell", "arm_pain",
    "breast_pain", "dental_pain", "diarrhea", "ear_pain", "eye_pain", "fever",
    "pelvic_pain", "arrival_mode_missing", "hr_missing", "pulse_rate_missing",
    "bp_systolic_missing", "bp_diastole_missing", "respiratory_missing",
    "temperature_missing", "spo2_missing",
]
CATEGORICAL = ["arrival_mode", "arrival_month", "arrival_day", "arrival_hour_bin"]
FEATURES = NUMERIC + CATEGORICAL


def make_model() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
        ]
    )
    return Pipeline([("preprocess", pre),
                     ("model", LogisticRegression(max_iter=3000, solver="lbfgs"))])


def reconstruct_membership(df: pd.DataFrame) -> pd.DataFrame:
    positive = lambda s: pd.to_numeric(s, errors="coerce").fillna(0).eq(1).astype(int)
    out = pd.DataFrame(index=df.index)
    out["arrival_mode_unknown"] = df["arrival_mode"].fillna("").astype(str).str.strip().str.lower().eq("unknown").astype(int)
    out["back_pain"] = positive(df["back_pain"])
    out["abdominal_pain_and_hr_missing"] = (positive(df["abdominal_pain"]).eq(1) & positive(df["hr_missing"]).eq(1)).astype(int)
    out["tachypnea"] = (df["respiratory"].notna() & df["respiratory"].gt(20)).astype(int)
    out["fever_temp"] = (df["temperature"].notna() & df["temperature"].ge(38.0)).astype(int)
    out["chest_pain_and_ambulance"] = (positive(df["chest_pain"]).eq(1) & df["arrival_mode"].fillna("").astype(str).str.lower().eq("ambulance")).astype(int)
    out["headache"] = positive(df["headache"])
    out["ckd_and_spo2_missing"] = (positive(df["ckd"]).eq(1) & positive(df["spo2_missing"]).eq(1)).astype(int)
    out["cough"] = positive(df["cough"])
    out["chd"] = positive(df["chd"])
    out["tachycardia"] = (df["heart_rate_final"].notna() & df["heart_rate_final"].gt(100)).astype(int)
    out["htn"] = positive(df["htn"])
    out["unclassified_and_vitals_missing_ge_3"] = (positive(df["unclassified"]).eq(1) & pd.to_numeric(df["vitals_missing_count"], errors="coerce").fillna(0).ge(3)).astype(int)
    out["hypoxia"] = (df["spo2"].notna() & df["spo2"].lt(95)).astype(int)
    out["dm"] = positive(df["dm"])
    out["pneumonia"] = positive(df["pneumonia"])
    return out


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, p),
        "Recall": recall_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
    }


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    # Prespecified Youden-style balance on validation data; no test information.
    grid = np.linspace(0.05, 0.95, 181)
    prevalence = y.mean()
    best = None
    for t in grid:
        pred = p >= t
        sensitivity = recall_score(y, pred, zero_division=0)
        specificity = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
        score = sensitivity + specificity - 1
        tie = -abs(pred.mean() - prevalence)
        candidate = (score, tie, -t)
        if best is None or candidate > best[0]:
            best = (candidate, float(t))
    return best[1]


def development_harm(df: pd.DataFrame, sm: pd.DataFrame, indices: np.ndarray, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    row_loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    pred = p >= 0.5
    fn = ((y == 1) & (~pred)).astype(float)
    hcc = (((y == 1) & (p < 0.2)) | ((y == 0) & (p > 0.8))).astype(float)
    rows = []
    for name in sm.columns:
        mask = sm.loc[indices, name].to_numpy(dtype=bool)
        if mask.sum() < 50 or np.unique(y[mask]).size < 2:
            continue
        rows.append({
            "slice_name": name,
            "support": int(mask.sum()),
            "support_rate": float(mask.mean()),
            "loss_lift": float(row_loss[mask].mean() - row_loss.mean()),
            "fn_lift": float(fn[mask].mean() - fn.mean()),
            "hcc_lift": float(hcc[mask].mean() - hcc.mean()),
        })
    out = pd.DataFrame(rows)
    out["composite_harm"] = out[["loss_lift", "fn_lift", "hcc_lift"]].sum(axis=1)
    lo, hi = out["composite_harm"].min(), out["composite_harm"].max()
    out["harm_norm"] = (out["composite_harm"] - lo) / (hi - lo) if hi > lo else 0.0
    out["dual_source_evidence"] = out["slice_name"].isin(DUAL_SUPPORTED).astype(int)
    out["support_norm"] = out["support"] / max(out["support"].max(), 1)
    out["rank_score"] = 0.60 * out["harm_norm"] + 0.30 * out["dual_source_evidence"] + 0.10 * out["support_norm"]
    return out.sort_values(["rank_score", "composite_harm"], ascending=False).reset_index(drop=True)


def sample_weights(sm: pd.DataFrame, indices: np.ndarray, ranking: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(indices), dtype=float)
    selected = ranking.loc[
        ranking["composite_harm"].gt(0)
        & ~ranking["slice_name"].isin(EXCLUDED_ARTIFACT_SLICES)
    ].head(TOP_K)
    for row in selected.itertuples(index=False):
        w += LAMBDA_REPAIR * float(row.harm_norm) * sm.loc[indices, row.slice_name].to_numpy(dtype=float)
    return np.clip(w, 1.0, WEIGHT_CAP)


def calibration_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pp = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(pp / (1 - pp)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, y)
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ece = float(np.mean(np.abs(frac - mean)))
    return {"calibration_intercept": float(cal.intercept_[0]), "calibration_slope": float(cal.coef_[0, 0]), "ECE10": ece}


def bootstrap_deltas(y: np.ndarray, p0: np.ndarray, p1: np.ndarray, sm_test: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    targets = {"global": np.ones(len(y), dtype=bool)}
    for s in ["abdominal_pain_and_hr_missing", "ckd_and_spo2_missing", "unclassified_and_vitals_missing_ge_3"]:
        targets[s] = sm_test[s].to_numpy(dtype=bool)
    records = []
    for target, target_mask in targets.items():
        loc = np.flatnonzero(target_mask)
        vals = {"delta_AUPRC": [], "delta_Brier": []}
        for _ in range(N_BOOT):
            idx = rng.choice(loc, size=len(loc), replace=True)
            if np.unique(y[idx]).size < 2:
                continue
            vals["delta_AUPRC"].append(average_precision_score(y[idx], p1[idx]) - average_precision_score(y[idx], p0[idx]))
            vals["delta_Brier"].append(brier_score_loss(y[idx], p1[idx]) - brier_score_loss(y[idx], p0[idx]))
        for metric, samples in vals.items():
            arr = np.asarray(samples)
            records.append({"target": target, "metric": metric, "n_success": len(arr), "mean_delta": arr.mean(), "ci_low": np.quantile(arr, .025), "ci_high": np.quantile(arr, .975), "p_two_sided": min(1.0, 2 * min((arr <= 0).mean(), (arr >= 0).mean()))})
    return pd.DataFrame(records)


def main() -> None:
    df = pd.read_csv(DATA, low_memory=False)
    sm = pd.read_csv(MEMBERSHIP).astype("int8")
    registry = pd.read_csv(REGISTRY)
    if len(df) != len(sm):
        raise ValueError(f"Row mismatch: cohort={len(df)}, membership={len(sm)}")
    reconstructed = reconstruct_membership(df)
    mismatch = (reconstructed[sm.columns].to_numpy() != sm.to_numpy()).sum(axis=0)
    alignment = pd.DataFrame({"slice_name": sm.columns, "mismatched_rows": mismatch})
    alignment.to_csv(OUT / "membership_alignment_audit.csv", index=False)
    if mismatch.sum() != 0:
        raise ValueError(f"Membership reconstruction mismatch: {alignment[alignment.mismatched_rows.gt(0)].to_dict('records')}")
    accepted = set(registry.loc[registry["final_accept"].astype(str).str.lower().eq("yes"), "slice_name"])
    if accepted != set(sm.columns):
        raise ValueError(f"Registry/membership mismatch: accepted_only={accepted-set(sm.columns)}, membership_only={set(sm.columns)-accepted}")

    y_all = df["y"].to_numpy(dtype=int)
    all_idx = np.arange(len(df))
    global_rows, slice_rows, rank_frames = [], [], []
    primary = {}

    for seed in SEEDS:
        dev_idx, temp_idx = train_test_split(all_idx, test_size=0.30, stratify=y_all, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y_all[temp_idx], random_state=seed)
        y_dev = y_all[dev_idx]
        oof = np.zeros(len(dev_idx), dtype=float)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_local, ho_local in cv.split(dev_idx, y_dev):
            model = make_model()
            model.fit(df.loc[dev_idx[tr_local], FEATURES], y_dev[tr_local])
            oof[ho_local] = model.predict_proba(df.loc[dev_idx[ho_local], FEATURES])[:, 1]
        ranking = development_harm(df, sm, dev_idx, y_dev, oof)
        ranking.insert(0, "seed", seed)
        rank_frames.append(ranking)
        weights = sample_weights(sm, dev_idx, ranking)

        preds = {}
        thresholds = {}
        for arm, w in [("ERM", None), ("BASiC_ASAP", weights)]:
            model = make_model()
            fit_kwargs = {} if w is None else {"model__sample_weight": w}
            model.fit(df.loc[dev_idx, FEATURES], y_dev, **fit_kwargs)
            p_val = model.predict_proba(df.loc[val_idx, FEATURES])[:, 1]
            p_test = model.predict_proba(df.loc[test_idx, FEATURES])[:, 1]
            threshold = choose_threshold(y_all[val_idx], p_val)
            preds[arm], thresholds[arm] = p_test, threshold
            row = {"seed": seed, "arm": arm, "n_dev": len(dev_idx), "n_val": len(val_idx), "n_test": len(test_idx), "threshold": threshold}
            row.update(metrics(y_all[test_idx], p_test, threshold))
            row.update(calibration_summary(y_all[test_idx], p_test))
            global_rows.append(row)
            for s in sm.columns:
                mask = sm.loc[test_idx, s].to_numpy(dtype=bool)
                if mask.sum() < 30 or np.unique(y_all[test_idx][mask]).size < 2:
                    continue
                sr = {"seed": seed, "arm": arm, "slice_name": s, "n": int(mask.sum()), "threshold": threshold}
                sr.update(metrics(y_all[test_idx][mask], p_test[mask], threshold))
                slice_rows.append(sr)
        if seed == PRIMARY_SEED:
            primary = {"test_idx": test_idx, "y": y_all[test_idx], "preds": preds, "thresholds": thresholds}

    global_df = pd.DataFrame(global_rows)
    slice_df = pd.DataFrame(slice_rows)
    ranks = pd.concat(rank_frames, ignore_index=True)
    global_df.to_csv(OUT / "locked_holdout_global_results.csv", index=False)
    slice_df.to_csv(OUT / "locked_holdout_slice_results.csv", index=False)
    ranks.to_csv(OUT / "development_only_slice_rankings.csv", index=False)

    deltas = bootstrap_deltas(primary["y"], primary["preds"]["ERM"], primary["preds"]["BASiC_ASAP"], sm.loc[primary["test_idx"]].reset_index(drop=True), PRIMARY_SEED)
    deltas.to_csv(OUT / "paired_bootstrap_deltas.csv", index=False)

    seed_summary = global_df.groupby("arm").agg({"AUROC": ["mean", "std"], "AUPRC": ["mean", "std"], "Brier": ["mean", "std"], "Recall": ["mean", "std"], "ECE10": ["mean", "std"]})
    seed_summary.to_csv(OUT / "repeated_seed_summary.csv")
    manifest = {
        "analysis_design": "development-only five-fold OOF ranking; validation-only threshold selection; locked test evaluation",
        "historical_data_caveat": "The dataset and candidate registry had been examined previously; this is a leakage-reduced internal re-analysis, not an untouched external confirmation.",
        "rows": len(df), "outcome_positive": int(y_all.sum()), "outcome_prevalence": float(y_all.mean()),
        "seeds": SEEDS, "primary_seed": PRIMARY_SEED, "n_boot": N_BOOT,
        "top_k": TOP_K, "lambda_repair": LAMBDA_REPAIR, "weight_cap": WEIGHT_CAP,
        "excluded_artifact_slices": sorted(EXCLUDED_ARTIFACT_SLICES),
        "membership_mismatches": int(mismatch.sum()),
    }
    (OUT / "confirmatory_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(global_df.to_string(index=False))
    print("\nPaired bootstrap deltas\n", deltas.to_string(index=False))


if __name__ == "__main__":
    main()
