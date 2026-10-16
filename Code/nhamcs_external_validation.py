from __future__ import annotations

import importlib.util
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "Code" / "multimodel_external_validation.py"
spec = importlib.util.spec_from_file_location("base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

NHAMCS = Path(os.environ.get("NHAMCS_DATA_PATH", ROOT / "Data" / "ed2022-stata.dta"))
OUT = ROOT / "basic_slice_prem" / "NHAMCS_External_Results"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
RFV_COLS = ["RFV1", "RFV2", "RFV3", "RFV4", "RFV5"]


def valid_numeric(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.mask(x < 0)


def any_rfv(df, codes):
    vals = df[RFV_COLS].apply(pd.to_numeric, errors="coerce")
    return vals.isin(codes).any(axis=1).astype("int8")


def nhamcs_transport(df):
    out = pd.DataFrame(index=df.index)
    for target, source in {
        "age": "AGE", "bp_diastole": "BPDIAS", "bp_systolic": "BPSYS",
        "heart_rate": "PULSE", "respiratory_rate": "RESPR", "spo2": "POPCT",
    }.items():
        out[target] = valid_numeric(df[source])
    out["temperature_c"] = (valid_numeric(df["TEMPF"]) / 10.0 - 32.0) * 5.0 / 9.0
    # NHAMCS Reason-for-Visit codes from the public-use coding system.
    out["abdominal_pain"] = any_rfv(df, {15450, 15451, 15452, 15453})
    out["chest_pain"] = any_rfv(df, {10500, 10501, 10502, 10503})
    out["back_pain"] = any_rfv(df, {19050, 19051, 19052, 19053})
    out["headache"] = any_rfv(df, {12100, 12101, 12102, 12103})
    out["other_pain"] = ((valid_numeric(df["PAINSCALE"]).notna()) &
                         ~out[["abdominal_pain", "chest_pain", "back_pain", "headache"]].any(axis=1)).astype("int8")
    out["cardiovascular_history"] = ((df["HTN"] == 1) | (df["CAD"] == 1) | (df["CHF"] == 1)).astype("int8")
    out["dm"] = ((df["DIABTYP1"] == 1) | (df["DIABTYP2"] == 1) | (df["DIABTYP0"] == 1)).astype("int8")
    out["ckd"] = ((df["CKD"] == 1) | (df["ESRD"] == 1)).astype("int8")
    return out[base.FEATURES]


def weights_from_saved_ranking(sm, ranking):
    w = np.ones(len(sm), dtype=float)
    chosen = ranking[(ranking["composite_harm"] > 0) &
                     ~ranking["slice_name"].isin(base.EXCLUDED_ARTIFACT_SLICES)].head(base.TOP_K)
    for row in chosen.itertuples(index=False):
        w += base.LAMBDA_REPAIR * float(row.harm_norm) * sm[row.slice_name].to_numpy(float)
    return np.clip(w, 1, base.WEIGHT_CAP)


def bootstrap(y, p0, p1, seed, n=1000):
    rng = np.random.default_rng(seed)
    rows = []
    for metric in ("AUPRC", "AUROC", "Brier"):
        vals = []
        for _ in range(n):
            ix = rng.integers(0, len(y), len(y)); yy = y[ix]
            if np.unique(yy).size < 2: continue
            if metric == "AUPRC": val = base.average_precision_score(yy, p1[ix]) - base.average_precision_score(yy, p0[ix])
            elif metric == "AUROC": val = base.roc_auc_score(yy, p1[ix]) - base.roc_auc_score(yy, p0[ix])
            else: val = base.brier_score_loss(yy, p1[ix]) - base.brier_score_loss(yy, p0[ix])
            vals.append(val)
        a = np.asarray(vals)
        rows.append({"metric": metric, "mean_delta": a.mean(), "ci_low": np.quantile(a, .025),
                     "ci_high": np.quantile(a, .975), "n_success": len(a)})
    return rows


def main():
    raw = pd.read_csv(base.KAGGLE, low_memory=False)
    X = base.kaggle_transport(raw); y = raw.y.to_numpy(int)
    sm = pd.read_csv(base.MEMBERSHIP).astype("int8")
    nh = pd.read_stata(NHAMCS, convert_categoricals=False)
    Xn = nhamcs_transport(nh)
    yn = nh["ADMITHOS"].eq(1).to_numpy(int)
    adult = valid_numeric(nh["AGE"]).ge(18).to_numpy()
    analyzable = adult & np.isfinite(valid_numeric(nh["AGE"]).to_numpy())
    rows, boots = [], []
    saved_thresholds = (pd.read_csv(base.OUT / "ripas_external_results.csv")
                        .query("arm == 'ERM' and cohort == 'RIPAS_adult'")
                        .set_index("model")["kaggle_threshold"])
    for i, name in enumerate(MODELS):
        ranking = pd.read_csv(base.OUT / f"full_kaggle_ranking__{name.replace(' ', '_')}.csv")
        weights = weights_from_saved_ranking(sm, ranking)
        threshold = float(saved_thresholds.loc[name])
        preds = {}
        for arm, w in (("ERM", None), ("BASiC_ASAP", weights)):
            model = base.model_factory(name, base.SEED)
            base.fit(model, X, y, w)
            p = model.predict_proba(Xn)[:, 1]; preds[arm] = p
            row = {"model": name, "arm": arm, "cohort": "NHAMCS_2022_adult", "n": int(analyzable.sum()),
                   "events": int(yn[analyzable].sum()), "prevalence": float(yn[analyzable].mean()),
                   "kaggle_threshold": threshold}
            row.update(base.metrics(yn[analyzable], p[analyzable], threshold))
            row.update(base.calibration(yn[analyzable], p[analyzable])); rows.append(row)
        for rec in bootstrap(yn[analyzable], preds["ERM"][analyzable], preds["BASiC_ASAP"][analyzable], base.SEED+i):
            rec.update({"model": name, "cohort": "NHAMCS_2022_adult"}); boots.append(rec)
        print("done", name, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "nhamcs_external_results.csv", index=False)
    pd.DataFrame(boots).to_csv(OUT / "nhamcs_paired_bootstrap_deltas.csv", index=False)
    manifest = {"source": "CDC 2022 NHAMCS ED public-use file", "rows": len(nh),
                "adult_rows": int(analyzable.sum()), "adult_admissions": int(yn[analyzable].sum()),
                "outcome": "ADMITHOS == 1", "features": base.FEATURES,
                "notes": ["Negative NHAMCS sentinel codes treated as missing.",
                          "TEMPF stored in tenths of degrees Fahrenheit and converted to Celsius.",
                          "Models, Kaggle-derived repair weights, and thresholds locked before NHAMCS evaluation."]}
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(boots).to_string(index=False))


if __name__ == "__main__":
    main()
