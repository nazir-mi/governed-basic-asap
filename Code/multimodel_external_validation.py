from __future__ import annotations

import json
import time
from pathlib import Path
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1] / "basic_slice_prem"
KAGGLE = ROOT / "Data" / "Raw" / "triage_cleaned_working.csv"
MEMBERSHIP = ROOT / "Notebooks" / "slice_membership.csv"
RIPAS = Path(os.environ["RIPAS_DATA_PATH"])
OUT = ROOT / "MultiModel_External_Results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026
OUTER_FOLDS = 5
INNER_FOLDS = 3
TOP_K = 10
LAMBDA_REPAIR = 1.5
WEIGHT_CAP = 5.0
N_BOOT = 1000
EXCLUDED_ARTIFACT_SLICES = {"fever_temp"}
DUAL_SUPPORTED = {
    "abdominal_pain_and_hr_missing", "ckd_and_spo2_missing",
    "unclassified_and_vitals_missing_ge_3",
}

FEATURES = [
    "age", "bp_diastole", "bp_systolic", "heart_rate", "respiratory_rate",
    "spo2", "temperature_c", "abdominal_pain", "chest_pain", "back_pain",
    "headache", "other_pain", "cardiovascular_history", "dm", "ckd",
]


def binary(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).gt(0).astype("int8")


def kaggle_transport(df):
    out = pd.DataFrame(index=df.index)
    out["age"] = pd.to_numeric(df["age"], errors="coerce")
    out["bp_diastole"] = pd.to_numeric(df["bp_diastole"], errors="coerce")
    out["bp_systolic"] = pd.to_numeric(df["bp_systolic"], errors="coerce")
    out["heart_rate"] = pd.to_numeric(df["heart_rate_final"], errors="coerce")
    out["respiratory_rate"] = pd.to_numeric(df["respiratory"], errors="coerce")
    out["spo2"] = pd.to_numeric(df["spo2"], errors="coerce")
    temp = pd.to_numeric(df["temperature"], errors="coerce")
    out["temperature_c"] = np.where(temp.gt(60), (temp - 32) * 5 / 9, temp)
    for c in ["abdominal_pain", "chest_pain", "back_pain", "headache", "dm", "ckd"]:
        out[c] = binary(df[c])
    other_cols = ["arm_pain", "breast_pain", "dental_pain", "ear_pain", "eye_pain", "pelvic_pain", "unclassified"]
    out["other_pain"] = pd.concat([binary(df[c]) for c in other_cols], axis=1).max(axis=1).astype("int8")
    out["cardiovascular_history"] = np.maximum(binary(df["htn"]), binary(df["chd"]))
    return out[FEATURES]


def ripas_transport(df):
    out = pd.DataFrame(index=df.index)
    for target, source in {
        "age":"age", "bp_diastole":"bp_diastole", "bp_systolic":"bp_systolic",
        "heart_rate":"pulse_rate", "respiratory_rate":"respiratory_rate",
        "spo2":"spo2", "temperature_c":"temperature",
    }.items(): out[target] = pd.to_numeric(df[source], errors="coerce")
    pain = df["site_of_pain"].fillna("").astype(str).str.strip().str.upper()
    out["abdominal_pain"] = pain.str.contains("ABDOMINAL|PELVIC", regex=True).astype("int8")
    out["chest_pain"] = pain.str.contains("CHEST", regex=True).astype("int8")
    out["back_pain"] = pain.str.contains("BACK", regex=True).astype("int8")
    out["headache"] = pain.str.contains("HEADACHE", regex=True).astype("int8")
    out["other_pain"] = (~(out[["abdominal_pain","chest_pain","back_pain","headache"]].max(axis=1).astype(bool))).astype("int8")
    hx = df["htn_chd"].fillna("NO").astype(str).str.upper().str.replace(" ", "", regex=False)
    out["cardiovascular_history"] = hx.ne("NO").astype("int8")
    out["dm"] = df["dm"].fillna("NO").astype(str).str.upper().ne("NO").astype("int8")
    out["ckd"] = binary(df["ckd"])
    return out[FEATURES]


def model_factory(name, seed):
    if name == "Logistic Regression":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("model", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed))])
    if name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=20,
                                       max_features="sqrt", n_jobs=-1, random_state=seed)
    elif name == "XGBoost":
        model = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=.05, subsample=.8,
                              colsample_bytree=.8, min_child_weight=10, reg_lambda=1.0,
                              objective="binary:logistic", eval_metric="logloss", n_jobs=-1,
                              random_state=seed)
    elif name == "LightGBM":
        model = LGBMClassifier(n_estimators=250, learning_rate=.05, num_leaves=31,
                               max_depth=-1, min_child_samples=40, subsample=.8,
                               colsample_bytree=.8, reg_lambda=1.0, n_jobs=-1,
                               random_state=seed, verbosity=-1)
    else: raise KeyError(name)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def fit(model, X, y, weights=None):
    kwargs = {} if weights is None else {"model__sample_weight": weights}
    return model.fit(X, y, **kwargs)


def metrics(y, p, threshold=.5):
    p = np.clip(np.asarray(p), 1e-6, 1-1e-6); y=np.asarray(y); pred=p>=threshold
    return {"AUROC":roc_auc_score(y,p), "AUPRC":average_precision_score(y,p),
            "Brier":brier_score_loss(y,p), "LogLoss":log_loss(y,p),
            "Recall":recall_score(y,pred,zero_division=0),
            "Precision":precision_score(y,pred,zero_division=0)}


def calibration(y, p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6); y=np.asarray(y)
    logits=np.log(p/(1-p)).reshape(-1,1)
    cal=LogisticRegression(C=1e6,solver="lbfgs").fit(logits,y)
    frac,mean=calibration_curve(y,p,n_bins=min(10,max(3,int(len(y)**.5/4))),strategy="quantile")
    return {"calibration_intercept":float(cal.intercept_[0]),
            "calibration_slope":float(cal.coef_[0,0]), "ECE":float(np.mean(np.abs(frac-mean)))}


def choose_threshold(y,p):
    best=(-np.inf,.5)
    for t in np.linspace(.05,.95,181):
        pred=p>=t; sens=recall_score(y,pred,zero_division=0)
        spec=((~pred)&(y==0)).sum()/max((y==0).sum(),1)
        if sens+spec-1>best[0]: best=(sens+spec-1,float(t))
    return best[1]


def rank_slices(y,p,sm,indices):
    p=np.clip(p,1e-6,1-1e-6); loss=-(y*np.log(p)+(1-y)*np.log(1-p)); pred=p>=.5
    fn=((y==1)&(~pred)).astype(float); hcc=(((y==1)&(p<.2))|((y==0)&(p>.8))).astype(float)
    rows=[]
    for s in sm.columns:
        m=sm.loc[indices,s].to_numpy(dtype=bool)
        if m.sum()<50 or np.unique(y[m]).size<2: continue
        rows.append({"slice_name":s,"support":int(m.sum()),"loss_lift":loss[m].mean()-loss.mean(),
                     "fn_lift":fn[m].mean()-fn.mean(),"hcc_lift":hcc[m].mean()-hcc.mean()})
    r=pd.DataFrame(rows); r["composite_harm"]=r[["loss_lift","fn_lift","hcc_lift"]].sum(axis=1)
    lo,hi=r.composite_harm.min(),r.composite_harm.max(); r["harm_norm"]=(r.composite_harm-lo)/(hi-lo) if hi>lo else 0
    r["dual_source_evidence"]=r.slice_name.isin(DUAL_SUPPORTED).astype(int); r["support_norm"]=r.support/r.support.max()
    r["rank_score"]=.6*r.harm_norm+.3*r.dual_source_evidence+.1*r.support_norm
    return r.sort_values(["rank_score","composite_harm"],ascending=False).reset_index(drop=True)


def make_weights(sm,indices,ranking):
    w=np.ones(len(indices)); selected=ranking[(ranking.composite_harm>0)&~ranking.slice_name.isin(EXCLUDED_ARTIFACT_SLICES)].head(TOP_K)
    for row in selected.itertuples(index=False):
        w += LAMBDA_REPAIR*float(row.harm_norm)*sm.loc[indices,row.slice_name].to_numpy(float)
    return np.clip(w,1,WEIGHT_CAP), selected.slice_name.tolist()


def oof_predictions(model_name,X,y,indices,n_splits,seed,weights=None):
    out=np.zeros(len(indices)); cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    for fold,(tr,ho) in enumerate(cv.split(indices,y),1):
        m=model_factory(model_name,seed+fold); ww=None if weights is None else weights[tr]
        fit(m,X.iloc[indices[tr]],y[tr],ww); out[ho]=m.predict_proba(X.iloc[indices[ho]])[:,1]
    return out


def external_slices(r):
    return {
        "abdominal_pain":r.abdominal_pain.eq(1).to_numpy(), "chest_pain":r.chest_pain.eq(1).to_numpy(),
        "back_pain":r.back_pain.eq(1).to_numpy(), "cardiovascular_history":r.cardiovascular_history.eq(1).to_numpy(),
        "dm":r.dm.eq(1).to_numpy(), "ckd":r.ckd.eq(1).to_numpy(),
        "tachycardia":r.heart_rate.gt(100).to_numpy(), "tachypnea":r.respiratory_rate.gt(20).to_numpy(),
        "hypoxia":r.spo2.lt(95).to_numpy(),
    }


def bootstrap_external(y,p0,p1,seed):
    rng=np.random.default_rng(seed); rows=[]
    for metric in ["AUPRC","Brier"]:
        vals=[]
        for _ in range(N_BOOT):
            idx=rng.integers(0,len(y),len(y)); yy=y[idx]
            if np.unique(yy).size<2: continue
            if metric=="AUPRC": v=average_precision_score(yy,p1[idx])-average_precision_score(yy,p0[idx])
            else: v=brier_score_loss(yy,p1[idx])-brier_score_loss(yy,p0[idx])
            vals.append(v)
        a=np.asarray(vals); rows.append({"metric":metric,"n_success":len(a),"mean_delta":a.mean(),
                                        "ci_low":np.quantile(a,.025),"ci_high":np.quantile(a,.975)})
    return rows


def main():
    start=time.time(); raw=pd.read_csv(KAGGLE,low_memory=False); X=kaggle_transport(raw); y=raw.y.to_numpy(int)
    sm=pd.read_csv(MEMBERSHIP).astype("int8"); ripas_raw=pd.read_excel(RIPAS); Xr=ripas_transport(ripas_raw); yr=ripas_raw.admission.to_numpy(int)
    adult_mask=Xr.age.ge(18).to_numpy(); models=["Logistic Regression","Random Forest","XGBoost","LightGBM"]
    outer_rows=[]; selection_rows=[]; external_rows=[]; external_slice_rows=[]; boot_rows=[]
    outer=StratifiedKFold(n_splits=OUTER_FOLDS,shuffle=True,random_state=SEED)
    all_idx=np.arange(len(y))
    for model_name in models:
        print("START",model_name,flush=True)
        for fold,(tr,te) in enumerate(outer.split(all_idx,y),1):
            tr_idx,te_idx=all_idx[tr],all_idx[te]; ytr=y[tr_idx]
            inner_oof=oof_predictions(model_name,X,ytr,tr_idx,INNER_FOLDS,SEED+fold)
            ranking=rank_slices(ytr,inner_oof,sm,tr_idx); weights,selected=make_weights(sm,tr_idx,ranking)
            threshold=choose_threshold(ytr,inner_oof)
            for rank,s in enumerate(selected,1): selection_rows.append({"model":model_name,"outer_fold":fold,"rank":rank,"slice_name":s})
            for arm,w in [("ERM",None),("BASiC_ASAP",weights)]:
                m=model_factory(model_name,SEED+fold); fit(m,X.iloc[tr_idx],ytr,w); p=m.predict_proba(X.iloc[te_idx])[:,1]
                row={"model":model_name,"outer_fold":fold,"arm":arm,"n_test":len(te_idx),"threshold":threshold}; row.update(metrics(y[te_idx],p,threshold)); row.update(calibration(y[te_idx],p)); outer_rows.append(row)
            print(model_name,"outer",fold,"done",flush=True)

        # Full-development OOF ranking and final transport model.
        full_oof=oof_predictions(model_name,X,y,all_idx,5,SEED); ranking=rank_slices(y,full_oof,sm,all_idx)
        weights,selected=make_weights(sm,all_idx,ranking); threshold=choose_threshold(y,full_oof)
        external_preds={}
        for arm,w in [("ERM",None),("BASiC_ASAP",weights)]:
            m=model_factory(model_name,SEED); fit(m,X,y,w); p=m.predict_proba(Xr)[:,1]; external_preds[arm]=p
            for cohort,mask in [("RIPAS_adult",adult_mask),("RIPAS_all",np.ones(len(yr),bool))]:
                row={"model":model_name,"arm":arm,"cohort":cohort,"n":int(mask.sum()),"events":int(yr[mask].sum()),"prevalence":float(yr[mask].mean()),"kaggle_threshold":threshold}
                row.update(metrics(yr[mask],p[mask],threshold)); row.update(calibration(yr[mask],p[mask])); external_rows.append(row)
                for s,smask in external_slices(Xr).items():
                    mm=mask&smask
                    if mm.sum()<30 or np.unique(yr[mm]).size<2: continue
                    sr={"model":model_name,"arm":arm,"cohort":cohort,"slice_name":s,"n":int(mm.sum()),"events":int(yr[mm].sum()),"threshold":threshold}; sr.update(metrics(yr[mm],p[mm],threshold)); external_slice_rows.append(sr)
        for rec in bootstrap_external(yr[adult_mask],external_preds["ERM"][adult_mask],external_preds["BASiC_ASAP"][adult_mask],SEED):
            rec.update({"model":model_name,"cohort":"RIPAS_adult"}); boot_rows.append(rec)
        ranking.assign(model=model_name).to_csv(OUT/f"full_kaggle_ranking__{model_name.replace(' ','_')}.csv",index=False)
        print("EXTERNAL",model_name,"done",flush=True)

    pd.DataFrame(outer_rows).to_csv(OUT/"nested_kaggle_results.csv",index=False)
    pd.DataFrame(selection_rows).to_csv(OUT/"nested_selection_stability.csv",index=False)
    pd.DataFrame(external_rows).to_csv(OUT/"ripas_external_results.csv",index=False)
    pd.DataFrame(external_slice_rows).to_csv(OUT/"ripas_external_slice_results.csv",index=False)
    pd.DataFrame(boot_rows).to_csv(OUT/"ripas_paired_bootstrap_deltas.csv",index=False)
    outer_df=pd.DataFrame(outer_rows)
    summary=outer_df.groupby(["model","arm"])[["AUROC","AUPRC","Brier","Recall","Precision","calibration_slope","ECE"]].agg(["mean","std"]).reset_index()
    summary.columns=["_".join(c).strip("_") for c in summary.columns]; summary.to_csv(OUT/"nested_kaggle_summary.csv",index=False)
    audit={"kaggle_rows":len(raw),"kaggle_events":int(y.sum()),"ripas_rows_observed":len(ripas_raw),"ripas_rows_stated_in_key":1637,
           "ripas_adult_rows":int(adult_mask.sum()),"ripas_all_events":int(yr.sum()),"ripas_adult_events":int(yr[adult_mask].sum()),
           "models":models,"features":FEATURES,"outer_folds":OUTER_FOLDS,"inner_folds":INNER_FOLDS,"seed":SEED,
           "mhrec_committee":"Brunei MHREC COMM","mhrec_reference":"RC 2025-013","mhrec_year":2025,
           "elapsed_seconds":time.time()-start}
    (OUT/"analysis_manifest.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(summary.to_string(index=False)); print(json.dumps(audit,indent=2),flush=True)


if __name__=="__main__": main()
