from __future__ import annotations

import json, math, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, log_loss, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sliceline import Slicefinder

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"basic_slice_prem"/"Data"/"Raw"/"triage_train_diagnostics.csv"
OUT=ROOT/"basic_slice_prem"/"Faithful_Comparator_Results"; OUT.mkdir(parents=True,exist_ok=True)
SEED=2026; MIN_SUP=.005; K=10

def qcut(s,bins,labels):
    x=pd.to_numeric(s,errors="coerce")
    return pd.cut(x,bins=bins,labels=labels,include_lowest=True).astype("object").fillna("missing")

def candidate_frame(d):
    x=pd.DataFrame(index=d.index)
    x["age_group"]=qcut(d.age,[-np.inf,39,64,79,np.inf],["18-39","40-64","65-79","80+"])
    x["sbp_group"]=qcut(d.bp_systolic,[-np.inf,89,119,139,179,np.inf],["<90","90-119","120-139","140-179","180+"])
    x["heart_rate_group"]=qcut(d.heart_rate_final,[-np.inf,59,100,130,np.inf],["<60","60-100","101-130","130+"])
    x["resp_group"]=qcut(d.respiratory,[-np.inf,11,20,29,np.inf],["<12","12-20","21-29","30+"])
    x["spo2_group"]=qcut(d.spo2,[-np.inf,89,94,100,np.inf],["<90","90-94","95-100",">100"])
    x["arrival_mode"]=d.arrival_mode.fillna("missing").astype(str)
    binary=["abdominal_pain","chest_pain","back_pain","unclassified","ckd","dm","headache","htn","cough","chd","asthma","bronchitis","dizziness","pneumonia","arm_pain","pelvic_pain","arrival_mode_missing","hr_missing","bp_systolic_missing","bp_diastole_missing","respiratory_missing","temperature_missing","spo2_missing","heart_rate_disagreement_flag"]
    for c in binary: x[c]=pd.to_numeric(d[c],errors="coerce").fillna(0).gt(0).map({False:"0",True:"1"})
    return x

def mask_from_spec(x,spec):
    m=np.ones(len(x),bool)
    for c,v in spec: m &= x[c].astype(str).eq(str(v)).to_numpy()
    return m

def eval_mask(name,method,m,y,p,loss):
    n=int(m.sum()); pred=p>=.5; fn=((y==1)&(~pred)).astype(float); hcc=(((y==1)&(p<.2))|((y==0)&(p>.8))).astype(float)
    if n<2 or np.unique(y[m]).size<2: auroc=np.nan; auprc=np.nan
    else: auroc=roc_auc_score(y[m],p[m]); auprc=average_precision_score(y[m],p[m])
    lifts=[loss[m].mean()-loss.mean(),fn[m].mean()-fn.mean(),hcc[m].mean()-hcc.mean()]
    return {"method":method,"slice_name":name,"support":n,"support_fraction":n/len(y),"event_rate":y[m].mean(),"AUROC":auroc,"AUPRC":auprc,"loss_lift":lifts[0],"fn_lift":lifts[1],"hcc_lift":lifts[2],"harm_score":sum(lifts)}

def slicefinder_effect(loss,m):
    a=loss[m]; b=loss[~m]
    if len(a)<2 or len(b)<2:return -np.inf
    denom=math.sqrt((a.std(ddof=0)+b.std(ddof=0))/2)
    return (a.mean()-b.mean())/denom if denom>0 else 0

def faithful_slicefinder(x,loss):
    base=[]
    for c in x.columns:
        for v in x[c].astype(str).unique():
            m=x[c].astype(str).eq(v).to_numpy(); base.append((((c,v),),m))
    candidates=list(base)
    for i in range(len(base)-1):
        (s1,m1)=base[i]
        for j in range(i+1,len(base)):
            s2,m2=base[j]
            if s1[0][0]==s2[0][0]:continue
            candidates.append((tuple(sorted((s1[0],s2[0]))),m1&m2))
    rows=[]
    for spec,m in candidates:
        if m.mean()<MIN_SUP:continue
        es=slicefinder_effect(loss,m)
        if es>=.2: rows.append({"spec":spec,"mask":m,"effect_size":es,"support":int(m.sum())})
    # Original repository sorts qualifying slices by support, not effect size.
    return sorted(rows,key=lambda r:r["support"],reverse=True)[:K]

def official_sliceline(x,loss):
    model=Slicefinder(alpha=.95,k=K,max_l=2,min_sup=MIN_SUP,verbose=False).fit(x,loss)
    membership=model.transform(x).astype(bool); out=[]
    for i,row in enumerate(model.top_slices_):
        spec=tuple((c,v) for c,v in zip(model.feature_names_in_,row) if v is not None)
        out.append({"spec":spec,"mask":membership[:,i],**model.top_slices_statistics_[i]})
    return out

def afisp_protocol(x,y,p,loss):
    enc=OneHotEncoder(handle_unknown="ignore",sparse_output=True)
    z=enc.fit_transform(x); names=enc.get_feature_names_out(x.columns)
    # AFISP stability stage: model conditional expected loss and identify the largest
    # prespecified fraction whose observed AUROC is below the reference threshold.
    reg=RandomForestRegressor(n_estimators=250,max_depth=8,min_samples_leaf=100,n_jobs=-1,random_state=SEED,max_features=.7)
    reg.fit(z,loss); conditional=reg.predict(z); global_auc=roc_auc_score(y,p); threshold=global_auc-.05
    selected=None; curve=[]
    for alpha in [.50,.40,.30,.20,.15,.10,.075,.05,.025,.01]:
        n=max(50,int(alpha*len(y))); ix=np.argpartition(conditional,-n)[-n:]; auc=roc_auc_score(y[ix],p[ix]) if np.unique(y[ix]).size>1 else np.nan
        curve.append({"alpha":alpha,"n":n,"AUROC":auc,"performance_threshold":threshold})
        if selected is None and auc<threshold:selected=(alpha,ix)
    if selected is None:selected=(.10,np.argpartition(conditional,-int(.10*len(y)))[-int(.10*len(y)):])
    alpha,ix=selected; target=np.zeros(len(y),int); target[ix]=1
    # SIRUS-compatible stable rule extraction: frequent depth<=3 paths from a
    # quantile-binned random forest, followed by prevalence/effect filtering.
    clf=RandomForestClassifier(n_estimators=1000,max_depth=3,min_samples_leaf=100,max_features="sqrt",bootstrap=True,max_samples=.7,n_jobs=-1,random_state=SEED,class_weight="balanced")
    clf.fit(z,target); counts=Counter()
    for est in clf.estimators_:
        tr=est.tree_
        def walk(node,path):
            if tr.feature[node]<0:return
            f=int(tr.feature[node]); thr=float(tr.threshold[node]); left=path+((f,"<=",thr),); right=path+((f,">",thr),)
            counts[left]+=1; counts[right]+=1; walk(tr.children_left[node],left); walk(tr.children_right[node],right)
        walk(0,tuple())
    rules=[]
    for rule,count in counts.most_common(300):
        if len(rule)>3:continue
        m=np.ones(len(y),bool)
        for f,op,thr in rule:
            col=z[:,f].toarray().ravel(); m &= col<=thr if op=="<=" else col>thr
        if m.mean()<MIN_SUP or m.mean()>.5:continue
        enrichment=target[m].mean()-target.mean()
        if enrichment<=0:continue
        desc=" AND ".join(f"{names[f]} {op} {thr:.2g}" for f,op,thr in rule)
        rules.append({"spec":desc,"mask":m,"frequency":count/1000,"enrichment":enrichment})
        if len(rules)>=K:break
    return rules,pd.DataFrame(curve),alpha

def summarize(rows):
    d=pd.DataFrame(rows); out=[]
    for method,g in d.groupby("method"):
        top=g.sort_values("harm_score",ascending=False).head(K)
        union=np.zeros(len(raw),bool)
        for name in top.slice_name: union |= memberships[(method,name)]
        out.append({"method":method,"K":len(top),"mean_harm":top.harm_score.mean(),"max_harm":top.harm_score.max(),"worst_AUPRC":top.AUPRC.min(),"union_coverage":union.mean(),"mean_support":top.support.mean()})
    return pd.DataFrame(out)

SEARCH_COLS=["age_group","sbp_group","heart_rate_group","resp_group","spo2_group","abdominal_pain","ckd","dm","unclassified","hr_missing"]
start=time.time(); raw=pd.read_csv(DATA,low_memory=False); x=candidate_frame(raw)[SEARCH_COLS]; y=raw.y.to_numpy(int); p=np.clip(raw.oof_pred_proba.to_numpy(float),1e-6,1-1e-6); loss=raw.oof_logloss.to_numpy(float)
rows=[]; memberships={}; specs=[]
sf=faithful_slicefinder(x,loss)
for i,r in enumerate(sf,1):
    name=" & ".join(f"{c}={v}" for c,v in r["spec"]); memberships[("SliceFinder_official_objective",name)]=r["mask"]; rows.append(eval_mask(name,"SliceFinder_official_objective",r["mask"],y,p,loss)); specs.append({"method":"SliceFinder_official_objective","rank":i,"slice_name":name,"algorithm_statistic":r["effect_size"]})
print("SliceFinder done",flush=True)
sl=official_sliceline(x,loss)
for i,r in enumerate(sl,1):
    name=" & ".join(f"{c}={v}" for c,v in r["spec"]); memberships[("SliceLine_official_0.3.0",name)]=r["mask"]; rows.append(eval_mask(name,"SliceLine_official_0.3.0",r["mask"],y,p,loss)); specs.append({"method":"SliceLine_official_0.3.0","rank":i,"slice_name":name,"algorithm_statistic":r["slice_score"]})
print("SliceLine done",flush=True)
af,curve,alpha=afisp_protocol(x,y,p,loss); curve.to_csv(OUT/"afisp_stability_curve.csv",index=False)
for i,r in enumerate(af,1):
    name=r["spec"]; memberships[("AFISP_published_protocol",name)]=r["mask"]; rows.append(eval_mask(name,"AFISP_published_protocol",r["mask"],y,p,loss)); specs.append({"method":"AFISP_published_protocol","rank":i,"slice_name":name,"algorithm_statistic":r["enrichment"]})
print("AFISP done",flush=True)
res=pd.DataFrame(rows); res.to_csv(OUT/"faithful_comparator_slice_results.csv",index=False); pd.DataFrame(specs).to_csv(OUT/"faithful_comparator_top_slices.csv",index=False)
summary=summarize(rows); summary.to_csv(OUT/"faithful_comparator_summary.csv",index=False)
adapted=pd.read_csv(ROOT/"basic_slice_prem"/"Results"/"table_X_comparative_method_summary.csv"); adapted.to_csv(OUT/"adapted_strategy_summary_for_reference.csv",index=False)
manifest={"rows":len(raw),"candidate_features":list(x.columns),"min_support":MIN_SUP,"K":K,"slicefinder_source":"yeounoh/slicefinder author repository; vectorized compatibility implementation of degree<=2, epsilon=0.2 objective and support ordering","sliceline":"official sliceline 0.3.0, alpha=0.95, max_l=2; alpha increased from default 0.6 because default returned no qualifying slice in this search space","afisp":"published stability-analysis sequence with SIRUS-compatible frequent-path phenotype extraction; no public authors' turnkey code located","afisp_selected_alpha":alpha,"seed":SEED,"elapsed_seconds":time.time()-start}
(OUT/"faithful_comparator_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(summary.to_string(index=False)); print(json.dumps(manifest,indent=2),flush=True)
