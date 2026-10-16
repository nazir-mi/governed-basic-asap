"""Paired profile-cluster bootstrap of equal-weight outer-fold summaries.

Integer profile multiplicities are used instead of expanding resampled records.
Predictions are frozen; this does not repeat training or repair selection.
"""
from pathlib import Path
import sys, json, os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
# Use the pinned active environment.
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
OUT=ROOT/'Local_Outputs/attribute_free_fold_aligned'
MODELS=['Logistic Regression','Random Forest','XGBoost','LightGBM']
ARMS=['ERM','BASiC_ASAP','Class-balanced','Focal-style','CVaR-DRO']

class Scores:
    def __init__(self,y,p,indices=None):
        self.indices=indices
        self.y=np.asarray(y)
        self.p=np.clip(np.asarray(p),1e-6,1-1e-6)
        self.order=np.argsort(-self.p,kind='stable')
        pp=self.p[self.order]
        self.ends=np.r_[np.flatnonzero(np.diff(pp)!=0),len(pp)-1]
        self.loss=-(self.y*np.log(self.p)+(1-self.y)*np.log(1-self.p))
        self.lossorder=np.argsort(self.loss,kind='stable')
    def ap(self,weights):
        if self.indices is not None:weights=weights[self.indices]
        w=weights[self.order]
        tp=np.cumsum(w*self.y[self.order])[self.ends]
        den=np.cumsum(w)[self.ends]
        if tp[-1]==0:return np.nan
        precision=np.divide(tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)
        return float(np.sum(np.diff(np.r_[0,tp])*precision)/tp[-1])
    def global_values(self,w):
        ap=self.ap(w)
        total=w.sum()
        brier=float(np.dot(w,(self.p-self.y)**2)/total)
        sw=w[self.lossorder]; sl=self.loss[self.lossorder]
        cum=np.cumsum(sw)
        rank=.9*(total-1)
        low=sl[np.searchsorted(cum,np.floor(rank),side='right')]
        high=sl[np.searchsorted(cum,np.ceil(rank),side='right')]
        threshold=low+(high-low)*(rank-np.floor(rank))
        mask=self.loss>=threshold
        tail=float(np.dot(w[mask],self.loss[mask])/w[mask].sum())
        return np.array([ap,brier,tail])

def unit_test():
    rng=np.random.default_rng(12)
    for n in [30,103]:
        y=rng.integers(0,2,n);p=np.round(rng.uniform(.01,.99,n),1)
        w=rng.integers(0,5,n);idx=np.repeat(np.arange(n),w)
        s=Scores(y,p)
        vals=s.global_values(w)
        loss=-(y[idx]*np.log(np.clip(p[idx],1e-6,1-1e-6))+(1-y[idx])*np.log(1-np.clip(p[idx],1e-6,1-1e-6)))
        expect=loss[loss>=np.quantile(loss,.9)].mean()
        assert np.isclose(vals[0],average_precision_score(y,p,sample_weight=w))
        assert np.isclose(vals[2],expect)
    print('Weighted AP and quantile-tail calculations match explicitly expanded samples',flush=True)

def run(models=MODELS):
    unit_test()
    sm=pd.read_csv(ROOT/'basic_slice_prem/Notebooks/slice_membership.csv').astype(bool)
    B=400
    for mi,model in enumerate(MODELS):
        if model not in models:continue
        target=OUT/('bootstrap_'+model.replace(' ','_')+'.csv')
        if target.exists():print('SKIP',target.name,flush=True);continue
        rng=np.random.default_rng(20260828+mi)
        metricfiles=[OUT/f'checkpoints/{model.replace(" ","_")}_fold{k}_metrics.csv' for k in range(1,6)]
        metrics=pd.concat([pd.read_csv(p) for p in metricfiles])
        means=metrics.groupby('arm')[['AUPRC','Brier','CVaR_10pct','registry_worst_AUPRC']].mean()
        expected=means.subtract(means.loc['ERM']).rename(columns={
            'AUPRC':'delta_global_AUPRC','Brier':'delta_global_Brier',
            'CVaR_10pct':'delta_CVaR_10pct','registry_worst_AUPRC':'delta_registry_worst_AUPRC'})
        delta=np.zeros((B,4,4))
        point=np.zeros((4,4))
        for fold in range(1,6):
            slug=model.replace(' ','_')+f'_fold{fold}'
            df=pd.read_csv(OUT/f'local_audit/{slug}_predictions.csv')
            ref=df[df.arm.eq('ERM')].sort_values('source_row')
            source=ref.source_row.to_numpy(int);y=ref.y.to_numpy(int)
            g=ref.profile_group.to_numpy('uint64');unique,inverse=np.unique(g,return_inverse=True)
            masks=sm.iloc[source].reset_index(drop=True)
            usable=[c for c in masks if masks[c].sum()>=50 and y[masks[c]].sum()>=10 and np.unique(y[masks[c]]).size==2]
            scores=[];slices=[]
            for arm in ARMS:
                part=df[df.arm.eq(arm)].sort_values('source_row')
                assert np.array_equal(part.source_row,source)
                p=part.probability.to_numpy()
                scores.append(Scores(y,p))
                slices.append([Scores(y[masks[c]],p[masks[c]],np.flatnonzero(masks[c])) for c in usable])
            def metric_values(w):
                values=[]
                for s,ss in zip(scores,slices):
                    glob=s.global_values(w)
                    ap=np.array([v.ap(w) for v in ss])
                    if not np.isfinite(ap).all():raise ValueError('Bootstrap sample has an unevaluable fixed slice; do not silently alter registry')
                    values.append(np.r_[glob,np.min(ap)])
                v=np.array(values)
                return v[1:]-v[0]
            point+=metric_values(np.ones(len(y),dtype=int))/5
            for b in range(B):
                sampled=rng.integers(0,len(unique),len(unique))
                multiplicity=np.bincount(sampled,minlength=len(unique))
                delta[b]+=metric_values(multiplicity[inverse])/5
            print('BOOTSTRAP',model,'fold',fold,'complete',flush=True)
        keys=['delta_global_AUPRC','delta_global_Brier','delta_CVaR_10pct','delta_registry_worst_AUPRC']
        rows=[]
        for ai,arm in enumerate(ARMS[1:]):
            for ki,key in enumerate(keys):
                assert np.isclose(point[ai,ki],expected.loc[arm,key],atol=1e-10),(model,arm,key)
                vals=delta[:,ai,ki]
                rows.append({'model':model,'arm':arm,'metric':key,'point_delta':point[ai,ki],
                    'bootstrap_mean':vals.mean(),'ci_low':np.quantile(vals,.025),'ci_high':np.quantile(vals,.975),
                    'replicates':B,'resampling_unit':'exact_predictor_profile_within_outer_fold',
                    'estimand':'difference_of_equal_weight_five_fold_metric_means',
                    'conditions':'frozen_predictions_fixed_registry_not_selection_aware'})
        pd.DataFrame(rows).to_csv(target,index=False)
    targets=[OUT/('bootstrap_'+m.replace(' ','_')+'.csv') for m in MODELS]
    if all(p.exists() for p in targets):
        pd.concat([pd.read_csv(p) for p in targets]).to_csv(OUT/'attribute_free_profile_cluster_intervals.csv',index=False)

if __name__=='__main__':
    if '--test' in sys.argv:unit_test()
    elif '--model' in sys.argv:run([sys.argv[sys.argv.index('--model')+1]])
    else:run()
