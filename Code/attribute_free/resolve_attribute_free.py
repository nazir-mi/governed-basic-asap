"""Fold-aligned, checkpointed re-evaluation; never replaces historical outputs.

Run with the bundled Python. Patient/encounter-level audit files stay local.
The fixed historical registry is a previously developed audit set, not a new
independent discovery cohort. No demographic-fairness inference is made.
"""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '4'
import argparse, ast, hashlib, json, platform, shutil, sys, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
# Install the pinned requirements into the active environment.
import numpy as np
import pandas as pd
import sklearn, xgboost, lightgbm
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits

OUT = ROOT / 'Local_Outputs/attribute_free_fold_aligned'
OLD = ROOT / 'Results/attribute_free_historical'
GOV = ROOT / 'basic_slice_prem/Governed_Leakage_Safe_Results'
DATA = ROOT / 'basic_slice_prem/Data/Raw/triage_cleaned_working.csv'
MEM = ROOT / 'basic_slice_prem/Notebooks/slice_membership.csv'
SOURCE = ROOT / 'Code/attribute_free/attribute_free_metrics.py'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'local_audit').mkdir(exist_ok=True)
(OUT / 'checkpoints').mkdir(exist_ok=True)
SEED = 20260828
CONFIGS = [(0., 0), (.5, 3), (1., 3), (2., 3), (.5, 5), (1., 5), (2., 5)]

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def load_helpers():
    # Execute the unchanged historical definitions without their output-directory
    # creation statements; main is not invoked. Save an exact source snapshot.
    tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'mkdir')]
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Import)
        and any(a.name.startswith('matplotlib') for a in n.names))]
    module = types.ModuleType('historical_attribute_free_helpers')
    module.__file__ = str(SOURCE)
    exec(compile(tree, str(SOURCE), 'exec'), module.__dict__)
    return module

H = load_helpers()
MODELS = H.MODELS
ARMS = ['ERM', 'BASiC_ASAP', *H.BASELINE_ARMS]

def load_data():
    needed=['age','bp_diastole','bp_systolic','heart_rate_final','respiratory','spo2','temperature',
        'abdominal_pain','chest_pain','back_pain','headache','dm','ckd','arm_pain','breast_pain',
        'dental_pain','ear_pain','eye_pain','pelvic_pain','unclassified','htn','chd','y']
    raw = pd.read_csv(DATA, usecols=needed, low_memory=False)
    X = H.transport(raw)
    y = raw.y.to_numpy(int)
    groups = H.profile_groups(X)
    sm = pd.read_csv(MEM).astype('int8')
    assert len(raw) == len(sm) == 204247
    assert set(np.unique(y)) == {0, 1}
    assert not X.columns.isin(['gender', 'sex', 'race', 'ethnicity']).any()
    return X, y, groups, sm

def prepare():
    X, y, groups, sm = load_data()
    # No dependency on the incomplete historical prediction export.
    reference = np.zeros(len(y), dtype='int8')
    for fold, (_, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True,
            random_state=SEED).split(X, y, groups), 1):
        assert not reference[te].any()
        reference[te] = fold
    assert (reference > 0).all()
    manifest = pd.DataFrame({'source_row': np.arange(len(y)),
        'profile_group': groups, 'outer_fold': reference, 'y': y})
    assert manifest.groupby('profile_group').outer_fold.nunique().max() == 1
    counts = []
    for fold in range(1, 6):
        tr, te = np.flatnonzero(reference != fold), np.flatnonzero(reference == fold)
        assert not np.intersect1d(groups[tr], groups[te]).size
        counts.append({'outer_fold': fold, 'n': len(te), 'events': int(y[te].sum()),
            'train_test_profile_overlap': 0})
    path = OUT / 'local_audit/outer_fold_assignment.csv'
    if path.exists():
        prior = pd.read_csv(path)
        assert np.array_equal(prior.outer_fold, reference), 'Never replace a frozen assignment'
    else: manifest.to_csv(path, index=False)
    pd.DataFrame(counts).to_csv(OUT / 'fold_integrity_audit.csv', index=False)
    # Historical output folds cannot be identical when these saved counts differ.
    old = pd.read_csv(OLD / 'attribute_free_outer_fold_results.csv')
    old = old[old.model.eq(MODELS[0]) & old.arm.eq('ERM')][['outer_fold', 'n', 'events']]
    pd.DataFrame(counts).merge(old, on='outer_fold', suffixes=('_aligned', '_historical')).to_csv(
        OUT / 'historical_fold_count_comparison.csv', index=False)
    (OUT / 'Code').mkdir(exist_ok=True)
    shutil.copy2(SOURCE, OUT / 'Code/attribute_free_metrics.py')
    shutil.copy2(__file__, OUT / 'Code/resolve_attribute_free.py')
    protocol = {
        'status': 'protocol frozen before corrected model fitting',
        'date': '2026-10-16', 'seed': SEED, 'models': MODELS, 'arms': ARMS,
        'outer_folds': 'fresh single frozen StratifiedGroupKFold assignment indexed by source_row and exact predictor profiles; shared across all models and arms',
        'historical_prediction_export': 'not required or read by this portable reproduction driver',
        'selection': 'fresh grouped 20% validation split wholly within each outer training set',
        'configurations': CONFIGS,
        'selection_rule': 'among admissible positive-lambda configurations choose greatest mean selected-slice sensitivity, then AUPRC, then lower Brier; otherwise ERM fallback',
        'admissibility': {'AUPRC_loss_max': .005, 'Brier_increase_max': .010},
        'ranking': 'historical basic_ranking; 0.60 harm, 0.30 dual provenance, 0.10 support; fever_temp excluded',
        'repair': 'positive-outcome members only; normalized harm floor 0.1; weight cap 5',
        'baseline_definitions': 'historical two-stage class-balanced, focal-style gamma=2, and CVaR-style worst-20% upweighting; not custom differentiable objectives',
        'matched_randomness': 'identical estimator seed for all arms and all selection candidates within a fold',
        'primary_summary': 'differences between equal-weight means of five outer-fold metrics; fixed evaluable registry per fold',
        'success_rule': 'worst-registry AUPRC delta >0 AND 10% CVaR delta <0 AND global AUPRC delta >=-0.005 AND global Brier delta <=0.010',
        'uncertainty': '400 paired exact-profile cluster bootstrap samples of frozen pooled OOF predictions; conditional on fitted pipelines, not a full selection-aware repeated training analysis',
        'historical_results': 'preserved unchanged; no outcome-based choice of corrected vs historical results',
        'concealed_group': 'separate historical construct experiment is not rerun or silently merged into the corrected comparison',
        'limitations': ['previously developed registry is fixed, not independently rediscovered', 'exact predictor profiles are not patient identifiers', 'non-demographic robustness is not demographic fairness'],
        'versions': {m.__name__: m.__version__ for m in [np, pd, sklearn, xgboost, lightgbm]},
        'python': platform.python_version(),
        'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [DATA, MEM, SOURCE]},
        'outer_assignment_sha256': digest(path),
    }
    pp = OUT / 'protocol.json'
    if not pp.exists(): pp.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps({'prepared': True, 'fold_counts': counts, 'versions': protocol['versions']}, indent=2), flush=True)

def fit(name, X, y, target, seed, weights=None):
    model = H.model_factory(name, seed)
    if 'n_jobs' in model.named_steps['model'].get_params(): model.set_params(model__n_jobs=4)
    with threadpool_limits(limits=4):
        model.fit(X, y, **({} if weights is None else {'model__sample_weight': weights}))
        return np.clip(model.predict_proba(target)[:, 1], 1e-6, 1-1e-6)

def oof(name, X, y, groups, seed):
    pred = np.full(len(y), np.nan)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(splitter.split(X, y, groups), 1):
        assert not np.intersect1d(groups[tr], groups[te]).size
        assert np.isnan(pred[te]).all()
        pred[te] = fit(name, X.iloc[tr], y[tr], X.iloc[te], seed+k)
    assert np.isfinite(pred).all()
    return pred

def select(name, X, y, groups, sm, seed, slug):
    fi, vi = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(X, y, groups))
    assert not np.intersect1d(groups[fi], groups[vi]).size
    diag = oof(name, X.iloc[fi].reset_index(drop=True), y[fi], groups[fi], seed)
    ranking = H.basic_ranking(y[fi], diag, sm.iloc[fi].reset_index(drop=True))
    threshold = H.choose_threshold(y[fi], diag)
    base = fit(name, X.iloc[fi], y[fi], X.iloc[vi], seed)
    bm = H.global_metrics(y[vi], base, threshold)
    rows = []
    for lam, topk in CONFIGS:
        w, names = H.basic_weights(y[fi], sm.iloc[fi].reset_index(drop=True), ranking, lam, topk)
        pred = base if lam == 0 else fit(name, X.iloc[fi], y[fi], X.iloc[vi], seed, w)
        met = H.global_metrics(y[vi], pred, threshold)
        tab = H.slice_metrics(y[vi], pred, threshold, sm.iloc[vi].reset_index(drop=True), names)
        utility = float(tab.Sensitivity.mean()) if not tab.empty else np.nan
        rows.append({'lambda': lam, 'top_k': topk, 'selected_slices': '|'.join(names),
            'target_sensitivity': utility, 'AUPRC': met['AUPRC'], 'Brier': met['Brier'],
            'eligible': met['AUPRC'] >= bm['AUPRC']-.005 and met['Brier'] <= bm['Brier']+.010})
    grid = pd.DataFrame(rows)
    eligible = grid[grid.eligible & grid.target_sensitivity.notna() & grid['lambda'].gt(0)]
    choice = grid.iloc[0] if eligible.empty else eligible.sort_values(
        ['target_sensitivity', 'AUPRC', 'Brier'], ascending=[False, False, True], kind='stable').iloc[0]
    grid.to_csv(OUT / f'checkpoints/{slug}_selection_grid.csv', index=False)
    return float(choice['lambda']), int(choice.top_k), fi, vi

def run(models):
    X, y, groups, sm = load_data()
    assign = pd.read_csv(OUT / 'local_audit/outer_fold_assignment.csv')
    assert np.array_equal(assign.y, y) and np.array_equal(assign.profile_group.to_numpy('uint64'), groups)
    fixed = assign.outer_fold.to_numpy(int)
    for mi, name in enumerate(MODELS):
        if name not in models: continue
        for fold in range(1, 6):
            slug = name.replace(' ', '_') + f'_fold{fold}'
            done = OUT / f'checkpoints/{slug}_complete.json'
            if done.exists(): print('SKIP verified checkpoint', slug, flush=True); continue
            started = time.time()
            tr, te = np.flatnonzero(fixed != fold), np.flatnonzero(fixed == fold)
            xx, yy, gg, mm = X.iloc[tr].reset_index(drop=True), y[tr], groups[tr], sm.iloc[tr].reset_index(drop=True)
            print('START', slug, 'train', len(tr), 'test', len(te), flush=True)
            lam, topk, fi, vi = select(name, xx, yy, gg, mm, SEED+mi*100+fold, slug)
            roles = pd.DataFrame({'source_row': tr, 'inner_selection_role': 'fit'})
            roles.loc[vi, 'inner_selection_role'] = 'validation'
            roles.to_csv(OUT / f'local_audit/{slug}_selection_roles.csv', index=False)
            diag = oof(name, xx, yy, gg, SEED+fold)
            ranking = H.basic_ranking(yy, diag, mm)
            threshold = H.choose_threshold(yy, diag)
            bw, selected = H.basic_weights(yy, mm, ranking, lam, topk)
            if lam == 0: selected = []
            ranking.to_csv(OUT / f'checkpoints/{slug}_training_ranking.csv', index=False)
            weights = {'ERM': None, 'BASiC_ASAP': None if lam == 0 else bw}
            weights.update({arm: H.baseline_weights(arm, yy, None if arm=='Class-balanced' else diag) for arm in H.BASELINE_ARMS})
            predrows, metricrows, slicerows = [], [], []
            base = None
            for arm in ARMS:
                p = base.copy() if arm=='BASiC_ASAP' and lam==0 else fit(name, xx, yy, X.iloc[te], SEED+fold, weights[arm])
                if arm == 'ERM': base = p
                row = {'model': name, 'outer_fold': fold, 'arm': arm, 'n': len(te), 'events': int(y[te].sum()),
                    'threshold': threshold, 'lambda': lam, 'top_k': topk}
                row.update(H.global_metrics(y[te], p, threshold)); row.update(H.tail_metrics(y[te], p))
                alltab = H.slice_metrics(y[te], p, threshold, sm.iloc[te].reset_index(drop=True), list(sm.columns))
                selectedtab = H.slice_metrics(y[te], p, threshold, sm.iloc[te].reset_index(drop=True), selected)
                row.update(H.aggregate_slice_endpoint(alltab, 'registry')); row.update(H.aggregate_slice_endpoint(selectedtab, 'selected'))
                metricrows.append(row)
                slicerows.append(alltab.assign(model=name, outer_fold=fold, arm=arm))
                predrows.append(pd.DataFrame({'model': name, 'outer_fold': fold, 'arm': arm, 'source_row': te,
                    'profile_group': groups[te], 'y': y[te], 'probability': p, 'threshold': threshold}))
                print('ARM', slug, arm, 'AUPRC', round(row['AUPRC'], 6), flush=True)
            pd.DataFrame(metricrows).to_csv(OUT / f'checkpoints/{slug}_metrics.csv', index=False)
            pd.concat(slicerows).to_csv(OUT / f'checkpoints/{slug}_slices.csv', index=False)
            pd.concat(predrows).to_csv(OUT / f'local_audit/{slug}_predictions.csv', index=False)
            done.write_text(json.dumps({'model': name, 'outer_fold': fold, 'lambda': lam, 'top_k': topk,
                'selected_slices': selected, 'threshold': threshold, 'elapsed_seconds': time.time()-started,
                'test_training_profile_overlap': 0, 'inner_fit_validation_profile_overlap': 0,
                'selection_settings_imported': False, 'outer_assignment_sha256': digest(OUT/'local_audit/outer_fold_assignment.csv')}, indent=2), encoding='utf-8')
            print('DONE', slug, 'seconds', round(time.time()-started), 'lambda', lam, 'topk', topk, flush=True)

def summarize():
    completed = sorted((OUT/'checkpoints').glob('*_complete.json'))
    assert len(completed) == 20, f'Only {len(completed)}/20 folds complete'
    folds = pd.concat([pd.read_csv(OUT/'checkpoints'/(p.stem.replace('_complete','')+'_metrics.csv')) for p in completed], ignore_index=True)
    assert len(folds)==100
    summary = H.summarize_baselines(folds)
    folds.to_csv(OUT/'attribute_free_outer_fold_results.csv', index=False)
    summary.to_csv(OUT/'attribute_free_model_summary.csv', index=False)
    pd.concat([pd.read_csv(p) for p in (OUT/'checkpoints').glob('*_slices.csv')]).to_csv(OUT/'attribute_free_registry_slice_results.csv', index=False)
    old = pd.read_csv(OLD/'attribute_free_model_summary.csv')
    cols = ['model','arm','delta_registry_worst_AUPRC','delta_CVaR_10pct','delta_global_AUPRC','delta_global_Brier','global_noninferiority_pass','primary_success']
    comparison = old[cols].merge(summary[cols], on=['model','arm'], suffixes=('_historical','_corrected'))
    comparison.to_csv(OUT/'historical_vs_corrected.csv', index=False)
    print(summary[cols].to_string(index=False), flush=True)
    return summary

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','run','summarize'])
    parser.add_argument('--models', nargs='+', default=MODELS)
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='run': run(args.models)
    else: summarize()
