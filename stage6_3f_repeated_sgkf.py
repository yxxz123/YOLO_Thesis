#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 6.3F-B: repeated nested StratifiedGroupKFold robustness test for M3/M4.

Run from YOLO_Thesis repository root. No YOLO training/inference and no test-set use.
Five repeated outer 5-fold SGKF; inner 4-fold SGKF; group key=sequence_id.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, math, shutil, time
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / 'runs/detect/stage6_scene_selector'
E3B = SCENE / 'stage6_3e_b_model_comparison'
E3A = SCENE / 'stage6_3e_budget_sweep'
F3A = SCENE / 'stage6_3f_paired_cluster_bootstrap'
OUT = SCENE / 'stage6_3f_repeated_sgkf'
FEATURE = SCENE / 'stage6_3e_b_features/expanded_scene_features.csv'
FEATURE_AUDIT = SCENE / 'stage6_3e_b_features/feature_audit.json'
UTILITY = SCENE / 'coco_utility_analysis.csv'
A_JSON = ROOT / 'runs/detect/stage6_selective_oracle/all_original_predictions.json'
C2_JSON = ROOT / 'runs/detect/stage6_size_aware_fusion_iou07/C2_area2048_predictions.json'
B80_JSON = E3A / 'budget_80_predictions.json'
GT_JSON = ROOT / 'runs/detect/source_validation/Stage5-Baseline-Unified/annotations_auto_detect_val.json'
E3B_SUMMARY = E3B / 'model_comparison_summary.json'
F3A_SUMMARY = F3A / 'paired_cluster_bootstrap_summary.json'
F3A_SAMPLES = F3A / 'paired_cluster_bootstrap_samples.csv'

F0 = ['n_det_conf025', 'median_box_area_640', 'tiny_ratio']
F1 = F0 + ['tiny_count', 'det_density_per_mp', 'mean_confidence_025', 'box_area_cv_640', 'grid_occupancy_4x4']
FEATURES = {'F0': F0, 'F1': F1}
MODELS = {'M3_Logistic_F0': 'F0', 'M4_Logistic_F1': 'F1'}
SEEDS_DEFAULT = [42, 2026, 2027, 3407, 2028]
EXPECTED_SOURCE = {'uavdt': 1320, 'visdrone': 247}
EXPECTED_LABELS = {'beneficial': 158, 'harmful': 89}
EXPECTED_GROUPS = 94
BUDGET, MIN_FRAC, MIN_ABS = 0.80, 0.10, 5
IOU = np.linspace(.50, .95, 10, dtype=np.float64)
RECALL = np.linspace(0., 1., 101, dtype=np.float64)
AREA_RANGES = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2], [96**2, 1e5**2]]
AREA_LABELS = ['all', 'small', 'medium', 'large']
METRICS = ['AP50-95', 'AP50', 'APs', 'APm', 'APl']


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    tmp.replace(path)


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: raise RuntimeError(f'拒绝写入空CSV: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction='raise'); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f: return json.load(f)


def read_csv_id(path: Path) -> dict[int, dict[str, str]]:
    out = {}
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        if 'image_id' not in (r.fieldnames or []): raise RuntimeError(f'{path}缺少image_id')
        for x in r:
            i = int(x['image_id'])
            if i in out: raise RuntimeError(f'{path}重复image_id={i}')
            out[i] = x
    return out


def check_env() -> dict[str, str]:
    e = {'numpy': np.__version__, 'scikit_learn': sklearn.__version__, 'faster_coco_eval': version('faster-coco-eval')}
    want = {'numpy': '1.26.4', 'scikit_learn': '1.5.2', 'faster_coco_eval': '1.8.0'}
    if e != want: raise RuntimeError(f'环境版本不匹配: actual={e}, expected={want}')
    return e


def load_rows():
    frows, urows = read_csv_id(FEATURE), read_csv_id(UTILITY)
    if set(frows) != set(urows) or len(frows) != 1567: raise RuntimeError('扩展特征与utility CSV的image_id不一致或图片数不是1567')
    all_rows, source_counts = {}, Counter()
    for i, f in frows.items():
        u = urows[i]
        for k in ('file_name', 'source', 'sequence_id'):
            if f[k] != u[k]: raise RuntimeError(f'image_id={i}的{k}在特征与标签文件中不一致')
        x = {'image_id': i, 'file_name': f['file_name'], 'source': f['source'].strip().lower(),
             'sequence_id': f['sequence_id'].strip(), 'selector_label': u['selector_label'].strip().lower()}
        for name in F1:
            v = f.get(name, '')
            x[name] = None if v is None or v.strip() == '' else float(v)
            if x[name] is not None and not math.isfinite(x[name]): raise RuntimeError(f'非法特征 {i}/{name}')
        if x['source'] not in EXPECTED_SOURCE: raise RuntimeError(f'未知source={x["source"]}')
        source_counts[x['source']] += 1; all_rows[i] = x
    if dict(source_counts) != EXPECTED_SOURCE: raise RuntimeError(f'source数量错误: {dict(source_counts)}')
    vis = sorted([x for x in all_rows.values() if x['source'] == 'visdrone'], key=lambda x: x['image_id'])
    labels = Counter(x['selector_label'] for x in vis)
    groups = {x['sequence_id'] for x in vis}
    if len(vis) != 247 or dict(labels) != EXPECTED_LABELS or len(groups) != EXPECTED_GROUPS:
        raise RuntimeError(f'VisDrone数据检查失败: n={len(vis)}, labels={dict(labels)}, groups={len(groups)}')
    if any(x['selector_label'] not in ('beneficial', 'harmful') for x in vis): raise RuntimeError('存在未知selector_label')
    return vis, all_rows, {'source_counts': dict(source_counts), 'labels': dict(labels), 'sequences': len(groups)}


def y_bin(rows): return np.asarray([1 if r['selector_label'] == 'beneficial' else 0 for r in rows], dtype=np.int64)


def X_matrix(rows, names):
    return np.asarray([[np.nan if r.get(n) is None else float(r[n]) for n in names] for r in rows], dtype=np.float64)


def configs(): return [{'C': c, 'class_weight': w} for c in (.01, .1, 1., 10.) for w in (None, 'balanced')]


def pipeline(cfg):
    return Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                     ('scaler', StandardScaler()),
                     ('classifier', LogisticRegression(C=float(cfg['C']), class_weight=cfg['class_weight'],
                                                       penalty='l2', solver='lbfgs', max_iter=5000, random_state=42))])


def pos_prob(model, X):
    p = model.predict_proba(X); classes = list(model.named_steps['classifier'].classes_)
    if 1 not in classes: raise RuntimeError(f'分类器缺少beneficial正类: {classes}')
    return p[:, classes.index(1)].astype(float)


def diag(rows, selected):
    labels = [r['selector_label'] for r in rows]; n = int(selected.sum())
    bt = sum(v == 'beneficial' for v in labels); ht = sum(v == 'harmful' for v in labels)
    bs = sum(bool(selected[i]) and labels[i] == 'beneficial' for i in range(len(rows)))
    hs = sum(bool(selected[i]) and labels[i] == 'harmful' for i in range(len(rows)))
    return {'images': len(rows), 'selected_images': n, 'trigger_rate': n/len(rows),
            'beneficial_total': bt, 'harmful_total': ht, 'beneficial_selected': bs, 'harmful_selected': hs,
            'beneficial_precision': bs/n if n else 0., 'beneficial_recall': bs/bt if bt else 0.,
            'harmful_capture_rate': hs/ht if ht else 0., 'net_label_utility': bs-hs,
            'net_label_utility_per_image': (bs-hs)/len(rows)}


def choose_threshold(scores, y):
    if len(scores) != len(y) or not np.all(np.isfinite(scores)): raise RuntimeError('inner OOF概率缺失/非法')
    n = len(y); min_n = max(MIN_ABS, int(math.ceil(MIN_FRAC*n))); max_n = int(math.floor(BUDGET*n)); best = None
    for t in sorted(set(float(v) for v in scores), reverse=True):
        sel = scores >= t; k = int(sel.sum())
        if k < min_n or k > max_n: continue
        b = int(np.sum(y[sel] == 1)); h = int(np.sum(y[sel] == 0)); prec = b/k; net = (b-h)/n
        rank = (net, prec, -k)
        if best is None or rank > best['rank']: best = {'rank': rank, 'threshold': t, 'selected': sel, 'selected_count': k, 'beneficial_selected': b, 'harmful_selected': h, 'beneficial_precision': prec, 'net_label_utility_per_image': net}
    if best is None or best['rank'][0] <= 0: return None, np.zeros(n, dtype=bool), {'selected_count': 0, 'reason': 'no_positive_net_utility_threshold'}
    details = {k:v for k,v in best.items() if k not in ('rank','selected')}; details['trigger_rate'] = details['selected_count']/n
    return float(best['threshold']), best['selected'], details


def class_counts(rows):
    c = Counter(r['selector_label'] for r in rows)
    return {'beneficial': int(c['beneficial']), 'harmful': int(c['harmful'])}


def audit_split(seed, stage, outer, inner, tr_rows, va_rows):
    tg = {x['sequence_id'] for x in tr_rows}; vg = {x['sequence_id'] for x in va_rows}; overlap = tg & vg
    tc, vc = class_counts(tr_rows), class_counts(va_rows)
    if overlap: raise RuntimeError(f'sequence泄漏 seed={seed}, {stage}, overlap={list(overlap)[:5]}')
    if min(tc.values()) == 0 or min(vc.values()) == 0: raise RuntimeError(f'折中缺少某一类别 seed={seed}, {stage}, train={tc}, val={vc}')
    return {'repeat_seed':seed, 'stage':stage, 'outer_fold':outer, 'inner_fold':'' if inner is None else inner,
            'train_images':len(tr_rows), 'validation_images':len(va_rows), 'train_groups':len(tg), 'validation_groups':len(vg),
            'train_beneficial':tc['beneficial'], 'train_harmful':tc['harmful'],
            'validation_beneficial':vc['beneficial'], 'validation_harmful':vc['harmful'],
            'train_beneficial_ratio':tc['beneficial']/len(tr_rows), 'validation_beneficial_ratio':vc['beneficial']/len(va_rows),
            'group_overlap_count':len(overlap)}


def fit_nested(feature_set, tr_rows, te_rows, inner_splits):
    names = FEATURES[feature_set]; Xt, Xv = X_matrix(tr_rows, names), X_matrix(te_rows, names); y = y_bin(tr_rows)
    best, records = None, []
    for cfg in configs():
        scores = np.full(len(tr_rows), np.nan)
        for itr, iva in inner_splits:
            if len(np.unique(y[itr])) != 2: raise RuntimeError('内层训练折缺少类别')
            m = pipeline(cfg); m.fit(Xt[itr], y[itr]); scores[iva] = pos_prob(m, Xt[iva])
        if not np.all(np.isfinite(scores)): raise RuntimeError('内层OOF未完整覆盖outer train')
        threshold, inner_sel, tinfo = choose_threshold(scores, y); d = diag(tr_rows, inner_sel)
        rank = (d['net_label_utility_per_image'], d['beneficial_precision'], -d['selected_images'])
        rec = {'config':cfg, 'threshold':threshold, 'threshold_info':tinfo, 'inner_diag':d, 'rank':rank}
        records.append(rec)
        if best is None or rank > best['rank']: best = rec
    if best is None: raise RuntimeError('没有可用的Logistic配置')
    final = pipeline(best['config']); final.fit(Xt, y); scores = pos_prob(final, Xv)
    selected = np.zeros(len(te_rows), dtype=bool) if best['threshold'] is None else scores >= float(best['threshold'])
    imp, clf = final.named_steps['imputer'], final.named_steps['classifier']
    try: coef_names = imp.get_feature_names_out(names).tolist()
    except Exception: coef_names = list(names)
    details = {'feature_set':feature_set, 'selected_config':best['config'], 'selected_probability_threshold':best['threshold'],
               'inner_training_diagnostics':best['inner_diag'], 'threshold_selection':best['threshold_info'],
               'tested_config_count':len(records), 'all_config_inner_results':[{'config':r['config'],'threshold':r['threshold'],'inner_diag':r['inner_diag'],'ranking':list(r['rank'])} for r in records],
               'coefficient_names':coef_names, 'standardized_coefficients':clf.coef_[0].astype(float).tolist()}
    return selected, scores, details


def read_preds(path, valid_ids):
    raw = load_json(path); grouped = defaultdict(list)
    if not isinstance(raw, list): raise RuntimeError(f'预测文件不是COCO list: {path}')
    for p in raw:
        i = int(p['image_id'])
        if i not in valid_ids or len(p.get('bbox', [])) != 4: raise RuntimeError(f'预测记录非法: {path}/{i}')
        grouped[i].append(p)
    return dict(grouped)


def signature(records):
    return Counter((int(p['category_id']), round(float(p['score']),8), tuple(round(float(v),6) for v in p['bbox'])) for p in records)


def flat(grouped): return [p for i in sorted(grouped) for p in grouped[i]]


def check_uavdt(all_rows, original, c2):
    bad = [i for i,r in all_rows.items() if r['source']=='uavdt' and signature(original.get(i,[])) != signature(c2.get(i,[]))]
    if bad: raise RuntimeError(f'C2与Original在UAVDT上不一致: {bad[:20]}')


def combined(selected_ids, all_rows, original, c2):
    out=[]
    for i in sorted(all_rows):
        use = all_rows[i]['source']=='visdrone' and i in selected_ids
        out.extend(c2.get(i,[]) if use else original.get(i,[]))
    return out


def eval_scope(gt, predictions, ids, cat_ids):
    dt = gt.loadRes(predictions)
    ev = COCOeval_faster(gt, dt, iouType='bbox', lvis_style=False, print_function=lambda *a,**k:None)
    ev.params.imgIds=sorted(ids); ev.params.catIds=list(cat_ids); ev.params.iouThrs=IOU.copy(); ev.params.recThrs=RECALL.copy()
    ev.params.maxDets=[1,10,100]; ev.params.areaRng=[x.copy() for x in AREA_RANGES]; ev.params.areaRngLbl=list(AREA_LABELS); ev.params.useCats=1
    ev.evaluate(); ev.accumulate(); ev.summarize(); s=ev.stats_as_dict
    out={'AP50-95':float(s['AP_all']), 'AP50':float(s['AP_50']), 'APs':float(s['AP_small']), 'APm':float(s['AP_medium']), 'APl':float(s['AP_large'])}
    del ev,dt
    return out


def eval_policy(gt, preds, full_ids, vis_ids, cat_ids):
    return {'full':eval_scope(gt,preds,full_ids,cat_ids), 'visdrone':eval_scope(gt,preds,vis_ids,cat_ids)}


def check_metric(actual, expected, name):
    for scope in ('full','visdrone'):
        for metric in ('AP50-95','APs'):
            if abs(actual[scope][metric]-float(expected[scope][metric])) > 1e-6:
                raise RuntimeError(f'{name}/{scope}/{metric}点估计复现失败: {actual[scope][metric]} vs {expected[scope][metric]}')


def build_repeat(seed, vis_rows, all_rows, original, c2, gt, full_ids, vis_ids, cat_ids, baseline):
    y=y_bin(vis_rows); groups=np.asarray([r['sequence_id'] for r in vis_rows],dtype=object); dummy=np.zeros((len(vis_rows),1))
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=int(seed)); splits=list(outer.split(dummy,y,groups))
    if len(splits)!=5: raise RuntimeError('outer fold数量错误')
    selected={m:np.zeros(len(vis_rows),dtype=bool) for m in MODELS}; scores={m:np.full(len(vis_rows),np.nan) for m in MODELS}; fold_ids={m:np.zeros(len(vis_rows),dtype=int) for m in MODELS}
    seen=np.zeros(len(vis_rows),dtype=int); assignments=[]; folds={m:[] for m in MODELS}; audits=[]
    for of,(tri,tei) in enumerate(splits,1):
        tr=[vis_rows[int(i)] for i in tri]; te=[vis_rows[int(i)] for i in tei]; seen[tei]+=1
        audits.append(audit_split(seed,'outer',of,None,tr,te))
        ytr=y_bin(tr); gtr=np.asarray([r['sequence_id'] for r in tr],dtype=object); dum=np.zeros((len(tr),1))
        inner_seed=int(seed*1000+of*37+11); inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=inner_seed)
        isplits=list(inner.split(dum,ytr,gtr))
        if len(isplits)!=4: raise RuntimeError('inner fold数量错误')
        for inf,(iti,ivi) in enumerate(isplits,1): audits.append(audit_split(seed,'inner',of,inf,[tr[int(i)] for i in iti],[tr[int(i)] for i in ivi]))
        for model,fs in MODELS.items():
            sel,sc,info=fit_nested(fs,tr,te,isplits)
            for li,gi in enumerate(tei): selected[model][int(gi)]=bool(sel[li]); scores[model][int(gi)]=float(sc[li]); fold_ids[model][int(gi)]=of
            folds[model].append({'outer_fold':of,'outer_train_images':len(tr),'outer_test_images':len(te),'outer_train_label_counts':class_counts(tr),'outer_test_label_counts':class_counts(te),'model_selection':info,'outer_test_diagnostics_audit_only':diag(te,sel)})
        for li,gi in enumerate(tei):
            r=vis_rows[int(gi)]
            for model in MODELS:
                info=folds[model][-1]['model_selection']
                assignments.append({'repeat_seed':seed,'outer_fold':int(fold_ids[model][int(gi)]),'model':model,'image_id':r['image_id'],'file_name':r['file_name'],'sequence_id':r['sequence_id'],
                    'selected_for_c2_2048':int(selected[model][int(gi)]),'selector_score':float(scores[model][int(gi)]),
                    'selected_probability_threshold':'' if info['selected_probability_threshold'] is None else info['selected_probability_threshold'],
                    'selected_C':info['selected_config']['C'],'selected_class_weight':'none' if info['selected_config']['class_weight'] is None else info['selected_config']['class_weight'],
                    'selector_label_audit_only':r['selector_label']})
    if not np.all(seen==1): raise RuntimeError(f'OOF覆盖不完整: {Counter(seen.tolist())}')
    metrics={}; metric_rows=[]
    for model in MODELS:
        if not np.all(np.isfinite(scores[model])): raise RuntimeError(f'{model} OOF概率不完整')
        ids={vis_rows[i]['image_id'] for i in range(len(vis_rows)) if selected[model][i]}
        m=eval_policy(gt,combined(ids,all_rows,original,c2),full_ids,vis_ids,cat_ids); d=diag(vis_rows,selected[model])
        m['selection_diagnostics']=d; m['delta_vs_B80']={}; m['delta_vs_C2']={}
        for ref in ('B80','C2_area2048'):
            for scope in ('full','visdrone'):
                for metric in ('AP50-95','APs'):
                    m['delta_vs_B80' if ref=='B80' else 'delta_vs_C2'].setdefault(scope,{})[metric]=m[scope][metric]-baseline[ref][scope][metric]
        metrics[model]=m
        row={'repeat_seed':seed,'model':model,**d}
        for scope in ('full','visdrone'):
            for metric in METRICS: row[f'{scope}_{metric}']=m[scope][metric]
            for ref in ('B80','C2_area2048'):
                for metric in ('AP50-95','APs'): row[f'delta_{scope}_{metric}_vs_{ref}']=m['delta_vs_B80' if ref=='B80' else 'delta_vs_C2'][scope][metric]
        metric_rows.append(row)
    paired={'M4_minus_M3':{scope:{metric:metrics['M4_Logistic_F1'][scope][metric]-metrics['M3_Logistic_F0'][scope][metric] for metric in ('AP50-95','APs')} for scope in ('full','visdrone')}}
    return {'repeat_seed':seed,'status':'completed','assignment_rows':assignments,'fold_details':folds,'split_audit_rows':audits,'metrics_by_model':metrics,'repeat_metrics_rows':metric_rows,'paired_repeat_deltas':paired}


def num_summary(vals):
    a=np.asarray(vals,dtype=float)
    return {'mean':float(np.mean(a)),'std_between_repeats':float(np.std(a,ddof=1)) if len(a)>1 else 0.,'min':float(np.min(a)),'max':float(np.max(a))}


def main():
    ap=argparse.ArgumentParser(description='Repeated nested StratifiedGroupKFold robustness test for M3/M4')
    ap.add_argument('--repeat-seeds',nargs='+',type=int,default=SEEDS_DEFAULT)
    ap.add_argument('--resume',action='store_true'); ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    if args.resume and args.overwrite: ap.error('--resume和--overwrite不能同时使用')
    if len(args.repeat_seeds)<3 or len(set(args.repeat_seeds))!=len(args.repeat_seeds): ap.error('至少3个互不重复seed，建议默认5个')
    env=check_env()
    required=[FEATURE,FEATURE_AUDIT,UTILITY,A_JSON,C2_JSON,B80_JSON,GT_JSON,E3B_SUMMARY,F3A_SUMMARY,F3A_SAMPLES]
    missing=[p for p in required if not p.is_file()]
    if missing: raise FileNotFoundError('缺少输入文件:\n'+'\n'.join(map(str,missing)))
    if OUT.exists() and any(OUT.iterdir()):
        if args.overwrite: shutil.rmtree(OUT)
        elif not args.resume: raise FileExistsError(f'{OUT}已有内容；中断续跑用--resume，重做用--overwrite')
    repeat_dir=OUT/'repeats'; repeat_dir.mkdir(parents=True,exist_ok=True)
    e3b=load_json(E3B_SUMMARY); f3a=load_json(F3A_SUMMARY); audit=load_json(FEATURE_AUDIT)
    if e3b.get('status')!='completed' or f3a.get('status')!='completed' or audit.get('status')!='completed': raise RuntimeError('E-B/F-A/feature audit状态未完成')
    if int(f3a.get('bootstrap',{}).get('replicates',0))!=1000 or f3a.get('m1_b80_prediction_identity',{}).get('status')!='passed': raise RuntimeError('F-A不是有效的1000次完成结果')
    vis,all_rows,data=load_rows(); all_ids=set(all_rows); full_ids=sorted(all_ids); vis_ids=sorted(r['image_id'] for r in vis)
    paths={'expanded_scene_features.csv':FEATURE,'feature_audit.json':FEATURE_AUDIT,'utility_csv':UTILITY,'original_predictions':A_JSON,'c2_predictions':C2_JSON,'b80_predictions':B80_JSON,'ground_truth':GT_JSON,'stage6_3e_b_summary':E3B_SUMMARY,'stage6_3f_a_summary':F3A_SUMMARY,'stage6_3f_a_samples':F3A_SAMPLES}
    hashes={k:sha(v) for k,v in paths.items()}
    for k in ('expanded_scene_features.csv','feature_audit.json','utility_csv','original_predictions','c2_predictions','ground_truth'):
        expected=e3b.get('input_sha256',{}).get(k)
        if expected and expected!=hashes[k]: raise RuntimeError(f'E-B输入哈希不一致: {k}')
    fhash=f3a.get('input_sha256',{})
    for k,v in {'ground_truth':hashes['ground_truth'],'stage6_3e_b_summary':hashes['stage6_3e_b_summary'],'A_Original_prediction':hashes['original_predictions'],'C2_area2048_prediction':hashes['c2_predictions'],'B80_prediction':hashes['b80_predictions']}.items():
        if fhash.get(k) and fhash[k]!=v: raise RuntimeError(f'F-A输入哈希不一致: {k}')
    original=read_preds(A_JSON,all_ids); c2=read_preds(C2_JSON,all_ids); b80=read_preds(B80_JSON,all_ids); check_uavdt(all_rows,original,c2)
    if hashes['b80_predictions']!=fhash.get('B80_prediction'): raise RuntimeError('B80预测与F-A记录的SHA256不一致')
    gt=COCO(str(GT_JSON)); gt_ids=sorted(int(x['id']) for x in gt.dataset['images']); cat_ids=sorted(int(x) for x in gt.getCatIds())
    if gt_ids!=full_ids or not cat_ids: raise RuntimeError('GT image/category索引不匹配')
    print('='*86); print('Stage 6.3F-B | Repeated nested StratifiedGroupKFold'); print('Outer=5, Inner=4, repeats=',args.repeat_seeds); print('Models=M3 F0 / M4 F1; no YOLO training/inference; no independent test'); print('Environment:',env,'Data:',data); print('='*86,flush=True)
    baseline={}
    for name,grouped in [('A_Original',original),('C2_area2048',c2),('B80',b80)]:
        print('[BASELINE]',name,flush=True); baseline[name]=eval_policy(gt,flat(grouped),full_ids,vis_ids,cat_ids)
        if name in f3a.get('point_metrics_reproduction',{}):
            for scope in ('full','visdrone'):
                for metric in ('AP50-95','APs'):
                    if abs(baseline[name][scope][metric]-f3a['point_metrics_reproduction'][name][scope][metric])>1e-6: raise RuntimeError(f'{name}/{scope}/{metric} baseline复现失败')
        print(' Full APs={:.6f} mAP={:.6f}'.format(baseline[name]['full']['APs'],baseline[name]['full']['AP50-95']),flush=True); gc.collect()
    seed_results=[]; start=time.perf_counter()
    for seed in args.repeat_seeds:
        sp=repeat_dir/f'repeat_seed_{seed}.json'
        if args.resume and sp.is_file():
            cached=load_json(sp)
            if cached.get('status')!='completed' or cached.get('input_sha256')!=hashes: raise RuntimeError(f'缓存校验失败: {sp}')
            seed_results.append(cached['result']); print('[RESUME] seed',seed,'loaded',flush=True); continue
        if sp.exists() and not args.overwrite: raise FileExistsError(f'{sp}已存在，使用--resume或--overwrite')
        print('\n[REPEAT] seed=',seed,flush=True)
        result=build_repeat(seed,vis,all_rows,original,c2,gt,full_ids,vis_ids,cat_ids,{'B80':baseline['B80'],'C2_area2048':baseline['C2_area2048']})
        dump_json(sp,{'status':'completed','input_sha256':hashes,'result':result}); seed_results.append(result)
        print('[CHECKPOINT] seed',seed,'elapsed_min={:.1f}'.format((time.perf_counter()-start)/60),flush=True); gc.collect()
    seed_results.sort(key=lambda x:args.repeat_seeds.index(int(x['repeat_seed'])))
    if len(seed_results)!=len(args.repeat_seeds): raise RuntimeError('repeat数量错误')
    # Aggregate repeat-level metrics. SD is split sensitivity, not a CI.
    metric_rows=[r for x in seed_results for r in x['repeat_metrics_rows']]
    summary={'experiment':'Stage6.3F_B_Repeated_Nested_StratifiedGroupKFold','created_at':datetime.now().isoformat(timespec='seconds'),'status':'completed','repeat_seeds':args.repeat_seeds,
      'protocol':{'outer_cv':'5-fold StratifiedGroupKFold, repeated with fixed seeds','inner_cv':'4-fold StratifiedGroupKFold nested in each outer training fold','group_key':'sequence_id','features':{'F0':F0,'F1':F1},'models':list(MODELS),'logistic_grid':configs(),'budget':BUDGET,'minimum_trigger_fraction':MIN_FRAC,'minimum_trigger_absolute':MIN_ABS,'prediction_policy':'UAVDT always A_Original; VisDrone uses C2_area2048 only when selected','no_yolo_training':True,'no_yolo_inference':True,'independent_test_used':False,'evaluator':'faster_coco_eval','environment':env,'repeat_sd_interpretation':'descriptive split sensitivity only; not a confidence interval'},
      'data':data,'baseline_metrics':baseline,'input_sha256':hashes,'repeat_metric_summary':{},'paired_repeat_delta_summary':{},'model_config_stability':{},'selection_stability':{},
      'interpretation_limit':'Repeated nested grouped CV measures sensitivity to sequence-group partition on the 247-image development set. It is not an independent test. SD across five repeats is not a 95% CI.'}
    for model in MODELS:
        mr=[r for r in metric_rows if r['model']==model]; summary['repeat_metric_summary'][model]={}
        for k in ['trigger_rate','beneficial_precision','beneficial_recall','harmful_capture_rate','net_label_utility_per_image','full_AP50-95','full_APs','visdrone_AP50-95','visdrone_APs','delta_full_AP50-95_vs_B80','delta_full_APs_vs_B80','delta_full_AP50-95_vs_C2_area2048','delta_full_APs_vs_C2_area2048']:
            summary['repeat_metric_summary'][model][k]=num_summary([float(r[k]) for r in mr])
        cfgs=[]; thresholds=[]; coefs=defaultdict(list)
        for res in seed_results:
            for fold in res['fold_details'][model]:
                info=fold['model_selection']; cfgs.append(json.dumps(info['selected_config'],sort_keys=True))
                if info['selected_probability_threshold'] is not None: thresholds.append(float(info['selected_probability_threshold']))
                for n,v in zip(info['coefficient_names'],info['standardized_coefficients']): coefs[n].append(float(v))
        summary['model_config_stability'][model]={'selected_config_frequency':dict(Counter(cfgs)),'threshold_summary':num_summary(thresholds) if thresholds else None,'coefficient_summary':{n:num_summary(v) for n,v in coefs.items()}}
    for scope in ('full','visdrone'):
        summary['paired_repeat_delta_summary'].setdefault('M4_minus_M3',{})[scope]={}
        for metric in ('AP50-95','APs'):
            vals=[x['paired_repeat_deltas']['M4_minus_M3'][scope][metric] for x in seed_results]
            summary['paired_repeat_delta_summary']['M4_minus_M3'][scope][metric]={**num_summary(vals),'positive_repeats':sum(v>0 for v in vals),'total_repeats':len(vals)}
    # Per-image selection frequency over repeated outer OOF assignments.
    per={}; label_by={}; seq_by={}
    for res in seed_results:
        for r in res['assignment_rows']:
            k=(r['model'],int(r['image_id'])); per.setdefault(k,[]).append(int(r['selected_for_c2_2048'])); label_by[int(r['image_id'])]=r['selector_label_audit_only']; seq_by[int(r['image_id'])]=r['sequence_id']
    stability=[]
    for model in MODELS:
        for i in sorted(label_by):
            vals=per[(model,i)]
            if len(vals)!=len(seed_results): raise RuntimeError(f'per-image repeat数错误: {model}/{i}')
            stability.append({'model':model,'image_id':i,'sequence_id':seq_by[i],'selector_label_audit_only':label_by[i],'selected_repeats':sum(vals),'total_repeats':len(vals),'selection_frequency':float(np.mean(vals))})
        rows=[r for r in stability if r['model']==model]
        summary['selection_stability'][model]={'images_selected_in_all_repeats':sum(r['selection_frequency']==1 for r in rows),'images_never_selected':sum(r['selection_frequency']==0 for r in rows),'images_with_variable_decisions':sum(0<r['selection_frequency']<1 for r in rows),'mean_per_image_selection_frequency':float(np.mean([r['selection_frequency'] for r in rows]))}
    assignments=[r for x in seed_results for r in x['assignment_rows']]
    folds={m:[{'repeat_seed':x['repeat_seed'],**f} for x in seed_results for f in x['fold_details'][m]] for m in MODELS}
    audits=[r for x in seed_results for r in x['split_audit_rows']]
    dump_csv(OUT/'repeated_oof_assignments.csv',assignments)
    dump_json(OUT/'fold_model_details.json',{'group_key':'sequence_id','outer_splits':5,'inner_splits':4,'repeat_seeds':args.repeat_seeds,'fold_details':folds})
    dump_csv(OUT/'stratified_split_audit.csv',audits); dump_csv(OUT/'repeat_metrics.csv',metric_rows); dump_csv(OUT/'per_image_selection_stability.csv',stability)
    summary['outputs']={'summary_json':str((OUT/'stage6_3f_b_summary.json').relative_to(ROOT)),'repeat_metrics_csv':str((OUT/'repeat_metrics.csv').relative_to(ROOT)),'assignments_csv':str((OUT/'repeated_oof_assignments.csv').relative_to(ROOT)),'fold_details_json':str((OUT/'fold_model_details.json').relative_to(ROOT)),'split_audit_csv':str((OUT/'stratified_split_audit.csv').relative_to(ROOT)),'selection_stability_csv':str((OUT/'per_image_selection_stability.csv').relative_to(ROOT))}
    dump_json(OUT/'stage6_3f_b_summary.json',summary)
    print('\nStage 6.3F-B COMPLETED')
    for model in MODELS:
        x=summary['repeat_metric_summary'][model]
        print(model,'Full APs={:.6f} +/- {:.6f}; mAP={:.6f} +/- {:.6f}'.format(x['full_APs']['mean'],x['full_APs']['std_between_repeats'],x['full_AP50-95']['mean'],x['full_AP50-95']['std_between_repeats']))
    print('Output:',OUT)

if __name__=='__main__': main()
