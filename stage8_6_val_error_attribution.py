#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 8.6: four-model VAL-only complex-scene error attribution.

- Uses the FROZEN 1,567-image Stage 7.1 validation GT + source map.
- Uses Stage 8.1's exact deterministic corruption definitions.
- Uses Stage 7.1's original prediction and COCO evaluator.
- 4 models x (clean + four S3 severity conditions); no training or TEST split.
- Fixed diagnostic matching: score >= 0.25; greedy IoU >= 0.50.
  These TP/FP/FN counts are diagnostics, NOT COCO AP metrics.
- AP50-95/APs use original conf=0.001 predictions and the frozen COCO evaluator.
- Save per-image predictions in append-only JSONL; fsync every 10 images,
  archive completed records to GZIP; support --resume without weight changes.
- Save qualitative four-model panels from VAL images (not for parameter tuning).

Run from YOLO_Thesis root. Requires stage7_1_p2bridge_slice_validation.py
and stage8_1_robustness_4models.py in that directory.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import stage7_1_p2bridge_slice_validation as stage7
import stage8_1_robustness_4models as stage8

ROOT = Path(__file__).resolve().parent
STAGE8_WEIGHTS = ROOT / 'runs/detect/stage8_complex_robustness_4models/model_weight_manifest.json'
STAGE7_REFERENCE = ROOT / 'runs/detect/stage7_p2bridge_slice/stage7_1_validation/metrics_summary.json'
OUT_DEFAULT = ROOT / 'runs/detect/stage8_6_val_error_attribution'

MODEL_PATHS = {
    'YOLO11s': ROOT/'runs/detect/stage2_screening/YOLO11s/weights/best.pt',
    'YOLO11s_DySample': ROOT/'runs/detect/stage2_screening/YOLO11s-DySample/weights/best.pt',
    'YOLO11s_DySample_P2Bridge': ROOT/'runs/detect/stage2_screening/YOLO11s-DySample-P2Bridge/weights/best.pt',
    'YOLO11s_DySample_P2Bridge_MLCA_TD': ROOT/'runs/detect/stage3_fast_screen/YOLO11s-DySample-P2Bridge-MLCA-TD/weights/best.pt',
}
FINAL = 'YOLO11s_DySample_P2Bridge_MLCA_TD'
DEFAULT_CONDITIONS = ['clean', 'blur_s3', 'lowlight_s3', 'backlight_s3', 'fog_s3']
ALL_CONDITIONS = {c['name']: c for c in stage8.CONDITIONS}
EXPECTED = {'uavdt': 1320, 'visdrone': 247}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    temp.replace(path)


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError('Empty CSV: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def iou_matrix(pred_boxes, gt_boxes):
    pred=np.asarray(pred_boxes,dtype=np.float64).reshape(-1,4)
    gt=np.asarray(gt_boxes,dtype=np.float64).reshape(-1,4)
    if not len(pred) or not len(gt):
        return np.zeros((len(pred),len(gt)),dtype=np.float64)
    tl=np.maximum(pred[:,None,:2],gt[None,:,:2])
    br=np.minimum(pred[:,None,2:],gt[None,:,2:])
    inter=np.prod(np.maximum(br-tl,0),axis=-1)
    a=np.prod(np.maximum(pred[:,2:]-pred[:,:2],0),axis=-1)[:,None]
    b=np.prod(np.maximum(gt[:,2:]-gt[:,:2],0),axis=-1)[None,:]
    return inter/np.maximum(a+b-inter,1e-12)


def diagnostic_counts(gt_anns, pred_coco, score_threshold=0.25, match_iou=0.50):
    """Greedy score-ordered match at fixed threshold, for interpretability only."""
    anns=[x for x in gt_anns if not x.get('iscrowd',0) and not x.get('ignore',0)]
    gt=[]; sizes=[]
    for ann in anns:
        x,y,w,h=map(float,ann['bbox'])
        gt.append([x,y,x+w,y+h])
        area=float(ann.get('area',w*h))
        sizes.append('small' if area < 1024 else ('medium' if area < 9216 else 'large'))
    detections=sorted((p for p in pred_coco if float(p['score'])>=score_threshold),
                      key=lambda x:-float(x['score']))
    pb=[]
    for p in detections:
        x,y,w,h=map(float,p['bbox']); pb.append([x,y,x+w,y+h])
    mat=iou_matrix(pb,gt)
    matched=set()
    counts=Counter({'TP':0,'FP':0,'FN':0,'FN_small':0,
                    'FN_medium':0,'FN_large':0,'FP_duplicate':0,
                    'FP_localization_like':0,'FP_background_like':0})
    for i in range(len(detections)):
        if len(gt):
            eligible=np.array([j not in matched for j in range(len(gt))])
            scores=np.where(eligible,mat[i],-1.0)
            j=int(np.argmax(scores))
            if scores[j]>=match_iou:
                matched.add(j)
                counts['TP']+=1
                continue
            best=float(mat[i].max())
            if best>=match_iou:
                counts['FP_duplicate']+=1
            elif best>=0.10:
                counts['FP_localization_like']+=1
            else:
                counts['FP_background_like']+=1
        else:
            counts['FP_background_like']+=1
        counts['FP']+=1
    for j,size in enumerate(sizes):
        if j not in matched:
            counts['FN']+=1
            counts['FN_'+size]+=1
    counts['GT']=len(gt)
    counts['GT_small']=sum(x=='small' for x in sizes)
    counts['GT_medium']=sum(x=='medium' for x in sizes)
    counts['GT_large']=sum(x=='large' for x in sizes)
    counts['predictions_ge_025']=len(detections)
    assert counts['TP']+counts['FN']==counts['GT']
    assert counts['TP']+counts['FP']==counts['predictions_ge_025']
    assert (counts['FP_duplicate']+counts['FP_localization_like']+
            counts['FP_background_like']==counts['FP'])
    return dict(counts)


def make_meta(model,weight_sha,condition,rows,args,input_hashes):
    return {
        'experiment':'Stage8.6_VAL_Error_Attribution',
        'split':'val','model':model,'weight_sha256':weight_sha,
        'condition':condition,
        'n_images':len(rows),
        'image_ids_sha256':hashlib.sha256(','.join(str(r['image_id']) for r in rows).encode()).hexdigest(),
        'input_hashes':input_hashes,
        'diagnostic_conf':args.diag_conf,'diagnostic_iou':args.diag_iou,
        'inference':{'imgsz':640,'conf':0.001,'nms_iou':0.7,'max_det':300,
                     'half':False,'augment':False,'original_only':True},
    }


def read_records(path,expected_sources):
    if not path.exists():
        return []
    records=[]; seen=set(); valid_len=0
    with path.open('rb') as f:
        while True:
            line=f.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                break
            try:
                rec=json.loads(line.decode('utf-8'))
            except (ValueError,UnicodeError):
                break
            iid=int(rec['image_id'])
            if iid not in expected_sources or rec['source']!=expected_sources[iid]:
                raise RuntimeError('Checkpoint contains foreign image_id or source')
            if iid in seen:
                raise RuntimeError('Duplicate checkpoint image_id: '+str(iid))
            if not isinstance(rec.get('predictions'),list) or not isinstance(rec.get('diagnostic'),dict):
                raise RuntimeError('Invalid checkpoint content: '+str(iid))
            seen.add(iid); records.append(rec);valid_len=f.tell()
    if path.stat().st_size!=valid_len:
        with path.open('r+b') as f:
            f.truncate(valid_len)
    return records


def eval_from_records(records,gt_path,cat_id,rows,out_dir,model_key,cond_name,smoke):
    diag_rows=[]
    gt_ids={scope:[int(r['image_id']) for r in rows if scope=='full' or r['source']==scope]
            for scope in ('full','uavdt','visdrone')}
    diagnostics={}
    for scope,ids in gt_ids.items():
        chosen=[r for r in records if scope=='full' or r['source']==scope]
        keys=('TP','FP','FN','FN_small','FN_medium','FN_large','GT',
              'GT_small','GT_medium','GT_large','FP_duplicate',
              'FP_localization_like','FP_background_like','predictions_ge_025')
        totals={k:sum(int(r['diagnostic'].get(k,0)) for r in chosen) for k in keys}
        totals['images']=len(ids)
        totals['fixed_conf_precision']=totals['TP']/max(totals['TP']+totals['FP'],1)
        totals['fixed_conf_recall']=totals['TP']/max(totals['GT'],1)
        totals['small_miss_fraction']=totals['FN_small']/max(totals['GT_small'],1)
        diagnostics[scope]=totals
    coco_metrics={}
    if not smoke:
        preds=[p for r in records for p in r['predictions']]
        if not preds:
            raise RuntimeError('No predictions for COCO evaluator')
        tmp=out_dir/'temporary_predictions'/model_key/(cond_name+'.json')
        write_json(tmp,preds)
        for scope,ids in gt_ids.items():
            print('[COCO]',model_key,cond_name,scope,len(ids),flush=True)
            coco_metrics[scope]=stage7.eval_scope(gt_path,tmp,ids,cat_id)
        tmp.unlink()
    return diagnostics,coco_metrics


def run_one(model,model_key,weight_sha,condition,rows,gt_by_id,cat_id,gt_path,out_dir,
            args,input_hashes,cv2,torch,batched_nms):
    name=condition['name']
    work=out_dir/'checkpoints'/model_key/name
    meta_path=work/'metadata.json'
    file_path=work/'records.jsonl'
    archived=work/'records.jsonl.gz'
    result_path=out_dir/'condition_metrics'/model_key/(name+'.json')
    meta=make_meta(model_key,weight_sha,condition,rows,args,input_hashes)
    if result_path.exists():
        if not args.resume:
            raise FileExistsError('Existing result: '+str(result_path)+'; pass --resume')
        old=read_json(result_path)
        if old.get('metadata')!=meta or old.get('status')!='completed':
            raise RuntimeError('Completed result metadata mismatch: '+str(result_path))
        # Recover if a process stopped between writing the evaluation result
        # and compressing the completed append-only records.
        if not archived.exists():
            if not file_path.exists():
                raise RuntimeError('Result exists but prediction archive is missing: '+str(work))
            recovered=read_records(file_path,{int(r['image_id']):r['source'] for r in rows})
            if len(recovered)!=len(rows):
                raise RuntimeError('Cannot archive partial checkpoint: '+str(work))
            with file_path.open('rb') as src,gzip.open(archived,'wb',compresslevel=6) as dst:
                shutil.copyfileobj(src,dst)
            file_path.unlink()
        print('[SKIP]',model_key,name,flush=True)
        return old
    if meta_path.exists():
        if not args.resume:
            raise FileExistsError('Partial run found; use --resume: '+str(work))
        if read_json(meta_path)!=meta:
            raise RuntimeError('Checkpoint metadata mismatch: '+str(work))
    else:
        if file_path.exists() or archived.exists():
            raise RuntimeError('Record file exists without metadata: '+str(work))
        write_json(meta_path,meta)
    expected_sources={int(r['image_id']):r['source'] for r in rows}
    records=read_records(file_path,expected_sources) if args.resume else []
    already={int(r['image_id']) for r in records}
    pred_args=stage7.pred_args(args.device)
    for i,row in enumerate(rows,1):
        iid=int(row['image_id'])
        if iid in already:
            continue
        image=cv2.imread(str(row['path']))
        if image is None:
            raise RuntimeError('Cannot read image: '+str(row['path']))
        h,w=image.shape[:2]
        if (w,h)!=(row['width'],row['height']):
            raise RuntimeError('GT geometry mismatch: '+str(row['path']))
        transformed=stage8.apply_corruption(image,condition,cv2)
        pred=stage7.predict_original(model,transformed,pred_args,torch,batched_nms)
        coco=stage7.coco_rows(iid,pred,cat_id)
        diag=diagnostic_counts(gt_by_id[iid],coco,args.diag_conf,args.diag_iou)
        rec={'image_id':iid,'source':row['source'],'predictions':coco,'diagnostic':diag}
        with file_path.open('a',encoding='utf-8',buffering=1) as f:
            f.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
            if (len(already)+1)%args.checkpoint_every==0:
                f.flush();os.fsync(f.fileno())
        records.append(rec);already.add(iid)
        if len(already)%100==0 or len(already)==len(rows):
            print('[PROGRESS]',model_key,name,len(already),'/',len(rows),flush=True)
    if len(records)!=len(rows) or already!=set(expected_sources):
        raise RuntimeError('Some validation images are missing')
    diagnostic,coco_metrics=eval_from_records(records,gt_path,cat_id,rows,out_dir,
                                               model_key,name,args.limit_per_source>0)
    result={'status':'completed','created_at':datetime.now().isoformat(timespec='seconds'),
            'metadata':meta,'diagnostic':diagnostic,'coco_metrics':coco_metrics}
    write_json(result_path,result)
    if file_path.exists():
        with file_path.open('rb') as src,gzip.open(archived,'wb',compresslevel=6) as dst:
            shutil.copyfileobj(src,dst)
        file_path.unlink()
    print('[DONE]',model_key,name,'small FN:',diagnostic['full']['FN_small'],flush=True)
    return result


def make_overlay(image,gt_anns,predictions,diag,text,cv2,box_w=720,box_h=480):
    """GT green, predictions red; diagnostic values printed in header."""
    h,w=image.shape[:2]
    scale=min(box_w/w,(box_h-52)/h)
    new_w,max_h=max(1,int(w*scale)),max(1,int(h*scale))
    frame=cv2.resize(image,(new_w,max_h),interpolation=cv2.INTER_AREA)
    panel=np.zeros((box_h,box_w,3),dtype=np.uint8)
    x0=(box_w-new_w)//2;y0=52+(box_h-52-max_h)//2
    panel[y0:y0+max_h,x0:x0+new_w]=frame
    for ann in gt_anns:
        if ann.get('iscrowd',0) or ann.get('ignore',0):
            continue
        x,y,bw,bh=map(float,ann['bbox'])
        a=(int(x0+x*scale),int(y0+y*scale))
        b=(int(x0+(x+bw)*scale),int(y0+(y+bh)*scale))
        cv2.rectangle(panel,a,b,(60,220,50),1)
    for p in predictions:
        if float(p['score'])<0.25:continue
        x,y,bw,bh=map(float,p['bbox'])
        a=(int(x0+x*scale),int(y0+y*scale))
        b=(int(x0+(x+bw)*scale),int(y0+(y+bh)*scale))
        cv2.rectangle(panel,a,b,(30,30,250),1)
    cv2.putText(panel,text[:43],(12,23),cv2.FONT_HERSHEY_SIMPLEX,.52,(250,250,250),1,cv2.LINE_AA)
    cv2.putText(panel,f"TP={diag['TP']} FP={diag['FP']} FN={diag['FN']} smallFN={diag['FN_small']}",
                (12,43),cv2.FONT_HERSHEY_SIMPLEX,.47,(250,250,250),1,cv2.LINE_AA)
    return panel


def generate_examples(results,rows,gt_by_id,out_dir,condition_names,topk,cv2):
    if topk<1:
        return
    row_by_id={r['image_id']:r for r in rows}
    all_models=list(MODEL_PATHS)
    baseline='YOLO11s'
    diag={(model,cond):{} for model in all_models for cond in condition_names}
    for model in all_models:
        for cond in condition_names:
            file=out_dir/'checkpoints'/model/cond/'records.jsonl.gz'
            if not file.exists():
                raise FileNotFoundError(file)
            with gzip.open(file,'rt',encoding='utf-8') as f:
                for line in f:
                    item=json.loads(line)
                    diag[(model,cond)][int(item['image_id'])]=item['diagnostic']['FN_small']
    examples=[]
    for cond in condition_names:
        pairs=[]
        for iid in row_by_id:
            b=diag[(baseline,cond)][iid]
            f=diag[(FINAL,cond)][iid]
            pairs.append((b-f,iid))
        for group,chosen in (
            ('improved',sorted((x for x in pairs if x[0]>0),reverse=True)[:topk]),
            ('worsened',sorted((x for x in pairs if x[0]<0))[:topk]),
        ):
            ids={iid for _,iid in chosen}
            if not ids:
                continue
            pred_by_model={}
            for model in all_models:
                source=out_dir/'checkpoints'/model/cond/'records.jsonl.gz'
                target={}
                with gzip.open(source,'rt',encoding='utf-8') as f:
                    for line in f:
                        r=json.loads(line)
                        if int(r['image_id']) in ids:
                            target[int(r['image_id'])]=r
                if set(target)!=ids:
                    raise RuntimeError('Missing selected prediction')
                pred_by_model[model]=target
            for gain,iid in chosen:
                row=row_by_id[iid]
                image=cv2.imread(str(row['path']))
                image=stage8.apply_corruption(image,ALL_CONDITIONS[cond],cv2)
                panels=[]
                for model in all_models:
                    d=pred_by_model[model][iid]
                    panels.append(make_overlay(image,gt_by_id[iid],d['predictions'],d['diagnostic'],
                                               model,cv2))
                grid=cv2.vconcat([cv2.hconcat(panels[:2]),cv2.hconcat(panels[2:])])
                dest=out_dir/'qualitative_examples'/cond/group/f'{row["source"]}_{iid}.jpg'
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(dest),grid):
                    raise RuntimeError('Failed to save '+str(dest))
                examples.append({'condition':cond,'group':group,'image_id':iid,
                                 'source':row['source'],'small_FN_reduction_vs_YOLO11s':gain,
                                 'filename':str(dest.relative_to(out_dir))})
    if examples:
        write_csv(out_dir/'qualitative_example_index.csv',examples)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='0')
    p.add_argument('--image-dir',type=Path,default=stage7.DEFAULTS['image_dir'])
    p.add_argument('--gt',type=Path,default=stage7.DEFAULTS['gt'])
    p.add_argument('--source-map',type=Path,default=stage7.DEFAULTS['source_map'])
    p.add_argument('--out-dir',type=Path,default=OUT_DEFAULT)
    p.add_argument('--conditions',nargs='+',default=DEFAULT_CONDITIONS,
                   choices=sorted(ALL_CONDITIONS))
    p.add_argument('--diag-conf',type=float,default=0.25)
    p.add_argument('--diag-iou',type=float,default=0.5)
    p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--limit-per-source',type=int,default=0,
                   help='Only for smoke test; e.g. 3 UAVDT + 3 VisDrone')
    p.add_argument('--examples',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args()
    if not (0<args.diag_conf<1 and 0<args.diag_iou<=1):
        p.error('Invalid thresholds')
    if args.checkpoint_every<1 or args.limit_per_source<0 or args.examples<0:
        p.error('Invalid checkpoint-every / limit-per-source / examples')
    if len(set(args.conditions))!=len(args.conditions):
        p.error('Duplicate conditions')
    for required in (args.image_dir,args.gt,args.source_map,STAGE8_WEIGHTS,STAGE7_REFERENCE):
        if not Path(required).exists():
            raise FileNotFoundError(required)
    ref=read_json(STAGE7_REFERENCE)
    assert ref['status']=='completed' and ref['dataset']['images']==1567
    assert ref['dataset']['gt_sha256']==sha256(args.gt)
    assert ref['dataset']['source_map_sha256']==sha256(args.source_map)
    manifest=read_json(STAGE8_WEIGHTS)
    input_hashes={'val_gt':sha256(args.gt),'val_source_map':sha256(args.source_map),
                  'stage7_1_py':sha256(Path(stage7.__file__)),
                  'stage8_1_py':sha256(Path(stage8.__file__))}
    weight_hashes={}
    for name,path in MODEL_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        h=sha256(path)
        if manifest[name]['sha256']!=h or str(path.resolve())!=manifest[name]['path']:
            raise RuntimeError('Stage 8 weight changed: '+name)
        weight_hashes[name]=h
    gt,records,cat_id,_cat_name,counts=stage7.load_records(
        args.gt.resolve(),args.image_dir.resolve(),args.source_map.resolve())
    if counts!=EXPECTED or len(records)!=1567:
        raise RuntimeError('Expected 1,567 validation images')
    rows=sorted(records,key=lambda x:x['image_id'])
    if args.limit_per_source:
        rows=sorted([r for s in EXPECTED for r in [x for x in records if x['source']==s][:args.limit_per_source]],
                    key=lambda x:x['image_id'])
        if len(rows)!=2*args.limit_per_source:
            raise RuntimeError('Smoke subset is incomplete')
    gt_by_id=defaultdict(list)
    for ann in gt['annotations']:
        gt_by_id[int(ann['image_id'])].append(ann)
    for r in rows:
        gt_by_id.setdefault(int(r['image_id']),[])
    print('[SPLIT] val | images',len(rows),'sources:',dict(Counter(r['source'] for r in rows)))
    print('[CONDITIONS]',args.conditions)
    print('[MODELS]',list(MODEL_PATHS))
    import cv2
    import torch
    from torchvision.ops import batched_nms
    from ultralytics import YOLO
    if args.device!='cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu to test on CPU')
    if args.preflight:
        sample=cv2.imread(str(rows[0]['path']))
        assert sample is not None
        for name in args.conditions:
            x=stage8.apply_corruption(sample,ALL_CONDITIONS[name],cv2)
            assert x.shape==sample.shape and x.dtype==np.uint8
        for name,path in MODEL_PATHS.items():
            m=YOLO(str(path))
            if len(m.names)!=1:
                raise RuntimeError('Non-single-class model: '+name)
            _=stage7.predict_original(m,sample,stage7.pred_args(args.device),torch,batched_nms)
            del m
            if torch.cuda.is_available():torch.cuda.empty_cache()
            print('[MODEL OK]',name)
        print('PREFLIGHT PASS | VAL ONLY')
        return
    out=args.out_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    archive_manifest={'split':'val','count':len(rows),
                      'sources':dict(Counter(r['source'] for r in rows)),
                      'conditions':args.conditions,'weights':weight_hashes,
                      'input_hashes':input_hashes,
                      'diag_conf':args.diag_conf,'diag_iou':args.diag_iou,
                      'smoke':bool(args.limit_per_source)}
    mf=out/'experiment_manifest.json'
    if mf.exists():
        if read_json(mf)!=archive_manifest:
            raise RuntimeError('Existing run does not match frozen inputs/conditions')
        if not args.resume:
            raise FileExistsError('Run already exists, pass --resume')
    else:
        write_json(mf,archive_manifest)
    results={}
    for model_key,path in MODEL_PATHS.items():
        print('\n[MODEL]',model_key,flush=True)
        model=YOLO(str(path))
        if len(model.names)!=1:
            raise RuntimeError('Only a single vehicle class is supported')
        results[model_key]={}
        for cname in args.conditions:
            cond=ALL_CONDITIONS[cname]
            results[model_key][cname]=run_one(model,model_key,weight_hashes[model_key],
                cond,rows,gt_by_id,cat_id,args.gt.resolve(),out,args,input_hashes,
                cv2,torch,batched_nms)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    # Positive control: final-model original clean COCO metrics must reproduce
    # the frozen Stage 7.1 validation experiment on the same 1,567 images.
    if not args.limit_per_source and 'clean' in args.conditions:
        for scope in ('full','uavdt','visdrone'):
            previous=ref['metrics']['A_original'][scope]
            current=results[FINAL]['clean']['coco_metrics'][scope]
            for metric in ('AP50-95','APs'):
                if abs(float(current[metric])-float(previous[metric]))>1e-5:
                    raise RuntimeError(
                        f'Stage7.1 clean consistency FAIL: {scope}/{metric}: '
                        f'{current[metric]} vs {previous[metric]}'
                    )
        print('[PASS] Final-model clean COCO metrics match Stage 7.1 VAL reference')
    summaries=[]
    image_rows=[]
    for model,cs in results.items():
        for cname,result in cs.items():
            for scope,tot in result['diagnostic'].items():
                metric=result['coco_metrics'].get(scope,{})
                summaries.append({'model':model,'condition':cname,'scope':scope,
                    'images':tot['images'],'GT':tot['GT'],'GT_small':tot['GT_small'],
                    'TP':tot['TP'],'FP':tot['FP'],'FN':tot['FN'],'FN_small':tot['FN_small'],
                    'FN_medium':tot['FN_medium'],'FN_large':tot['FN_large'],
                    'FP_duplicate':tot['FP_duplicate'],
                    'FP_localization_like':tot['FP_localization_like'],
                    'FP_background_like':tot['FP_background_like'],
                    'fixed_conf_precision':tot['fixed_conf_precision'],
                    'fixed_conf_recall':tot['fixed_conf_recall'],
                    'small_miss_fraction':tot['small_miss_fraction'],
                    'AP50-95':metric.get('AP50-95'),'APs':metric.get('APs')})
            with gzip.open(out/'checkpoints'/model/cname/'records.jsonl.gz','rt',encoding='utf-8') as f:
                for line in f:
                    record=json.loads(line)
                    image_rows.append({'model':model,'condition':cname,
                        'image_id':record['image_id'],'source':record['source'],
                        **record['diagnostic']})
    write_csv(out/'val_error_summary.csv',summaries)
    write_csv(out/'val_per_image_diagnostics.csv',image_rows)
    if not args.limit_per_source:
        generate_examples(results,rows,gt_by_id,out,args.conditions,args.examples,cv2)
    write_json(out/'metrics_summary.json',{
        'status':'completed','split':'val','images':len(rows),
        'smoke':bool(args.limit_per_source),
        'models':list(MODEL_PATHS),'conditions':args.conditions,
        'input_hashes':input_hashes,'weight_hashes':weight_hashes,
        'diagnostic_matching':{'conf':args.diag_conf,'IoU':args.diag_iou,
            'matching':'score-descending greedy, max IoU on unmatched GT',
            'scope':'diagnostic only, not COCO metrics',
            'FP_type':'heuristic duplicate / IoU 0.1-0.5 localization-like / background-like'},
        'coco':'Stage7.1 faster-coco-eval with conf=0.001, IoUs=0.5:0.95',
        'interpretation':'The conditions are synthetic, and source strata are not zero-shot transfer.',
        'rows':summaries})
    print('[COMPLETED] Stage8.6 VAL',len(rows),'images,',len(MODEL_PATHS),'models,',len(args.conditions),'conditions')
    print('summary:',out/'metrics_summary.json')
    print('error table:',out/'val_error_summary.csv')
    print('image diagnostic:',out/'val_per_image_diagnostics.csv')


if __name__=='__main__':
    main()
