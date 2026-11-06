#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 7.1: P2Bridge-MLCA-TD original vs source-aware 2x2+C2-2048 validation."""
import argparse, csv, gc, hashlib, json, math, shutil, time
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(__file__).resolve().parent
DROOT = ROOT / "datasets/HighwayVehicle-v1.0"
DEFAULTS = {
    "weights": ROOT / "runs/detect/stage3_fast_screen/YOLO11s-DySample-P2Bridge-MLCA-TD/weights/best.pt",
    "image_dir": DROOT / "images/val",
    "gt": ROOT / "runs/detect/source_validation/Stage5-Baseline-Unified/annotations_auto_detect_val.json",
    "source_map": ROOT / "runs/detect/stage6_selective_oracle/image_source_mapping.csv",
    "out": ROOT / "runs/detect/stage7_p2bridge_slice/stage7_1_validation",
}
IMGSZ, CONF, NMS_IOU, MAX_DET, OVERLAP = 640, 0.001, 0.7, 300, 0.15
TILE_AREA = 2048.0
EXPECTED_COUNTS = {"uavdt": 1320, "visdrone": 247}
IOU_THR = np.linspace(.50, .95, 10, dtype=np.float64)
REC_THR = np.linspace(0., 1., 101, dtype=np.float64)
MAX_DETS = [1, 10, 100]
AREA_RNG = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2], [96**2, 1e5**2]]
AREA_LABELS = ["all", "small", "medium", "large"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(path)

def load_json(path):
    with open(path, encoding="utf-8") as f: return json.load(f)

def load_records(gt_path, image_dir, map_path):
    gt = load_json(gt_path)
    if len(gt["images"]) != sum(EXPECTED_COUNTS.values()):
        raise RuntimeError(f"GT图片数异常：{len(gt['images'])}")
    if len(gt["categories"]) != 1:
        raise RuntimeError("当前实验要求单类别COCO GT")
    cat_id = int(gt["categories"][0]["id"])
    cat_name = str(gt["categories"][0]["name"])
    source_map, id_map = {}, {}
    with open(map_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            name, src, iid = Path(r["file_name"]).name, r["source"].lower(), int(r["image_id"])
            if name in source_map or iid in id_map: raise RuntimeError("source map存在重复文件名或image_id")
            if src not in EXPECTED_COUNTS: raise RuntimeError(f"未知来源：{src}")
            source_map[name], id_map[iid] = src, name
    files = {}
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}:
            if p.name in files: raise RuntimeError(f"验证图片重名：{p.name}")
            files[p.name] = p
    records, counts, gt_ids = [], Counter(), set()
    for x in gt["images"]:
        iid, name = int(x["id"]), Path(x["file_name"]).name
        if iid in gt_ids: raise RuntimeError(f"GT image_id重复：{iid}")
        gt_ids.add(iid)
        if name not in files or name not in source_map: raise FileNotFoundError(f"图片或来源映射缺失：{name}")
        src = source_map[name]
        records.append({"image_id":iid,"file_name":name,"path":files[name],
                        "width":int(x["width"]),"height":int(x["height"]),"source":src})
        counts[src] += 1
    if dict(counts) != EXPECTED_COUNTS: raise RuntimeError(f"来源数量不符：{dict(counts)}")
    if gt_ids != set(id_map): raise RuntimeError("GT image_id与source map不一致")
    return gt, records, cat_id, cat_name, dict(counts)

def empty():
    return (np.empty((0,4),np.float32),np.empty((0,),np.float32),np.empty((0,),np.int64))

def extract(res, ox=0, oy=0):
    if res.boxes is None or len(res.boxes)==0: return empty()
    b=res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    s=res.boxes.conf.detach().cpu().numpy().astype(np.float32)
    c=res.boxes.cls.detach().cpu().numpy().astype(np.int64)
    b[:,[0,2]] += ox; b[:,[1,3]] += oy
    return b,s,c

def nms_merge(boxes,scores,classes,w,h,torch,batched_nms):
    if len(boxes)==0: return empty()
    b=np.asarray(boxes,np.float32).reshape(-1,4).copy()
    s=np.asarray(scores,np.float32).reshape(-1); c=np.asarray(classes,np.int64).reshape(-1)
    b[:,[0,2]]=np.clip(b[:,[0,2]],0,w); b[:,[1,3]]=np.clip(b[:,[1,3]],0,h)
    valid=(b[:,2]>b[:,0])&(b[:,3]>b[:,1])&np.isfinite(s)&np.isfinite(b).all(axis=1)
    b,s,c=b[valid],s[valid],c[valid]
    if len(b)==0: return empty()
    keep=batched_nms(torch.from_numpy(b),torch.from_numpy(s),torch.from_numpy(c),NMS_IOU)[:MAX_DET].cpu().numpy()
    return b[keep],s[keep],c[keep]

def pred_args(device):
    return dict(imgsz=IMGSZ,conf=CONF,iou=NMS_IOU,max_det=MAX_DET,device=device,
                half=False,rect=False,augment=False,agnostic_nms=False,verbose=False,save=False)

def predict_original(model,img,args,torch,bnms):
    r=model.predict(source=img,batch=1,**args)[0]
    b,s,c=extract(r); h,w=img.shape[:2]
    return nms_merge(b,s,c,w,h,torch,bnms)

def make_tiles(img):
    h,w=img.shape[:2]
    tw=min(math.ceil(w/(2-OVERLAP)),w); th=min(math.ceil(h/(2-OVERLAP)),h)
    xs=sorted(set([0,max(0,w-tw)])); ys=sorted(set([0,max(0,h-th)]))
    tiles,offs=[],[]
    for y in ys:
        for x in xs: tiles.append(img[y:y+th,x:x+tw].copy()); offs.append((x,y))
    if len(tiles)!=4: raise RuntimeError(f"切片数不是4：{len(tiles)}")
    return tiles,offs

def predict_tiles(model,img,args,torch,bnms):
    tiles,offs=make_tiles(img)
    results=model.predict(source=tiles,batch=4,**args)
    if len(results)!=4: raise RuntimeError("切片推理结果数不是4")
    bs,ss,cs=[],[],[]
    for r,(ox,oy) in zip(results,offs):
        b,s,c=extract(r,ox,oy)
        if len(b): bs.append(b); ss.append(s); cs.append(c)
    if not bs: return empty()
    h,w=img.shape[:2]
    return nms_merge(np.concatenate(bs),np.concatenate(ss),np.concatenate(cs),w,h,torch,bnms)

def fuse_c2(original,tiled,w,h,torch,bnms):
    b0,s0,c0=original; b1,s1,c1=tiled
    if len(b1):
        keep=((b1[:,2]-b1[:,0])*(b1[:,3]-b1[:,1])) < TILE_AREA
        b1,s1,c1=b1[keep],s1[keep],c1[keep]
    if not len(b1): return original
    if not len(b0): b,s,c=b1,s1,c1
    else: b,s,c=np.concatenate([b0,b1]),np.concatenate([s0,s1]),np.concatenate([c0,c1])
    return nms_merge(b,s,c,w,h,torch,bnms)

def coco_rows(iid,result,cat_id):
    b,s,c=result; out=[]
    for box,score,cls in zip(b,s,c):
        if int(cls)!=0: raise RuntimeError(f"单类别模型出现非0类别：{iid}/{cls}")
        x1,y1,x2,y2=map(float,box)
        if x2>x1 and y2>y1:
            out.append({"image_id":iid,"category_id":cat_id,"bbox":[x1,y1,x2-x1,y2-y1],"score":float(score)})
    return out

def sync(torch,device):
    if torch.cuda.is_available() and str(device).lower() not in {"cpu","-1"}: torch.cuda.synchronize()

def eval_scope(gt_path,pred_path,ids,cat_id):
    gt=COCO(str(gt_path)); dt=gt.loadRes(str(pred_path))
    ev=COCOeval_faster(gt,dt,iouType="bbox",lvis_style=False,print_function=print)
    ev.params.imgIds=sorted(ids); ev.params.catIds=[cat_id]
    ev.params.iouThrs=IOU_THR.copy(); ev.params.recThrs=REC_THR.copy()
    ev.params.maxDets=MAX_DETS.copy(); ev.params.areaRng=[x.copy() for x in AREA_RNG]
    ev.params.areaRngLbl=AREA_LABELS.copy(); ev.params.useCats=1
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s=ev.stats_as_dict
    return {"images":len(ids),"AP50-95":float(s["AP_all"]),"AP50":float(s["AP_50"]),
            "AP75":float(s["AP_75"]),"APs":float(s["AP_small"]),
            "APm":float(s["AP_medium"]),"APl":float(s["AP_large"])}

def main():
    ap=argparse.ArgumentParser()
    for k,p in DEFAULTS.items(): ap.add_argument("--"+k.replace("_","-"),default=str(p))
    ap.add_argument("--device",default="0"); ap.add_argument("--warmup",type=int,default=3)
    ap.add_argument("--resume",action="store_true"); ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()
    if a.resume and a.overwrite: ap.error("--resume与--overwrite不能同时使用")
    import cv2, torch, torchvision
    from torchvision.ops import batched_nms
    from ultralytics import YOLO
    weights,image_dir,gt_path,map_path,out=map(lambda x:Path(x).expanduser().resolve(),
        [a.weights,a.image_dir,a.gt,a.source_map,a.out])
    for p in [weights,image_dir,gt_path,map_path]:
        if not p.exists(): raise FileNotFoundError(p)
    if version("faster-coco-eval")!="1.8.0": raise RuntimeError("要求faster-coco-eval==1.8.0")
    if out.exists() and any(out.iterdir()):
        if a.overwrite: shutil.rmtree(out)
        elif not a.resume: raise FileExistsError(f"{out}已有文件；续跑用--resume，重做用--overwrite")
    out.mkdir(parents=True,exist_ok=True)
    gt,records,cat_id,cat_name,counts=load_records(gt_path,image_dir,map_path)
    whash,ghash,mhash=sha256(weights),sha256(gt_path),sha256(map_path)
    model=YOLO(str(weights))
    if len(model.names)!=1: raise RuntimeError(f"模型类别数不是1：{model.names}")
    args=pred_args(a.device)
    # Warm-up
    first=cv2.imread(str(records[0]["path"]))
    vis=cv2.imread(str(next(r for r in records if r["source"]=="visdrone")["path"]))
    if first is None or vis is None: raise RuntimeError("warmup图像读取失败")
    tiles,_=make_tiles(vis)
    for _ in range(a.warmup):
        sync(torch,a.device); model.predict(source=first,batch=1,**args); sync(torch,a.device)
        model.predict(source=tiles,batch=4,**args); sync(torch,a.device)
    del first,vis,tiles; gc.collect()

    ckpt=out/"inference_checkpoint.json"
    pa,pb,per,done=[],[],[],set()
    if a.resume and ckpt.is_file():
        old=json.load(open(ckpt,encoding="utf-8"))
        if (old["weights_sha256"],old["gt_sha256"],old["source_map_sha256"])!=(whash,ghash,mhash):
            raise RuntimeError("checkpoint输入哈希不一致，拒绝续跑")
        pa,pb,per=old["predictions_A"],old["predictions_B"],old["per_image"]
        done=set(old["completed_image_ids"])
        if len(done)!=len(per): raise RuntimeError("checkpoint完成数不一致")

    start=time.perf_counter()
    for i,r in enumerate(records,1):
        iid=r["image_id"]
        if iid in done: continue
        t=time.perf_counter(); img=cv2.imread(str(r["path"])); read_ms=(time.perf_counter()-t)*1000
        if img is None: raise RuntimeError(f"读取失败：{r['path']}")
        h,w=img.shape[:2]
        if (w,h)!=(r["width"],r["height"]): raise RuntimeError(f"图像尺寸与GT不一致：{r['file_name']}")
        sync(torch,a.device); t=time.perf_counter()
        orig=predict_original(model,img,args,torch,batched_nms); sync(torch,a.device)
        orig_ms=(time.perf_counter()-t)*1000
        ar=coco_rows(iid,orig,cat_id); pa.extend(ar)
        tile_ms=fuse_ms=0.0
        if r["source"]=="visdrone":
            sync(torch,a.device); t=time.perf_counter()
            tiled=predict_tiles(model,img,args,torch,batched_nms); sync(torch,a.device)
            tile_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter(); fused=fuse_c2(orig,tiled,w,h,torch,batched_nms)
            fuse_ms=(time.perf_counter()-t)*1000
        else: fused=orig
        br=coco_rows(iid,fused,cat_id); pb.extend(br)
        per.append({"image_id":iid,"file_name":r["file_name"],"source":r["source"],
                    "original_count":len(ar),"fused_count":len(br),"read_ms":read_ms,
                    "original_ms":orig_ms,"tile_ms":tile_ms,"fusion_ms":fuse_ms,
                    "tile_applied":int(r["source"]=="visdrone")})
        done.add(iid)
        if len(done)%10==0 or len(done)==len(records):
            save_json(ckpt,{"status":"running","weights_sha256":whash,"gt_sha256":ghash,
                "source_map_sha256":mhash,"completed_image_ids":sorted(done),
                "predictions_A":pa,"predictions_B":pb,"per_image":per})
        if i%20==0 or i==len(records):
            print(f"progress {len(done)}/{len(records)} A={len(pa)} B={len(pb)}",flush=True)

    if len(done)!=len(records): raise RuntimeError("没有覆盖全部验证图片")
    pA,pB=out/"predictions_A_original.json",out/"predictions_B_C2_area2048.json"
    save_json(pA,pa); save_json(pB,pb)
    with (out/"per_image_runtime.csv").open("w",encoding="utf-8-sig",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(per[0])); wr.writeheader(); wr.writerows(per)

    scopes={"full":[r["image_id"] for r in records],
            "uavdt":[r["image_id"] for r in records if r["source"]=="uavdt"],
            "visdrone":[r["image_id"] for r in records if r["source"]=="visdrone"]}
    metrics={}
    for policy,path in [("A_original",pA),("B_C2_area2048",pB)]:
        metrics[policy]={}
        for scope,ids in scopes.items():
            print(f"\nEvaluate {policy} / {scope} / {len(ids)} images",flush=True)
            metrics[policy][scope]=eval_scope(gt_path,path,ids,cat_id)

    delta={}
    for scope in scopes:
        delta[scope]={m:metrics["B_C2_area2048"][scope][m]-metrics["A_original"][scope][m]
                      for m in ["AP50-95","AP50","AP75","APs","APm","APl"]}
    summary={"experiment":"Stage7.1_P2Bridge_MLCA_TD_Original_vs_C2_2048",
        "status":"completed","created_at":datetime.now().isoformat(timespec="seconds"),
        "weights":str(weights),"weights_sha256":whash,
        "dataset":{"split":"validation/development","images":len(records),"source_counts":counts,
                   "annotations":len(gt.get("annotations",[])),"gt_sha256":ghash,"source_map_sha256":mhash},
        "protocol":{"imgsz":IMGSZ,"conf":CONF,"nms_iou":NMS_IOU,"max_det":MAX_DET,
            "overlap":OVERLAP,"tile_layout":"2x2",
            "tile_area_filter":"mapped original-image tile box area < 2048 px^2",
            "source_policy":"UAVDT original-only; VisDrone original + C2_area2048",
            "half":False,"augment":False,"evaluator":"faster_coco_eval",
            "evaluator_version":version("faster-coco-eval"),
            "coco_iou_thresholds":IOU_THR.tolist(),"coco_max_dets":MAX_DETS,
            "coco_area_ranges":AREA_RNG},
        "metrics":metrics,"delta_B_minus_A":delta,
        "prediction_sha256":{"A_original":sha256(pA),"B_C2_area2048":sha256(pB)},
        "runtime":{"inference_loop_seconds":time.perf_counter()-start,
            "mean_original_ms":float(np.mean([x["original_ms"] for x in per])),
            "mean_tile_ms_per_all_image":float(np.mean([x["tile_ms"] for x in per]))},
        "note":"Validation/development result, not untouched independent test. M3 is not used."}
    save_json(out/"metrics_summary.json",summary)
    save_json(ckpt,{"status":"completed","weights_sha256":whash,"gt_sha256":ghash,
        "source_map_sha256":mhash,"completed_image_ids":sorted(done),
        "predictions_A":pa,"predictions_B":pb,"per_image":per})
    print("\nSTAGE 7.1 COMPLETED\n"+json.dumps({"metrics":metrics,"delta":delta},ensure_ascii=False,indent=2))
    print("Output:",out)

if __name__=="__main__": main()
