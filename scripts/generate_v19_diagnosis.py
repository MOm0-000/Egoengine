#!/usr/bin/env python3
"""Generate v19 stereo error diagnosis from saved v16 predictions."""

from __future__ import annotations

import csv
import json
import math
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT=Path("/data_all/zzx/egoengine/video_to_spider")
OUT=ROOT/"runs/hot3d_hand_diagnosis/stereo_error_diagnosis_v19"
V16=ROOT/"runs/hot3d_hand_diagnosis/true_auto_stereo_ba_v16"
V12=ROOT/"runs/hot3d_hand_diagnosis/hand_pose_oracle_v12"
CLIPS=Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"third_party/EgoForce")); sys.path.insert(0,"/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo")
from hand_tracking_toolkit import camera
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
from scripts.evaluate_hot3d_fov_ablation import _build_gt_world, MANO_DIR

def read_csv(p):
 with p.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))
def jload(s): return json.loads(s) if s not in ("",None) else None
def write_csv(path,rows):
 if not rows: path.write_text("",encoding="utf-8"); return
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields: fields.append(k)
 with path.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writerow({k:k for k in fields}); w.writerows(rows)
def crop_cam(row):
 f=float(row["f"]); T=np.asarray(jload(row["T_world_from_eye"]),dtype=np.float64); return camera.PinholePlaneCameraModel(width=256,height=256,f=(f,f),c=(127.5,127.5),distort_coeffs=[],T_world_from_eye=T)
def ray(cam,uv):
 p=cam.window_to_eye(uv); world=cam.eye_to_world(p); d=world-cam.pos(); return cam.pos(), d/np.linalg.norm(d)
def tri(r1,r2):
 c1,d1=r1; c2,d2=r2; a=float(d1@d1); b=float(d1@d2); c=float(d2@d2); den=a*c-b*b
 if abs(den)<1e-12:return None
 w=c1-c2; e=float(d1@w); f=float(d2@w); s=(b*f-c*e)/den; t=(a*f-b*e)/den
 if s<=0 or t<=0:return None
 return 0.5*(c1+s*d1+c2+t*d2)
def mpjpe(pred,gt):
 return float(np.mean(np.linalg.norm(pred-gt,axis=-1))*1000)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 rows=[]
 with open(ROOT/"runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv") as f:
  for r in csv.DictReader(f):
   if int(r["gap"])>5: rows.append((r["clip"],int(r["frame"])))
 manifest=json.load(open(ROOT/"runs/hot3d_hand_diagnosis/subset_manifest.json")); ce={i["clip"]:i for i in manifest["items"]}
 mano=MANOHandModel(str(MANO_DIR)); gt_cache={}
 for clip in sorted({c for c,_ in rows}):
  frames,world=_build_gt_world(Path(ce[clip]["run_dir"]),mano); gt_cache[clip]={int(f):world[i] for i,f in enumerate(frames)}
 strong={}
 for name in ["real_automatic_strong_predictions.csv","real_automatic_weak_predictions.csv"]:
  for r in read_csv(V16/name): strong[(r["clip"],int(r["frame"]),r["camera"])]=r
 v13views={(r["clip"],int(r["frame"]),r["camera"]):r for r in read_csv(ROOT/"runs/hot3d_hand_diagnosis/ego_fullframe_hand_finder_v13/per_view_hand_finder_metrics.csv")}
 # both frames from v16 actual rows
 both=[]
 for clip,frame in rows:
  l=strong.get((clip,frame,"left")); r=strong.get((clip,frame,"right"))
  if l and r: both.append((clip,frame))
 random_rows=[]; replay_rows=[]
 for clip,frame in both:
  lr=strong[(clip,frame,"left")]; rr=strong[(clip,frame,"right")]; gt=gt_cache[clip][frame]
  cl=crop_cam(lr); cr=crop_cam(rr)
  gul=cl.world_to_window(gt); gur=cr.world_to_window(gt)
  pul=np.asarray(jload(lr["joints_2d_crop"])); pur=np.asarray(jload(rr["joints_2d_crop"]))
  # replay
  cases=[("S0_gt",gul,gur),("S1_left_residual",pul,gur),("S2_right_residual",gul,pur),("S3_real_both",pul,pur)]
  for name,ul,ur in cases:
   pts=[]
   for j in range(21):
    p=tri(ray(cl,ul[j]),ray(cr,ur[j])); pts.append(p if p is not None else np.full(3,np.nan))
   pts=np.asarray(pts); mask=np.isfinite(pts).all(axis=-1)
   if mask.sum()>=3: replay_rows.append({"clip":clip,"frame":frame,"case":name,"mpjpe_mm":mpjpe(pts[mask],gt[mask]),"valid_joints":int(mask.sum())})
  # random noise
  for sigma in [2,5,10,20,30,40,60]:
   vals=[]
   for rep in range(50):
    n1=np.random.normal(0,sigma,gul.shape); n2=np.random.normal(0,sigma,gur.shape)
    pts=[]
    for j in range(21):
     p=tri(ray(cl,gul[j]+n1[j]),ray(cr,gur[j]+n2[j])); pts.append(p if p is not None else np.full(3,np.nan))
    pts=np.asarray(pts); mask=np.isfinite(pts).all(axis=-1)
    if mask.sum()>=3: vals.append(mpjpe(pts[mask],gt[mask]))
   random_rows.append({"clip":clip,"frame":frame,"sigma_px":sigma,"median_mpjpe_mm":float(np.median(vals)) if vals else math.nan})
 write_csv(OUT/"synthetic_random_2d_noise_to_3d.csv",random_rows)
 write_csv(OUT/"real_error_replay_stereo.csv",replay_rows)
 # aggregate random by sigma
 agg=[]
 for sigma in [2,5,10,20,30,40,60]:
  vals=[float(r["median_mpjpe_mm"]) for r in random_rows if int(r["sigma_px"])==sigma]
  agg.append({"sigma_px":sigma,"median_mpjpe_mm":float(np.median(vals)) if vals else math.nan,"n":len(vals)})
 write_csv(OUT/"synthetic_random_2d_noise_to_3d_summary.csv",agg)
 print(json.dumps(agg,indent=2))
 print("replay rows",len(replay_rows))
 return 0

if __name__=="__main__":
 raise SystemExit(main())
