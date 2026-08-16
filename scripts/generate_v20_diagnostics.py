#!/usr/bin/env python3
"""Generate v20 stereo joint correspondence diagnostics from saved v16 predictions."""

from __future__ import annotations

import csv
import json
import math
import sys
import tarfile
from pathlib import Path

import numpy as np


ROOT=Path("/data_all/zzx/egoengine/video_to_spider")
OUT=ROOT/"runs/hot3d_hand_diagnosis/stereo_joint_matching_v20"
V16=ROOT/"runs/hot3d_hand_diagnosis/true_auto_stereo_ba_v16"
V12=ROOT/"runs/hot3d_hand_diagnosis/hand_pose_oracle_v12"
CLIPS=Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"third_party/EgoForce")); sys.path.insert(0,"/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo")
from hand_tracking_toolkit import camera
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
from scripts.evaluate_hot3d_fov_ablation import _build_gt_world, MANO_DIR

def read_csv(p):
 with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def jload(s):return json.loads(s) if s not in ("",None) else None
def write_csv(path,rows):
 if not rows:path.write_text("",encoding="utf-8");return
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with path.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writerow({k:k for k in fields});w.writerows(rows)
def crop_cam(row):
 f=float(row["f"]);T=np.asarray(jload(row["T_world_from_eye"]),dtype=np.float64);return camera.PinholePlaneCameraModel(width=256,height=256,f=(f,f),c=(127.5,127.5),distort_coeffs=[],T_world_from_eye=T)
def ray(cam,uv):
 p=cam.window_to_eye(uv);world=cam.eye_to_world(p);d=world-cam.pos();return cam.pos(),d/np.linalg.norm(d)
def tri(r1,r2):
 c1,d1=r1;c2,d2=r2;a=float(d1@d1);b=float(d1@d2);c=float(d2@d2);den=a*c-b*b
 if abs(den)<1e-12:return None
 w=c1-c2;e=float(d1@w);f=float(d2@w);s=(b*f-c*e)/den;t=(a*f-b*e)/den
 if s<=0 or t<=0:return None
 return 0.5*(c1+s*d1+c2+t*d2)
def mpjpe(pred,gt):return float(np.mean(np.linalg.norm(pred-gt,axis=-1))*1000)
def build_gt(rows):
 manifest=json.load(open(ROOT/"runs/hot3d_hand_diagnosis/subset_manifest.json"));ce={i["clip"]:i for i in manifest["items"]}
 mano=MANOHandModel(str(MANO_DIR));cache={}
 for clip in sorted({c for c,_ in rows}):
  frames,world=_build_gt_world(Path(ce[clip]["run_dir"]),mano);cache[clip]={int(f):world[i] for i,f in enumerate(frames)}
 return cache

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 rows=[]
 with open(ROOT/"runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv") as f:
  for r in csv.DictReader(f):
   if int(r["gap"])>5:rows.append((r["clip"],int(r["frame"])))
 gt=build_gt(rows)
 pred={}
 for name in ["real_automatic_strong_predictions.csv","real_automatic_weak_predictions.csv"]:
  for r in read_csv(V16/name):pred[(r["clip"],int(r["frame"]),r["camera"])]=r
 both=[]
 for clip,frame in rows:
  if (clip,frame,"left") in pred and (clip,frame,"right") in pred:both.append((clip,frame))
 bias_rows=[];cross_rows=[];structured=[];per_joint=[];oracle=[]
 for clip,frame in both:
  lr=pred[(clip,frame,"left")];rr=pred[(clip,frame,"right")];g=gt[clip][frame]
  cl=crop_cam(lr);cr=crop_cam(rr)
  gul=cl.world_to_window(g);gur=cr.world_to_window(g)
  pul=np.asarray(jload(lr["joints_2d_crop"]));pur=np.asarray(jload(rr["joints_2d_crop"]))
  # bias/crossview per joint
  for j in range(21):
   dl=pul[j]-gul[j];dr=pur[j]-gur[j]
   disp_gt=gul[j,0]-gur[j,0];disp_pred=pul[j,0]-pur[j,0];disp_res=disp_pred-disp_gt
   bias_rows.append({"clip":clip,"frame":frame,"joint":j,"left_dx":dl[0],"left_dy":dl[1],"right_dx":dr[0],"right_dy":dr[1]})
   cross_rows.append({"clip":clip,"frame":frame,"joint":j,"left_epe":float(np.linalg.norm(dl)),"right_epe":float(np.linalg.norm(dr)),"gt_disparity":disp_gt,"pred_disparity":disp_pred,"disparity_residual":disp_res})
   per_joint.append({"joint":j,"joint_type":("wrist" if j==0 else "fingertip" if j in [4,8,12,16,20] else "other"),"disparity_residual":disp_res})
  # structured bias experiments using GT uv
  patterns={
   "A_common_direction":np.ones((21,2)),
   "B_opposite_direction":np.array([[1,0]]*21)*np.array([[1 if j%2==0 else -1,1 if j%2==0 else -1] for j in range(21)]),
   "C_disparity_only":np.array([[1,0]]*21),
   "D_vertical_only":np.array([[0,1]]*21),
   "E_fingertip_contract":np.array([[0,0] if j not in [4,8,12,16,20] else [-1,0] for j in range(21)]),
  }
  for off in [5,10,20,30]:
   for name,pat in patterns.items():
    ul=gul+pat*off;ur=gur+(pat if name!="B_opposite_direction" else -pat)*off
    pts=[]
    for j in range(21):
     p=tri(ray(cl,ul[j]),ray(cr,ur[j]));pts.append(p if p is not None else np.full(3,np.nan))
    pts=np.asarray(pts);mask=np.isfinite(pts).all(axis=-1)
    if mask.sum()>=3:structured.append({"clip":clip,"frame":frame,"pattern":name,"offset_px":off,"mpjpe_mm":mpjpe(pts[mask],g[mask])})
  # epipolar correction oracle: fix left WiLoR, search right correction within radius.
  for radius in [5,10,20,40]:
   best=[]
   for j in range(21):
    candidates=[pur[j]+np.array([dx,dy]) for dx in [-radius,0,radius] for dy in [-radius,0,radius]]
    vals=[]
    for cur in candidates:
     p=tri(ray(cl,pul[j]),ray(cr,cur))
     if p is not None:vals.append(np.linalg.norm(p-g[j]))
    best.append(min(vals) if vals else np.inf)
   best=np.asarray(best);mask=np.isfinite(best)
   if mask.sum()>=3:oracle.append({"clip":clip,"frame":frame,"allowed_radius_px":radius,"best_mpjpe_mm":float(np.mean(best[mask]*1000)),"valid_joints":int(mask.sum())})
 write_csv(OUT/"left_right_2d_bias.csv",bias_rows)
 write_csv(OUT/"crossview_joint_error.csv",cross_rows)
 write_csv(OUT/"synthetic_structured_2d_bias_to_3d.csv",structured)
 write_csv(OUT/"per_joint_disparity_error.csv",per_joint)
 write_csv(OUT/"epipolar_correction_oracle.csv",oracle)
 # aggregate oracle by radius
 agg=[]
 for radius in [5,10,20,40]:
  vals=[float(r["best_mpjpe_mm"]) for r in oracle if int(r["allowed_radius_px"])==radius]
  agg.append({"allowed_radius_px":radius,"median_best_mpjpe_mm":float(np.median(vals)) if vals else math.nan,"n":len(vals)})
 write_csv(OUT/"epipolar_correction_oracle_summary.csv",agg)
 print(json.dumps(agg,indent=2))
 print("rows",len(bias_rows),len(cross_rows),len(structured),len(oracle))
 return 0

if __name__=="__main__":
 raise SystemExit(main())
