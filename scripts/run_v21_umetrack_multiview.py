#!/usr/bin/env python3
"""UmeTrack known-skeleton multi-view on v16 automatic crops."""

from __future__ import annotations
import csv,json,sys,tarfile
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path("/data_all/zzx/egoengine/video_to_spider");OUT=ROOT/"runs/hot3d_hand_diagnosis/public_multiview_hand_v21";V16=ROOT/"runs/hot3d_hand_diagnosis/true_auto_stereo_ba_v16";CLIPS=Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
sys.path.insert(0,str(ROOT/"third_party/UmeTrack"));sys.path.insert(0,"/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo")
from hand_tracking_toolkit import camera
from hand_tracking_toolkit.dataset import warp_image
from lib.common.camera import PinholePlaneCameraModel
from lib.common.hand import LEFT_HAND_INDEX,RIGHT_HAND_INDEX
from lib.models.model_loader import load_pretrained_model
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.tracker.tracker import HandTracker,HandTrackerOpts,InputFrame,ViewData
from lib.tracker.video_pose_data import _load_json,load_hand_model_from_dict
MODEL=ROOT/"third_party/UmeTrack/pretrained_models/pretrained_weights.torch";GENERIC=ROOT/"third_party/UmeTrack/dataset/generic_hand_model.json"
def read_csv(p):
 with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def jload(s):return json.loads(s) if s not in ("",None) else None
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 pred={}
 for name in ["real_automatic_strong_predictions.csv","real_automatic_weak_predictions.csv"]:
  for r in read_csv(V16/name):pred[(r["clip"],int(r["frame"]),r["camera"])]=r
 rows=[]
 with open(ROOT/"runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv") as f:
  for r in csv.DictReader(f):
   if int(r["gap"])>5:rows.append((r["clip"],int(r["frame"])))
 both=[(c,f) for c,f in rows if (c,f,"left") in pred and (c,f,"right") in pred][:10]
 model=load_pretrained_model(str(MODEL));tracker=HandTracker(model,HandTrackerOpts());generic=load_hand_model_from_dict(_load_json(str(GENERIC)))
 out=[]
 for clip,frame in both:
  lr=pred[(clip,frame,"left")];rr=pred[(clip,frame,"right")];right=0 if lr["selected_side"]=="left" else 1
  with tarfile.open(CLIPS/clip,"r") as t:
   imgs={}
   for view,row in [("left",lr),("right",rr)]:
    b=t.extractfile(f"{frame:06d}.image_{'1201-1' if view=='left' else '1201-2'}.jpg").read();imgs[view]=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
  cams={}
  for view,row in [("left",lr),("right",rr)]:
   T=np.array(jload(row["T_world_from_eye"]));f256=float(row["f"]);f96=f256*96/256
   # use HOT3D pinhole to warp 96 crop from raw
   # raw cam needed; get from tar cameras
   with tarfile.open(CLIPS/clip,"r") as t:camsj=json.load(t.extractfile(f"{frame:06d}.cameras.json"))
   raw=camera.from_json(camsj["1201-1" if view=="left" else "1201-2"])
   hcam=camera.PinholePlaneCameraModel(width=96,height=96,f=(f96,f96),c=(47.5,47.5),distort_coeffs=[],T_world_from_eye=T)
   crop=warp_image(raw,hcam,imgs[view]);crop=cv2.cvtColor(crop.astype(np.uint8),cv2.COLOR_BGR2GRAY)
   Tmm=T.copy();Tmm[:3,3]*=1000
   uc=PinholePlaneCameraModel(width=96,height=96,f=(f96,f96),c=(47.5,47.5),distort_coeffs=[],camera_to_world_xf=Tmm)
   cams[view]=(crop,uc)
  input_frame=InputFrame(views=[ViewData(image=cams["left"][0],camera=cams["left"][1],camera_angle=0.0),ViewData(image=cams["right"][0],camera=cams["right"][1],camera_angle=0.0)])
  hand_idx=RIGHT_HAND_INDEX if right else LEFT_HAND_INDEX
  crop_cams={hand_idx:{0:cams["left"][1],1:cams["right"][1]}}
  try:
   with torch.no_grad():res=tracker.track_frame(input_frame,generic,crop_cams)
  except Exception as e:
   print("skip",clip,frame,e,file=sys.stderr);continue
  if hand_idx in res.hand_poses:
   lm=landmarks_from_hand_pose(generic,res.hand_poses[hand_idx],hand_idx)
   out.append({"clip":clip,"frame":frame,"landmarks_world_mm":json.dumps(lm.tolist())})
 out_path=OUT/"umetrack_multiview_known.csv"
 with out_path.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(out[0].keys()));w.writeheader();w.writerows(out)
 print("wrote",out_path,"rows",len(out),flush=True)
if __name__=="__main__":raise SystemExit(main())
