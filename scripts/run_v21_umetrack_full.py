#!/usr/bin/env python3
"""Run UmeTrack mono/multi-view variants on v16 automatic crop pairs."""

from __future__ import annotations

import csv
import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT=Path("/data_all/zzx/egoengine/video_to_spider")
OUT=ROOT/"runs/hot3d_hand_diagnosis/public_multiview_hand_v21"
V16=ROOT/"runs/hot3d_hand_diagnosis/true_auto_stereo_ba_v16"
CLIPS=Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")

sys.path.insert(0,str(ROOT/"third_party/UmeTrack"))
sys.path.insert(0,"/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo")
from hand_tracking_toolkit import camera
from hand_tracking_toolkit.dataset import warp_image
from lib.common.camera import PinholePlaneCameraModel
from lib.common.hand import LEFT_HAND_INDEX, RIGHT_HAND_INDEX
from lib.models.model_loader import load_pretrained_model
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.tracker.tracker import HandTracker, HandTrackerOpts, InputFrame, ViewData
from lib.tracker.video_pose_data import _load_json, load_hand_model_from_dict

MODEL=ROOT/"third_party/UmeTrack/pretrained_models/pretrained_weights.torch"
GENERIC=ROOT/"third_party/UmeTrack/dataset/generic_hand_model.json"


def read_csv(p):
    with p.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jload(s):
    return json.loads(s) if s not in ("",None) else None


def build_view(raw_cam, row, img):
    T=np.asarray(jload(row["T_world_from_eye"]),dtype=np.float64)
    f256=float(row["f"]); f96=f256*96/256
    hcam=camera.PinholePlaneCameraModel(width=96,height=96,f=(f96,f96),c=(47.5,47.5),distort_coeffs=[],T_world_from_eye=T)
    crop=warp_image(raw_cam,hcam,img)
    crop=cv2.cvtColor(crop.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    Tmm=T.copy(); Tmm[:3,3]*=1000
    uc=PinholePlaneCameraModel(width=96,height=96,f=(f96,f96),c=(47.5,47.5),distort_coeffs=[],camera_to_world_xf=Tmm)
    return crop,uc


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pred={}
    for name in ["real_automatic_strong_predictions.csv","real_automatic_weak_predictions.csv"]:
        for r in read_csv(V16/name):
            pred[(r["clip"],int(r["frame"]),r["camera"])]=r
    rows=[]
    with open(ROOT/"runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv") as f:
        for r in csv.DictReader(f):
            if int(r["gap"])>5:
                rows.append((r["clip"],int(r["frame"])))
    both=[(c,f) for c,f in rows if (c,f,"left") in pred and (c,f,"right") in pred]
    model=load_pretrained_model(str(MODEL))
    tracker=HandTracker(model,HandTrackerOpts())
    generic=load_hand_model_from_dict(_load_json(str(GENERIC)))
    out={"mono_left":[],"mono_right":[],"multi_known":[],"multi_unknown":[]}
    for clip,frame in both:
        lr=pred[(clip,frame,"left")]; rr=pred[(clip,frame,"right")]
        right=0 if lr["selected_side"]=="left" else 1
        hand_idx=RIGHT_HAND_INDEX if right else LEFT_HAND_INDEX
        with tarfile.open(CLIPS/clip,"r") as t:
            camsj=json.load(t.extractfile(f"{frame:06d}.cameras.json"))
            imgs={}
            for view,stream in [("left","1201-1"),("right","1201-2")]:
                b=t.extractfile(f"{frame:06d}.image_{stream}.jpg").read()
                imgs[view]=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
        raw_left=camera.from_json(camsj["1201-1"]); raw_right=camera.from_json(camsj["1201-2"])
        left_view=build_view(raw_left,lr,imgs["left"])
        right_view=build_view(raw_right,rr,imgs["right"])
        def run(crop_cams, views, name):
            try:
                with torch.no_grad():
                    res=tracker.track_frame(InputFrame(views=views),generic,crop_cams)
            except Exception as e:
                print("skip",name,clip,frame,e,file=sys.stderr)
                return
            if hand_idx in res.hand_poses:
                lm=landmarks_from_hand_pose(generic,res.hand_poses[hand_idx],hand_idx)
                out[name].append({"clip":clip,"frame":frame,"landmarks_world_mm":json.dumps(lm.tolist())})
        run({hand_idx:{0:left_view[1]}},[ViewData(image=left_view[0],camera=left_view[1],camera_angle=0.0)],"mono_left")
        run({hand_idx:{0:right_view[1]}},[ViewData(image=right_view[0],camera=right_view[1],camera_angle=0.0)],"mono_right")
        run({hand_idx:{0:left_view[1],1:right_view[1]}},[ViewData(image=left_view[0],camera=left_view[1],camera_angle=0.0),ViewData(image=right_view[0],camera=right_view[1],camera_angle=0.0)],"multi_known")
        # unknown scale calibration on the same two-view input
        try:
            with torch.no_grad():
                res=tracker.track_frame_and_calibrate_scale(InputFrame(views=[ViewData(image=left_view[0],camera=left_view[1],camera_angle=0.0),ViewData(image=right_view[0],camera=right_view[1],camera_angle=0.0)]),{hand_idx:{0:left_view[1],1:right_view[1]}})
        except Exception as e:
            print("skip unknown",clip,frame,e,file=sys.stderr)
            continue
        if hand_idx in res.hand_poses:
            lm=landmarks_from_hand_pose(generic,res.hand_poses[hand_idx],hand_idx)
            out["multi_unknown"].append({"clip":clip,"frame":frame,"landmarks_world_mm":json.dumps(lm.tolist())})
    for name,records in out.items():
        path=OUT/f"umetrack_{name}.csv"
        with path.open("w",newline="",encoding="utf-8") as f:
            if not records:
                f.write("clip,frame,landmarks_world_mm\n")
            else:
                w=csv.DictWriter(f,fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)
        print(name,len(records),flush=True)


if __name__=="__main__":
    raise SystemExit(main())
