#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
egoengine_root=${EGOENGINE_ROOT:-$(dirname -- "$repo_root")}
spider_root=${SPIDER_ROOT:-$egoengine_root/spider}
gpu=${GPU:-7}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

check_revision() {
  local path=$1
  local expected=$2
  local actual
  [[ -d "$path/.git" ]] || fail "missing Git checkout: $path"
  actual=$(git -C "$path" rev-parse HEAD)
  [[ "$actual" == "$expected" ]] || fail "revision mismatch: $path expected=$expected actual=$actual"
  printf 'revision ok: %s %s\n' "$path" "$actual"
}

check_file() {
  local path=$1
  [[ -s "$path" ]] || fail "missing or empty required file: $path"
  printf 'asset ok: %s\n' "$path"
}

check_revision "$repo_root/third_party/sam3" 6dbb02bd38288df755dfa1378000a861e65b84f6
check_revision "$repo_root/third_party/sam-3d-objects" f91db411c50efee93d8db7aeb323885650f6f722
check_revision "$repo_root/third_party/FoundationPose" a1b694b83e633c2cb6115b9063d940a687759392
check_revision "$repo_root/third_party/WiLoR" fcb911312a38fa8badd30d9656a167485d61b8f9
check_revision "$repo_root/third_party/Depth-Anything-V2" a561b849ebae10a6f5ef49e26c83cbbcd36c71bf

[[ -d "$spider_root/.git" ]] || fail "missing SPIDER checkout: $spider_root"
git -C "$spider_root" merge-base --is-ancestor \
  71238456bf97a7eeb3d0471aa31974e2d404d4ae HEAD \
  || fail "SPIDER does not descend from the verified base revision"
spider_lock_hash=$(sha256sum "$spider_root/uv.lock" | awk '{print $1}')
[[ "$spider_lock_hash" == 23b98373fd8e0d0664871e4bcbaa9389009c06276f7aacb5d56afc946c9e5b85 ]] \
  || fail "SPIDER uv.lock hash mismatch: $spider_lock_hash"
printf 'SPIDER lock ok: %s\n' "$spider_lock_hash"

check_file "$repo_root/third_party/sam3/checkpoints/sam3.1_multiplex.pt"
check_file "$repo_root/third_party/WiLoR/pretrained_models/wilor_final.ckpt"
check_file "$repo_root/third_party/WiLoR/pretrained_models/model_config.yaml"
check_file "$repo_root/third_party/WiLoR/pretrained_models/detector.pt"
check_file "$repo_root/third_party/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"
check_file "$repo_root/third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml"
check_file "$repo_root/third_party/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"
check_file "$repo_root/third_party/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth"

for environment in v2s-core v2s-sam3 v2s-wilor v2s-depth v2s-sam3d v2s-foundationpose v2s-opt; do
  PYTHONNOUSERSITE=1 conda run -n "$environment" python -c \
    'import sys; print(sys.version.split()[0])' >/dev/null \
    || fail "Conda environment failed: $environment"
  printf 'environment ok: %s\n' "$environment"
done

uv_executable=${UV_EXECUTABLE:-$egoengine_root/.tools/uv/bin/uv}
if [[ ! -x "$uv_executable" ]]; then
  uv_executable=$(command -v uv || true)
fi
[[ -n "$uv_executable" && -x "$uv_executable" ]] || fail "uv executable not found"
(
  cd "$spider_root"
  "$uv_executable" lock --check >/dev/null
  "$uv_executable" run --frozen --no-sync python -c \
    'import mujoco, mujoco_warp, torch, warp' >/dev/null
) || fail "SPIDER frozen environment failed"
printf 'SPIDER environment ok: %s\n' "$uv_executable"

command -v ffmpeg >/dev/null || fail "ffmpeg not found"
command -v ffprobe >/dev/null || fail "ffprobe not found"
nvidia-smi -i "$gpu" --query-gpu=index,name,memory.total \
  --format=csv,noheader >/dev/null || fail "GPU $gpu is not queryable"
printf 'GPU query ok: physical index %s\n' "$gpu"

printf 'release preflight passed\n'
