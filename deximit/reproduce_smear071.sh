#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEXIMIT_ROOT="${DEXIMIT_ROOT:?Set DEXIMIT_ROOT to the patched DexImit-Open checkout}"
PYTHON="${PYTHON:-python3}"
DEXIMIT_PYTHON="${DEXIMIT_PYTHON:-$PYTHON}"
MUJOCO_PYTHON="${MUJOCO_PYTHON:-$PYTHON}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TMPDIR:-/tmp}/egoengine-deximit-repro}"

mkdir -p "$OUTPUT_ROOT"
RUN_DIR="$(mktemp -d "$OUTPUT_ROOT/smear071.XXXXXX")"
export PYTHONPATH="$ROOT/deximit${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

echo "Checking the frozen smear071 fixture"
"$MUJOCO_PYTHON" deximit/verify_smear071_backup.py

echo "Running the original DexImit SAPIEN screen"
"$DEXIMIT_PYTHON" deximit/diagnostics/run_deximit_sapien_screen.py \
  --deximit-root "$DEXIMIT_ROOT" \
  --candidate "$ROOT/deximit/fixture/smear071/input/candidate_pools/bodex_right_f4_d0_seed20260827_v2.npz" \
  --mesh "$ROOT/deximit/fixture/smear071/input/tool_078_meters.ply" \
  --human-reference "$ROOT/deximit/fixture/smear071/input/human_reference_v2.npz" \
  --deximit-prompt "$ROOT/deximit/fixture/smear071/input/mano_prompt_v1.npz" \
  --contact-v3 "$ROOT/deximit/fixture/smear071/input/contact_geom_v3.npz" \
  --manual-label "$ROOT/deximit/fixture/smear071/input/manual_pickup_subactions_v1.json" \
  --output-dir "$RUN_DIR/sapien" \
  --pool-depth 0 \
  --max-rollout 120 \
  --error-threshold-m 0.02 \
  --candidate-pose-policy legacy \
  --motion-mode source \
  --source-candidate-index 21 \
  --seed 20260829 \
  --render-mode raster \
  --export-trajectory

TRACE="$($MUJOCO_PYTHON - "$RUN_DIR/sapien/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
rows = [
    row for row in summary["original_sapien_passes"]
    if row["source_candidate_index"] == 21 and row["depth"] == 0
]
if len(rows) != 1 or rows[0]["original_sapien_pass"] is not True:
    raise SystemExit("candidate 21 did not pass the original SAPIEN screen")
print(rows[0]["exported_trajectory"])
PY
)"

echo "Replaying the exported SAPIEN trace in MuJoCo"
"$MUJOCO_PYTHON" deximit/diagnostics/replay_exact_deximit_mujoco.py \
  --scene "$ROOT/deximit/fixture/smear071/mujoco/scene.xml" \
  --sapien-trace "$TRACE" \
  --sapien-summary "$RUN_DIR/sapien/summary.json" \
  --home-probe "$ROOT/deximit/fixture/smear071/mujoco/bindings/joint_drive_probe.npz" \
  --output-dir "$RUN_DIR/mujoco_exact" \
  --hand-friction-mode strong-anchor

"$MUJOCO_PYTHON" - "$RUN_DIR/sapien/summary.json" "$RUN_DIR/mujoco_exact/report.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
report = json.load(open(sys.argv[2], encoding="utf-8"))
if summary.get("original_sapien_pass_count") != 1:
    raise SystemExit("SAPIEN pass count is not exactly one")
if report.get("strict_gate", {}).get("passed") is not True:
    raise SystemExit("MuJoCo strict gate failed")
print("smear071 candidate 21 reproduced")
print("SAPIEN summary:", sys.argv[1])
print("MuJoCo report:", sys.argv[2])
PY

echo "Reproduction output: $RUN_DIR"
