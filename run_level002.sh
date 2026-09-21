#!/bin/bash
# level_002 only.  Seed frame 0 with GT pixel + GT voxel grid + GT camera pose;
# feed all 600 GT actions.  Camera for frames 1..599 is PREDICTED by the model
# (NO --use-camera-gt), i.e. only the first-frame camera pose is ground truth.
set -u
cd ~/PERSIST
source .venv/bin/activate

# GPU index: honor one already exported by the caller (e.g. wait_and_run.sh),
# otherwise default to 0.  Check free GPUs with nvidia-smi before running by hand.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

INST=level_002
OUTDIR=outputs/${INST}/first_frame_pvc      # p(ixel) v(oxel) c(amera) seeded at frame 0

if [ -f "${OUTDIR}/XL_${INST}.npz" ]; then
  echo "SKIP: ${OUTDIR}/XL_${INST}.npz already exists"
  exit 0
fi

python -m scripts.run_inference \
  --pipeline-variant XL \
  --checkpoints-namespace PERSIST-team \
  --dataset-repo PERSIST-team/persist-eval-sample \
  --num-frames 600 \
  --eval-instances "${INST}" \
  --include-initial-pixel-frame \
  --include-initial-voxel-frame \
  --save-voxel-grids \
  --output-dir "${OUTDIR}"

echo "DONE: ${INST} -> ${OUTDIR}"
