#!/bin/bash
#SBATCH --job-name=persist_camgt
#SBATCH --output=grace_rollout_%j.log
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Request an A100 (40GB). A generic gpu:1 gave a ~24GB card that OOM'd on the
# XL voxel-VAE step. Confirm the type name with `sinfo -o "%P %G %N" | grep -i gpu`.
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --account=132689576523

# ---- Parameterized full-GT-camera rollout on TAMU Grace ----
# Pick the episode with INST, e.g.:
#   sbatch --export=ALL,INST=level_004 grace_rollout.sbatch
#   sbatch --export=ALL,INST=level_005 grace_rollout.sbatch
# Seed frame 0 with GT pixel + GT voxel; feed all 600 GT camera poses AND all 600
# GT actions.  Compute nodes have NO internet, so the dataset and weights MUST be
# pre-downloaded on the LOGIN node first.  Runs fully OFFLINE from $SCRATCH HF cache.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Which episode to roll out (default level_002 if not passed).
INST="${INST:-level_002}"

# Scratch-backed caches (must already be populated on the login node).
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv_cache}"
# Force offline: never touch the network from a compute node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Reduce CUDA memory fragmentation (PyTorch's own OOM hint).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# A100 arch for nvdiffrast/JIT kernels.
export TORCH_CUDA_ARCH_LIST=8.0
# nvdiffrast JIT build cache on scratch (home has a quota).
export TORCH_EXTENSIONS_DIR="$SCRATCH/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

module purge
# GCC >= 9 REQUIRED for the nvdiffrast plugin build. Adjust to `module spider GCC`.
module load GCC/12.3.0 2>/dev/null \
  || module load GCC/11.3.0 2>/dev/null \
  || module load GCC/12.2.0 2>/dev/null \
  || module load GCCcore/12.3.0 2>/dev/null \
  || echo "WARN: no GCC module loaded -- nvdiffrast build will likely fail"
module load CUDA/12.6.0 2>/dev/null || echo "note: CUDA module not loaded (relying on venv/runtime CUDA)"
echo "=== compiler: $(c++ --version | head -1) ==="

# Drop any auto-activated conda env(s) so they don't shadow module GCC / venv python.
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

source .venv/bin/activate

echo "=== instance: ${INST} ==="
echo "=== node: $(hostname) ==="
echo "=== python: $(which python) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import nvdiffrast, torch; print('nvdiffrast OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())"

OUTDIR="outputs/${INST}/task2_camgt"
if [ -f "${OUTDIR}/XL_${INST}.npz" ]; then
  echo "SKIP: ${OUTDIR}/XL_${INST}.npz already exists"; exit 0
fi

python -m scripts.run_inference \
  --pipeline-variant XL \
  --checkpoints-namespace PERSIST-team \
  --dataset-repo PERSIST-team/persist-eval-sample \
  --num-frames 600 \
  --eval-instances "${INST}" \
  --include-initial-pixel-frame \
  --include-initial-voxel-frame \
  --use-camera-gt \
  --save-voxel-grids \
  --output-dir "${OUTDIR}"

echo "DONE: ${INST} -> ${OUTDIR}"
