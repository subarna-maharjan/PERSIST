#!/bin/bash
#SBATCH --job-name=persist_l002_camgt
#SBATCH --output=grace_level002_%j.log
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
# >>> account with SUs. 132689579351 is overdrawn; 132689576523 (Zhang, Cheng,
#     FY2027) has a full 100000 SU balance. Check with `myproject` before submitting. <<<
#SBATCH --account=132689576523

# ---- level_002 full-GT-camera rollout on TAMU Grace ----
# Seed frame 0 with GT pixel + GT voxel; feed all 600 GT camera poses AND all 600
# GT actions.  Compute nodes have NO internet, so the dataset and weights MUST be
# pre-downloaded on the LOGIN node first (see the preflight block in the chat).
# This job runs fully OFFLINE from the $SCRATCH HF cache.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Scratch-backed caches (must already be populated on the login node).
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv_cache}"
# Force offline: never touch the network from a compute node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# A100 arch, so nvdiffrast/JIT kernels compile for the right target.
export TORCH_CUDA_ARCH_LIST=8.0
# nvdiffrast JIT-compiles a C++/CUDA plugin on first use; keep its build cache on
# scratch (home has a quota) and force a clean rebuild location.
export TORCH_EXTENSIONS_DIR="$SCRATCH/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

module purge
# GCC >= 9 is REQUIRED: nvdiffrast's plugin fails to compile with the old system
# c++ ("We need GCC 9 or later"). Load a modern GCC before CUDA. Adjust the exact
# version to whatever `module spider GCC` lists on Grace.
module load GCC/12.3.0 2>/dev/null \
  || module load GCC/11.3.0 2>/dev/null \
  || module load GCC/12.2.0 2>/dev/null \
  || module load GCCcore/12.3.0 2>/dev/null \
  || echo "WARN: no GCC module loaded -- nvdiffrast build will likely fail"
# Load the CUDA toolkit the venv's torch (cu126) was built against.
module load CUDA/12.6.0 2>/dev/null || echo "note: CUDA module not loaded (relying on venv/runtime CUDA)"
echo "=== compiler: $(c++ --version | head -1) ==="

source .venv/bin/activate

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import nvdiffrast, torch; print('nvdiffrast OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())"

OUTDIR=outputs/level_002/task2_camgt
if [ -f "${OUTDIR}/XL_level_002.npz" ]; then
  echo "SKIP: ${OUTDIR}/XL_level_002.npz already exists"; exit 0
fi

python -m scripts.run_inference \
  --pipeline-variant XL \
  --checkpoints-namespace PERSIST-team \
  --dataset-repo PERSIST-team/persist-eval-sample \
  --num-frames 600 \
  --eval-instances level_002 \
  --include-initial-pixel-frame \
  --include-initial-voxel-frame \
  --use-camera-gt \
  --save-voxel-grids \
  --output-dir "${OUTDIR}"

echo "DONE: level_002 -> ${OUTDIR}"
