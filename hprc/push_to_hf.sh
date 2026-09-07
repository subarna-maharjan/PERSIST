#!/usr/bin/env bash
# =====================================================================
# Push the generated datasets (train / eval1 / eval2 + seed manifest) to a
# Hugging Face *dataset* repo, from Grace. Run AFTER grace_generate.slurm
# finishes.
#
# Requirements:
#   - A HF *write* token:  https://huggingface.co/settings/tokens  ("Write")
#   - Outbound internet:   module load WebProxy   (done below)
#
# Usage (interactive test on a login node, small data) OR via push_hf.slurm:
#   export HF_TOKEN=hf_xxxxxxxxxxancxxxxxxxxxxxxxxx
#   export HF_REPO=<your-hf-username>/<dataset-name>     # e.g. Subarna2026/craftium-dynamic-agents
#   bash hprc/push_to_hf.sh
#
# Optional:
#   HF_PRIVATE=1   -> create the repo private (default public)
# =====================================================================
set -e
cd "${SCRATCH}/PERSIST"

: "${HF_TOKEN:?set HF_TOKEN=<your HF write token>}"
: "${HF_REPO:?set HF_REPO=<username>/<dataset-name>}"
ENV="${ENVID:-OpenWorldCreative-v0}"
SIF="${SIF:-${SCRATCH}/craftium.sif}"
PRIVATE_PY="True"; [ "${HF_PRIVATE:-0}" = "1" ] || PRIVATE_PY="False"

module load WebProxy 2>/dev/null || true    # outbound internet

# Run a command inside the container venv, with the token + Xet high-performance
# transfer on. (Newer huggingface_hub uses the `hf` CLI + Xet, not the old
# `huggingface-cli` / hf_transfer.)
run() {
  singularity exec --bind /scratch "$SIF" bash -lc \
    "cd '$PWD' && source .venv/bin/activate && \
     export HF_TOKEN='$HF_TOKEN' HF_XET_HIGH_PERFORMANCE=1 \
            HF_HOME='$SCRATCH/.hf-cache' && $1"
}

# 1) Make sure the modern HF client is in the venv (provides the `hf` CLI + Xet).
run "python -c 'import huggingface_hub' 2>/dev/null || uv pip install -U 'huggingface_hub[cli,hf_xet]'"

# 2) Create the dataset repo once (idempotent; sets public/private).
echo "==> Ensuring dataset repo exists: $HF_REPO (private=$PRIVATE_PY)"
run "python -c \"import os; from huggingface_hub import HfApi; \
HfApi(token=os.environ['HF_TOKEN']).create_repo('$HF_REPO', repo_type='dataset', private=$PRIVATE_PY, exist_ok=True)\""

# 3) Upload each split under its own folder in the repo. `hf upload` handles large
#    files via LFS/Xet and resumes cleanly if re-run.
for NAME in train eval1 eval2; do
  if [ -d "datasets/$NAME" ]; then
    echo "==> Uploading datasets/$NAME  ->  $HF_REPO:/$NAME"
    run "hf upload '$HF_REPO' 'datasets/$NAME' '$NAME' \
           --repo-type dataset --commit-message 'add $NAME split'"
  else
    echo "!! datasets/$NAME not found, skipping"
  fi
done

# 4) Upload the seed manifest at the repo root, if present.
if [ -f "datasets/seeds_used.csv" ]; then
  echo "==> Uploading seeds_used.csv"
  run "hf upload '$HF_REPO' 'datasets/seeds_used.csv' 'seeds_used.csv' \
         --repo-type dataset --commit-message 'add seed manifest'"
fi

echo "==> DONE. View at: https://huggingface.co/datasets/$HF_REPO"
