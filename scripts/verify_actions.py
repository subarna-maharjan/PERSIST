"""Audit that each generated rollout used the ground-truth per-frame actions.

For every rollout ``.npz`` written by ``run_inference --save-voxel-grids`` this
compares its stored ``action`` array against the ground-truth action array in the
dataset's per-episode ``data.npz`` -- element-wise and in order -- and reports any
file whose actions differ. It scans recursively, so it covers both task folders
(e.g. ``outputs/task1_first_frame`` and ``outputs/task2_first_frame_plus_w0``) in
one run.

Examples:
    # scan everything under outputs/, GT taken from the HF cache ($HF_HOME)
    python -m scripts.verify_actions

    # only one task folder
    python -m scripts.verify_actions --outputs outputs/task1_first_frame

    # point at an explicit dataset snapshot, require exact (not just allclose) equality
    python -m scripts.verify_actions --dataset-root /path/to/persist-eval-sample --exact
"""
import argparse
import glob
import os
import re
import sys

import numpy as np


def find_dataset_root(explicit=None):
    """Locate the persist-eval-sample dataset directory (a built dataset with per-episode data.npz)."""
    if explicit:
        return explicit
    hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    pats = sorted(glob.glob(os.path.join(hf, "hub/datasets--PERSIST-team--persist-eval-sample/snapshots/*")))
    if not pats:
        sys.exit(
            "Could not find the persist-eval-sample cache under "
            f"{hf}/hub/. Pass --dataset-root <snapshot dir> explicitly."
        )
    return pats[-1]


def gt_action_path(root, inst):
    hits = glob.glob(os.path.join(root, "**", inst, "data.npz"), recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="outputs",
                    help="Directory containing rollout .npz files (searched recursively). Default: outputs")
    ap.add_argument("--dataset-root", default=None,
                    help="persist-eval-sample snapshot dir. Default: resolved from $HF_HOME cache.")
    ap.add_argument("--exact", action="store_true",
                    help="Require exact equality (np.array_equal) instead of np.allclose.")
    args = ap.parse_args()

    root = find_dataset_root(args.dataset_root)
    npzs = sorted(glob.glob(os.path.join(args.outputs, "**", "XL_*.npz"), recursive=True))
    if not npzs:
        sys.exit(f"No rollout .npz found under {args.outputs!r}.")

    print(f"dataset root : {root}")
    print(f"rollout files: {len(npzs)}  (searched under {args.outputs!r})")
    print(f"match mode   : {'exact (array_equal)' if args.exact else 'allclose'}\n")

    ok = differ = missing_gt = no_action = 0
    differing_files = []
    for p in npzs:
        m = re.search(r"(level_\d+)", os.path.basename(p))
        if not m:
            print(f"SKIP       {p}  (cannot parse instance id)")
            continue
        inst = m.group(1)

        d = np.load(p)
        if "action" not in d.files:
            print(f"NO-ACTION  {p}  (this .npz has no 'action' key)")
            no_action += 1
            continue
        used = np.asarray(d["action"], dtype=np.float32)

        gp = gt_action_path(root, inst)
        if gp is None:
            print(f"NO-GT      {p}  (no ground-truth data.npz found for {inst})")
            missing_gt += 1
            continue
        gt = np.asarray(np.load(gp, allow_pickle=True)["action"][: len(used)], dtype=np.float32)

        same_shape = used.shape == gt.shape
        equal = same_shape and (np.array_equal(used, gt) if args.exact else np.allclose(used, gt))
        if equal:
            ok += 1
        else:
            differ += 1
            differing_files.append(p)
            if not same_shape:
                print(f"DIFFER     {p}  [{inst}]  shape mismatch: used={used.shape} gt={gt.shape}")
            else:
                bad = int((~np.all(np.isclose(used, gt), axis=1)).sum())
                first = int(np.argmax(~np.all(np.isclose(used, gt), axis=1)))
                print(f"DIFFER     {p}  [{inst}]  {bad}/{len(used)} frames differ (first at frame {first})")

    print(f"\nSummary: OK={ok}  DIFFER={differ}  NO-GT={missing_gt}  NO-ACTION={no_action}  (total {len(npzs)})")
    if differing_files:
        print("\nFiles whose actions differ from ground truth:")
        for p in differing_files:
            print(f"  {p}")
    else:
        print("\nAll rollouts used the ground-truth per-frame actions. No differences found.")

    sys.exit(1 if (differ or missing_gt or no_action) else 0)


if __name__ == "__main__":
    main()
