"""Audit that each --use-camera-gt rollout used the ground-truth per-frame camera pose.

When a rollout is produced with ``--use-camera-gt``, the pipeline copies the
ground-truth camera trajectory into ``rollout["camera"]`` one frame at a time
(``pipelines/pipeline.py`` line ~716):

    rollout["camera"] = torch.cat([rollout["camera"], context["camera"][:, i]...], dim=1)

So the ``camera`` array saved inside each rollout ``.npz`` must be *identical* to
the camera the dataset loader produces for that episode. This script rebuilds that
ground-truth camera with the SAME dataset class that ``run_inference`` uses
(``MinetestLatentCameraActionEval``), then compares it, element-wise and in frame
order, against every rollout ``.npz`` under the output directory. Any file whose
camera differs is reported.

Because the check reconstructs the camera through the exact loader path, a match
proves the generated video's viewpoint at every frame is the recorded (ground-truth)
camera pose -- rotation (6D) + translation (xyz) + FOV -- and NOT a camera the model
predicted for itself.

Examples:
    # scan everything under outputs/, dataset pulled/cached from HF
    python -m scripts.verify_camera --dataset-repo PERSIST-team/persist-eval-sample

    # only the camera-GT task folders for the 5 levels, exact equality
    python -m scripts.verify_camera --dataset-repo PERSIST-team/persist-eval-sample \
        --outputs outputs --exact
"""
import argparse
import glob
import os
import re
import sys

import numpy as np


def build_gt_camera(root, inst, num_frames):
    """Return the ground-truth camera (T, cam_dim) for one episode, exactly as the
    model receives it -- via the same dataset class run_inference uses."""
    import torch  # noqa: F401  (dataset class needs torch available)
    from data_loaders.minetest_latent_camera_action_dataset import MinetestLatentCameraActionEval

    ds = MinetestLatentCameraActionEval(
        root,
        clip_len=num_frames,
        instances=[inst],
        num_instances=None,
    )
    if len(ds) == 0:
        return None
    sample = ds[0]                       # dict with key 'camera' -> (clip_len, cam_dim)
    return np.asarray(sample["camera"], dtype=np.float32)


def find_dataset_root(args):
    if args.dataset_root:
        return args.dataset_root
    if args.dataset_repo:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=args.dataset_repo, repo_type="dataset")
    hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    pats = sorted(glob.glob(os.path.join(hf, "hub/datasets--PERSIST-team--persist-eval-sample/snapshots/*")))
    if not pats:
        sys.exit("persist-eval-sample not found; pass --dataset-repo or --dataset-root.")
    return pats[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="outputs",
                    help="Directory with rollout .npz files (searched recursively). Default: outputs")
    ap.add_argument("--dataset-repo", default=None, help="e.g. PERSIST-team/persist-eval-sample")
    ap.add_argument("--dataset-root", default=None, help="Explicit local snapshot dir (overrides --dataset-repo).")
    ap.add_argument("--exact", action="store_true",
                    help="Require exact equality (array_equal) instead of allclose.")
    ap.add_argument("--atol", type=float, default=1e-5, help="allclose absolute tolerance (default 1e-5).")
    args = ap.parse_args()

    root = find_dataset_root(args)
    npzs = sorted(glob.glob(os.path.join(args.outputs, "**", "XL_*.npz"), recursive=True))
    if not npzs:
        sys.exit(f"No rollout .npz found under {args.outputs!r}.")

    print(f"dataset root : {root}")
    print(f"rollout files: {len(npzs)}  (searched under {args.outputs!r})")
    print(f"match mode   : {'exact (array_equal)' if args.exact else f'allclose (atol={args.atol})'}\n")

    ok = differ = missing_gt = no_cam = 0
    differing = []
    gt_cache = {}
    for p in npzs:
        m = re.search(r"(level_\d+)", os.path.basename(p))
        if not m:
            print(f"SKIP       {p}  (cannot parse instance id)")
            continue
        inst = m.group(1)

        d = np.load(p)
        if "camera" not in d.files:
            print(f"NO-CAMERA  {p}  (this .npz has no 'camera' key)")
            no_cam += 1
            continue
        used = np.asarray(d["camera"], dtype=np.float32)

        if inst not in gt_cache:
            gt_cache[inst] = build_gt_camera(root, inst, used.shape[0])
        gt_full = gt_cache[inst]
        if gt_full is None:
            print(f"NO-GT      {p}  (episode {inst} not found in dataset)")
            missing_gt += 1
            continue
        gt = gt_full[: len(used)]

        same_shape = used.shape == gt.shape
        equal = same_shape and (np.array_equal(used, gt) if args.exact
                                else np.allclose(used, gt, atol=args.atol))
        if equal:
            ok += 1
            print(f"OK         {p}  [{inst}]  camera matches GT for all {len(used)} frames "
                  f"(shape {used.shape})")
        else:
            differ += 1
            differing.append(p)
            if not same_shape:
                print(f"DIFFER     {p}  [{inst}]  shape mismatch: used={used.shape} gt={gt.shape}")
            else:
                bad_mask = ~np.all(np.isclose(used, gt, atol=args.atol), axis=1)
                bad = int(bad_mask.sum())
                first = int(np.argmax(bad_mask))
                maxdiff = float(np.abs(used - gt).max())
                print(f"DIFFER     {p}  [{inst}]  {bad}/{len(used)} frames differ "
                      f"(first at frame {first}, max abs diff {maxdiff:.3g})")

    print(f"\nSummary: OK={ok}  DIFFER={differ}  NO-GT={missing_gt}  NO-CAMERA={no_cam}  (total {len(npzs)})")
    if differing:
        print("\nFiles whose camera differs from ground truth:")
        for p in differing:
            print(f"  {p}")
    else:
        print("\nAll rollouts used the ground-truth per-frame camera pose. No differences found.")

    sys.exit(1 if (differ or missing_gt or no_cam) else 0)


if __name__ == "__main__":
    main()
