"""Confirm how each rollout was initialized.

Two claims are verified per episode, straight from the produced outputs:

1. Voxel init (w0):
   - Task 2 (`--include-initial-voxel-frame`) must START from the ground-truth
     frame-0 voxel grid, so rollout `voxel[0]` == dataset `voxel_classes[0]`.
   - Task 1 (no voxel frame) must INVENT w0 from the first image, so its
     `voxel[0]` should differ from the ground-truth grid.

2. Pixel init (first frame):
   - Both tasks are conditioned on the ground-truth first RGB frame, so the
     generated video's frame 0 should be far closer to the GT video's frame 0
     than to any other GT frame. We report the PSNR of gen-frame-0 against every
     GT frame and check the best match is at (or right next to) frame 0.

Usage:
    python -m scripts.verify_init                      # auto-pick an episode present in both tasks
    python -m scripts.verify_init --instance level_001
"""
import argparse
import glob
import os
import re
import sys

import numpy as np


def dataset_root(explicit=None):
    if explicit:
        return explicit
    hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    pats = sorted(glob.glob(os.path.join(hf, "hub/datasets--PERSIST-team--persist-eval-sample/snapshots/*")))
    if not pats:
        sys.exit("persist-eval-sample cache not found; pass --dataset-root")
    return pats[-1]


def read_first_and_all(path):
    """Return (frame0 uint8 HxWx3, all_frames uint8 TxHxWx3) for a video file."""
    from torchvision.io import read_video
    v, _, _ = read_video(path, pts_unit="sec")   # (T, H, W, 3) uint8
    v = v.numpy()
    return v[0], v


def psnr(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    if a.shape != b.shape:
        # resize b to a via simple nearest crop/pad-free interpolation
        from PIL import Image
        b = np.asarray(Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0])))
        b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default=None, help="e.g. level_001 (default: first episode present in both tasks)")
    ap.add_argument("--task1", default="outputs/task1_first_frame")
    ap.add_argument("--task2", default="outputs/task2_first_frame_plus_w0")
    ap.add_argument("--dataset-root", default=None)
    args = ap.parse_args()

    root = dataset_root(args.dataset_root)

    inst = args.instance
    if inst is None:
        for p in sorted(glob.glob(os.path.join(args.task2, "XL_level_*.npz"))):
            cand = re.search(r"(level_\d+)", os.path.basename(p)).group(1)
            if os.path.exists(os.path.join(args.task1, f"XL_{cand}.npz")):
                inst = cand
                break
    if inst is None:
        sys.exit("No episode found in both task folders yet.")
    print(f"episode: {inst}\n")

    # ---- ground truth ----
    vox_gt = np.load(glob.glob(root + f"/**/voxel_classes/{inst}.npz", recursive=True)[0])["node_classes"]
    gt_w0 = np.asarray(vox_gt[0])
    gt_rgb_path = glob.glob(root + f"/**/{inst}/rgb.mp4", recursive=True)[0]
    gt_f0, gt_all = read_first_and_all(gt_rgb_path)

    print("=== 1. Voxel init (w0) ===")
    for label, d in [("Task1 (first frame only) ", args.task1),
                     ("Task2 (first frame + w0) ", args.task2)]:
        npz = os.path.join(d, f"XL_{inst}.npz")
        if not os.path.exists(npz):
            print(f"{label}: (missing {npz})"); continue
        vox0 = np.asarray(np.load(npz)["voxel"][0])
        match = vox0.shape == gt_w0.shape and np.array_equal(vox0.astype(np.int64), gt_w0.astype(np.int64))
        frac = float((vox0.astype(np.int64) == gt_w0.astype(np.int64)).mean()) if vox0.shape == gt_w0.shape else 0.0
        print(f"{label}: voxel[0]==GT_w0? {match}   (cell agreement {frac:.3f})")
    print("  expect  Task2 -> True (seeded with GT w0),  Task1 -> False (model-invented)\n")

    print("=== 2. Pixel init (first frame) ===")
    for label, d in [("Task1", args.task1), ("Task2", args.task2)]:
        mp4 = os.path.join(d, f"XL_{inst}_rgb.mp4")
        if not os.path.exists(mp4):
            print(f"{label}: (missing {mp4})"); continue
        gen_f0, _ = read_first_and_all(mp4)
        # PSNR of generated frame 0 vs every GT frame
        psnrs = np.array([psnr(gen_f0, gt_all[t]) for t in range(min(len(gt_all), 600))])
        best = int(np.argmax(psnrs))
        print(f"{label}: PSNR(gen[0], GT[0]) = {psnrs[0]:.2f} dB | "
              f"best-matching GT frame = {best} ({psnrs[best]:.2f} dB) | "
              f"median GT-frame PSNR = {np.median(psnrs):.2f} dB")
    print("  expect: GT frame 0 is the best (or near-best) match, and well above the median\n")


if __name__ == "__main__":
    main()
