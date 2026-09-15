"""Render a decoded voxel-class rollout to a colored RGB video.

The PERSIST pipeline returns, per rolled-out episode, the decoded voxel-class grid
(``rollout["voxel"]``, shape ``(T, X, Y, Z)`` with integer class ids) alongside the
voxel-local camera trajectory (``rollout["camera"]``, shape ``(T, 10)`` = 6D rotation
+ xyz + fov).  ``scripts/run_inference.py`` historically only wrote the pixel (RGB)
video; this module turns the persistent 3D voxel state into a video as well, so it can
be inspected directly.

Rendering deliberately reuses the *same* rasterization path the pixel denoiser uses
(``VoxelMeshRasterizer.rasterize`` on the fixed ``[-0.5, 0.5]`` unit-cube grid, fed the
worldcam-projected camera).  That guarantees the voxel video is spatially aligned with
the RGB video frame-for-frame.  Each voxel class is mapped to a fixed color; empty/air
classes are given zero alpha and composited out so only solid geometry is visible.
"""
from typing import Iterable, Optional, Sequence

import torch

from utils.camera_util import camera_params_to_matrices
from utils.voxel_rasterizer import VoxelMeshRasterizer
from pipelines.pipeline import BasePipeline


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vectorized HSV->RGB for 1D h/s/v tensors in [0, 1]. Returns (N, 3)."""
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i.long() % 6
    r = torch.stack([v, q, p, p, t, v], dim=-1)
    g = torch.stack([t, v, v, q, p, p], dim=-1)
    b = torch.stack([p, p, t, v, v, q], dim=-1)
    idx = i.unsqueeze(-1)
    return torch.stack(
        [
            r.gather(-1, idx).squeeze(-1),
            g.gather(-1, idx).squeeze(-1),
            b.gather(-1, idx).squeeze(-1),
        ],
        dim=-1,
    )


def build_class_palette(num_classes: int, seed: int = 0) -> torch.Tensor:
    """Deterministic, visually distinct color per class id. Returns (num_classes, 3) in [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    hues = torch.rand(num_classes, generator=g)
    sat = 0.45 + 0.45 * torch.rand(num_classes, generator=g)
    val = 0.55 + 0.40 * torch.rand(num_classes, generator=g)
    return _hsv_to_rgb(hues, sat, val).clamp(0.0, 1.0)


def _composite(pix: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    """Front-to-back pick the first depth layer with a solid (alpha>0.5) voxel.

    Args:
        pix: (L, H, W, 4) per-layer [rgb, alpha], depth-sorted near->far.
        bg:  (3,) background color used where no solid voxel is hit.
    Returns:
        (H, W, 3) in [0, 1].
    """
    L, H, W, _ = pix.shape
    rgb = pix[..., :3]
    alpha = pix[..., 3]
    out = bg.view(1, 1, 3).expand(H, W, 3).clone()
    filled = torch.zeros(H, W, dtype=torch.bool, device=pix.device)
    for layer in range(L):
        take = (alpha[layer] > 0.5) & (~filled)
        if take.any():
            out[take] = rgb[layer][take]
            filled |= take
        if bool(filled.all()):
            break
    return out.clamp(0.0, 1.0)


@torch.no_grad()
def render_voxel_video(
    voxel: torch.Tensor,
    camera: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
    air_classes: Optional[Sequence[int]] = None,
    background: Iterable[float] = (0.53, 0.81, 0.92),
    max_layers: Optional[int] = None,
    seed: int = 0,
) -> torch.Tensor:
    """Render a voxel-class rollout to an RGB video aligned with the pipeline camera.

    Args:
        voxel:  (T, X, Y, Z) integer voxel class ids (X==Y==Z; rasterizer needs a cubic grid).
        camera: (T, 10) voxel-local camera [6D rotation, xyz, fov] (as returned in rollout["camera"]).
        height, width: output resolution (match the pixel video so the two line up).
        device: CUDA device for the rasterizer.
        air_classes: class ids treated as empty (transparent). If None, defaults to {0} plus the
            single most frequent class over the whole rollout (a robust heuristic for air/empty).
        background: RGB (in [0, 1]) shown where no solid voxel is hit.
        max_layers: depth-peeling layers. Defaults to ``dim * 4`` (matches the pixel denoiser).
        seed: palette seed (keep fixed to compare rollouts).

    Returns:
        (T, H, W, 3) float in [-1, 1] (matches the pixel-video convention in run_inference).
    """
    voxel = voxel.to(torch.long)
    T, X, Y, Z = voxel.shape
    assert X == Y == Z, f"rasterizer needs a cubic voxel grid, got {(X, Y, Z)}"
    dim = X

    num_classes = int(voxel.max().item()) + 1
    palette = build_class_palette(num_classes, seed=seed).to(device)  # (C, 3)
    alpha = torch.ones(num_classes, device=device)

    air = {0} if air_classes is None else set(int(a) for a in air_classes)
    if air_classes is None:
        modal = torch.bincount(voxel.reshape(-1), minlength=num_classes).argmax().item()
        air.add(int(modal))
    for a in air:
        if 0 <= a < num_classes:
            alpha[a] = 0.0
    feats = torch.cat([palette, alpha[:, None]], dim=-1)  # (C, 4)

    rasterizer = VoxelMeshRasterizer(
        dim=dim, width=width, height=height, max_layers=max_layers or dim * 4, device=device
    )
    bg_feat = torch.zeros(4, device=device)  # background voxel: alpha 0

    # Match the pipeline: voxel-local camera -> worldcam -> view/projection matrices.
    cam = camera.unsqueeze(0).to(device).float()  # (1, T, 10)
    cam_world = BasePipeline.project_voxel_local_to_worldcam(cam)
    view, proj = camera_params_to_matrices(
        cam_world, image_width=width, image_height=height, device=device
    )
    view, proj = view[0], proj[0]  # (T, 4, 4)

    bg = torch.tensor(tuple(background), device=device, dtype=torch.float32)
    frames = []
    for t in range(T):
        vids = voxel[t].reshape(-1).to(device)          # (N,)
        vox_feat = feats[vids].unsqueeze(0)             # (1, N, 4)
        pix, _ = rasterizer.rasterize(vox_feat, view[t : t + 1], proj[t : t + 1], bg_feat)
        frames.append(_composite(pix[:, 0], bg))        # pix: (L, 1, H, W, 4)

    frames = torch.stack(frames, dim=0)                 # (T, H, W, 3) in [0, 1]
    return frames * 2.0 - 1.0                            # -> [-1, 1]
