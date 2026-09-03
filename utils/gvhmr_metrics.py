"""GVHMR's world-frame pose metrics, ported verbatim (``hmr4d/utils/eval/eval_utils.py``).

The alignment kernels (``align_pcl``, ``global_align_joints``,
``first_align_joints``), ``compute_jpe``, ``compute_rte`` and
``compute_jitter`` are GVHMR's code, formatting aside. :func:`global_metrics`
is its ``compute_global_metrics`` minus the vertex / foot-sliding part, with
the clip's real sampled fps passed to the jitter (GVHMR hard-codes 30 for its
30-fps benchmarks; the corpus clips run at 24-60 fps / stride auto) and a
guard on a motionless GT trajectory (RTE divides by its total displacement).

Names: GVHMR's ``wa2_mpjpe`` (first-two-frame alignment per 100-frame chunk)
is the paper's **W-MPJPE100**, ``waa_mpjpe`` (global similarity alignment per
chunk) the **WA-MPJPE100**; ``rte`` is in percent, ``jitter`` in 10 m/s^3.
Everything runs on the CPU, as in GVHMR.
"""
from __future__ import annotations

import numpy as np
import torch

CHUNK_LENGTH = 100


def compute_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean(dim=-1).numpy()


def align_pcl(Y, X, weight=None, fixed_scale=False):
    """align similarity transform to align X with Y using umeyama method
    X' = s * R * X + t is aligned with Y
    :param Y (*, N, 3) first trajectory
    :param X (*, N, 3) second trajectory
    :param weight (*, N, 1) optional weight of valid correspondences
    :returns s (*, 1), R (*, 3, 3), t (*, 3)
    """
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1) * N

    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

    # subtract mean
    my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]  # (*, N, 3)
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = torch.eye(3).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = (
            torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(
                dim=-1, keepdim=True
            )
            / var[..., 0]
        )  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    return s, R, t


def global_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_glob, R_glob, t_glob = align_pcl(
        gt_joints.reshape(-1, 3), pred_joints.reshape(-1, 3)
    )
    pred_glob = (
        s_glob * torch.einsum("ij,tnj->tni", R_glob, pred_joints) + t_glob[None, None]
    )
    return pred_glob


def first_align_joints(gt_joints, pred_joints):
    """
    align the first two frames
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    # (1, 1), (1, 3, 3), (1, 3)
    s_first, R_first, t_first = align_pcl(
        gt_joints[:2].reshape(1, -1, 3), pred_joints[:2].reshape(1, -1, 3)
    )
    pred_first = (
        s_first * torch.einsum("tij,tnj->tni", R_first, pred_joints) + t_first[:, None]
    )
    return pred_first


def compute_rte(target_trans, pred_trans):
    # Compute the global alignment
    _, rot, trans = align_pcl(target_trans[None, :], pred_trans[None, :], fixed_scale=True)
    pred_trans_hat = (
        torch.einsum("tij,tnj->tni", rot, pred_trans[None, :]) + trans[None, :]
    )[0]

    # Compute the entire displacement of ground truth trajectory
    disps, disp = [], 0
    for p1, p2 in zip(target_trans, target_trans[1:]):
        delta = (p2 - p1).norm(2, dim=-1)
        disp += delta
        disps.append(disp)

    # Compute absolute root-translation-error (RTE)
    rte = torch.norm(target_trans - pred_trans_hat, 2, dim=-1)

    # Normalize it to the displacement
    return (rte / disp).numpy()


def compute_jitter(joints, fps=30):
    """compute jitter of the motion
    Args:
        joints (N, J, 3).
        fps (float).
    Returns:
        jitter (N-3).
    """
    pred_jitter = torch.norm(
        (joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * (fps**3),
        dim=2,
    ).mean(dim=-1)

    return pred_jitter.cpu().numpy() / 10.0


@torch.no_grad()
def global_metrics(
    pred_j3d_glob: torch.Tensor, target_j3d_glob: torch.Tensor, fps: float,
) -> dict[str, np.ndarray]:
    """GVHMR's ``compute_global_metrics`` (joints only) on one VALID sequence.

    :param pred_j3d_glob: ``(T, J, 3)`` world metres, invalid frames already
        dropped (GVHMR compacts by its mask before chunking); joint 0 = root.
    :param target_j3d_glob: ``(T, J, 3)``.
    :param fps: the sequence's sampled frame rate (jitter only).
    :returns: per-frame arrays ``w_mpjpe100`` (``wa2_mpjpe``, mm),
        ``wa_mpjpe100`` (``waa_mpjpe``, mm), ``rte`` (%), ``jitter`` (10 m/s^3;
        ``T - 3`` entries). ``rte`` is empty for a motionless GT root.
    """
    pred_j3d_glob = pred_j3d_glob.float().cpu()
    target_j3d_glob = target_j3d_glob.float().cpu()
    seq_length = pred_j3d_glob.shape[0]

    wa2_mpjpe, waa_mpjpe = [], []
    for start in range(0, seq_length, CHUNK_LENGTH):
        end = min(seq_length, start + CHUNK_LENGTH)

        target_j3d = target_j3d_glob[start:end].clone().cpu()
        pred_j3d = pred_j3d_glob[start:end].clone().cpu()

        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)

        wa2_mpjpe.append(compute_jpe(target_j3d, w_j3d))
        waa_mpjpe.append(compute_jpe(target_j3d, wa_j3d))

    m2mm = 1000
    root = target_j3d_glob[:, 0]
    if float((root[1:] - root[:-1]).norm(dim=-1).sum()) > 0.0:
        rte = compute_rte(root.cpu(), pred_j3d_glob[:, 0].cpu()) * 1e2
    else:
        rte = np.zeros(0, np.float32)
    return {
        "w_mpjpe100": np.concatenate(wa2_mpjpe) * m2mm,
        "wa_mpjpe100": np.concatenate(waa_mpjpe) * m2mm,
        "rte": rte,
        "jitter": compute_jitter(pred_j3d_glob, fps=fps),
    }


__all__ = ["CHUNK_LENGTH", "align_pcl", "compute_jitter", "compute_jpe", "compute_rte",
           "first_align_joints", "global_align_joints", "global_metrics"]
