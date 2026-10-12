#!/usr/bin/env python3
"""Compare two pose tracks the way doc/pose_redeploy_rwt95.md does.

Columns, per the doc's protocol:
  centre median   median ‖C_a - C_b‖ in each track's OWN world frame (no alignment)
  rot median      median angle(R_a_iᵀ R_b_i), also unaligned
  sim3 rmse       Umeyama Sim(3) on camera centres, then RMSE in metres
  rot median*     same per-frame rotation but after applying the Sim(3) gauge rotation
  scale           the Sim(3) scale that takes `a` onto `b`
  path            Σ‖ΔC‖ of each track

Usage: compare_poses.py <a.npy|a_dir> <b.npy|b_dir> [more...]
Every argument after the first is compared against the first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Sim(3) aligning src->dst (N,3). Returns (s, R, t, rms)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    rms = float(np.sqrt(((s * (R @ src.T).T + t - dst) ** 2).sum(1).mean()))
    return s, R, t, rms


def rot_angles(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Per-frame angle between two rotation tracks, in degrees."""
    M = np.einsum("nij,nkj->nik", Ra, Rb)        # Ra_i @ Rb_iᵀ
    tr = np.trace(M, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def load(p: Path) -> np.ndarray:
    p = Path(p)
    if p.is_dir():
        p = p / "poses.npy"
    a = np.load(p)
    if a.shape[1:] != (4, 4):
        raise SystemExit(f"{p}: expected (N,4,4), got {a.shape}")
    return a.astype(np.float64)


def path_len(C: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())


def bbox_diag(C: np.ndarray) -> float:
    return float(np.linalg.norm(C.max(0) - C.min(0)))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    ref_p, others = Path(argv[0]), [Path(x) for x in argv[1:]]
    ref = load(ref_p)
    Cr = ref[:, :3, 3]
    print(f"reference: {ref_p}")
    print(f"  {len(ref)} frames, path {path_len(Cr):.1f} m, bbox diag {bbox_diag(Cr):.1f} m\n")

    hdr = (f"{'track':<22} {'centre med':>10} {'rot med':>8} {'sim3 rmse':>10} "
           f"{'rot med*':>9} {'rot p90*':>9} {'scale':>7} {'path':>7}")
    print(hdr)
    print("-" * len(hdr))
    for op in others:
        o = load(op)
        if len(o) != len(ref):
            print(f"{op.name:<22}  frame-count mismatch: {len(o)} vs {len(ref)}")
            continue
        Co = o[:, :3, 3]
        centre_med = float(np.median(np.linalg.norm(Co - Cr, axis=1)))
        rot_med = float(np.median(rot_angles(o[:, :3, :3], ref[:, :3, :3])))
        s, R, t, rmse = umeyama_sim3(Co, Cr)
        # rotation after taking out the alignment gauge: R_ref_i vs R_g @ R_o_i
        rot_al = rot_angles(np.einsum("ij,njk->nik", R, o[:, :3, :3]), ref[:, :3, :3])
        # label a window dir by its run (the dir above the clip) so the rows differ
        name = op.parent.name if op.is_dir() else op.stem
        if name in (".", ""):
            name = op.name
        print(f"{name:<22} {centre_med:>10.2f} {rot_med:>8.2f} {rmse:>10.2f} "
              f"{np.median(rot_al):>9.1f} {np.percentile(rot_al, 90):>9.1f} "
              f"{s:>7.3f} {path_len(Co):>7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
