#!/usr/bin/env python3
"""Run ONLY steps 2-3 of the default pose path against an already-computed fused depth.

doc/pose_redeploy_rwt95.md's protocol computes the Pi3X+MoGe-2 fused depth once and then
runs the patched VIPE on it several times, so the spread between runs is VIPE's GPU BA
alone and not a second depth pass. ``scripts/pose_videos.py`` always recomputes the depth,
so this driver mirrors ``annotate_pose_vipe_cli`` steps 2 (``vipe infer``) and 3 (convert
to the corpus format) while pointing SOLAR_WM_FUSED_DEPTH_DIR at the shared dir, then
writes the same window layout ``pose_videos.py`` does.

    rerun_vipe.py <video.mp4> <shared-depth-dir> <out-window-dir> --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from solar_wm_data.ingest import _probe_video  # noqa: E402
from solar_wm_data.manifest import ClipRecord  # noqa: E402
from solar_wm_data.pose.vipe_cli import (  # noqa: E402
    _load_perframe_intrinsics,
    _load_vipe_pose,
    _real_env,
    _vipe_cfg,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("depth_dir", type=Path, help="existing fused-depth dir (fused.npy, scales.npy, ...)")
    ap.add_argument("out", type=Path, help="window dir to write poses.npy/intrinsics.npy/meta.json into")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args(argv)

    os.environ.setdefault("SOLAR_WM_ROOT", str(_REPO))
    os.environ.setdefault("SOLAR_WM_WEIGHTS", str(_REPO / "weights"))
    cfg = _vipe_cfg({"vipe": {"gpu": args.gpu}})
    commit = subprocess.run(
        ["git", "-C", f"{cfg['wm_root']}/third_party/vipe", "rev-parse", "--short=7", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    video = args.video.resolve()
    wd = args.out
    wd.mkdir(parents=True, exist_ok=True)
    work = wd / "_work"
    vipe_out = work / "vipe_out"
    pf_dump = work / "intr_pf.npy"
    work.mkdir(parents=True, exist_ok=True)

    rec = ClipRecord(clip_id=wd.name, source="web", video_path=str(video), mode="default")
    for key, value in _probe_video(video).items():
        if value:
            setattr(rec, key, value)

    t0 = time.time()
    vipe_env = dict(
        _real_env(cfg),
        SOLAR_WM_FUSED_DEPTH_DIR=str(args.depth_dir.resolve()),
        SOLAR_WM_PF_DUMP=str(pf_dump),
        PYTHONPATH=f"{cfg['wm_root']}",
    )
    subprocess.run(
        [cfg["vipe_bin"], "infer", str(video), "-o", str(vipe_out), "-p", "solarwm"],
        check=True, env=vipe_env,
    )

    poses = _load_vipe_pose(vipe_out)
    intr = _load_perframe_intrinsics(pf_dump, poses.shape[0])
    scales_npy = args.depth_dir / "scales.npy"
    scales = np.load(scales_npy).astype(np.float64).tolist() if scales_npy.exists() else []

    if not (len(poses) == len(intr) == rec.num_frames):
        raise RuntimeError(
            f"frame misalignment: video {rec.num_frames}, poses {len(poses)}, intrinsics {len(intr)}")
    if not (np.isfinite(poses).all() and np.isfinite(intr).all()):
        raise RuntimeError("non-finite poses or intrinsics")

    np.save(wd / "poses.npy", poses)
    np.save(wd / "intrinsics.npy", intr)
    rec.pose_path = str((wd / "poses.npy").resolve())
    rec.intrinsics_path = str((wd / "intrinsics.npy").resolve())
    rec.scale_factors = [float(x) for x in scales]
    rec.pose_mode = rec.mode
    rec.pose_units = "metric"
    rec.extra.update(pose_backend="vipe_cli", spec="none", vipe_commit=commit,
                     shared_depth_dir=str(args.depth_dir.resolve()))
    (wd / "meta.json").write_text(json.dumps(rec.to_dict(), ensure_ascii=False), encoding="utf-8")
    print(f"RERUN_VIPE_DONE {wd} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
