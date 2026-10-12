#!/usr/bin/env python3
"""Multi-run pose reproducibility eval: N runs, each vs reference, and run-to-run.

Implements doc/pose_redeploy_rwt95.md's protocol on either backend:

  local   scripts/pose_videos.py computes the fused depth once (run1, --keep-work);
          test/rerun_vipe.py then runs VIPE N-1 more times on that SAME depth dir, so
          the spread is VIPE's GPU BA alone.
  triton  solarwm-triton-pipeline/test/client_demo.py hits the production service N
          times, each with its own --outdir (the service recomputes depth each call,
          and the response files are named after the video stem, so outdirs must differ).

Then reports, per run, the Sim(3) RMSE against the reference track, and the full
run-to-run (i, j) matrix, plus a clustering check on those distances that flags the
bimodal BA-convergence pattern seen on the zx backend.

  eval_repro.py --clip test_data/pose_rwt95_in/Rwt95Vkw0_0003750_0005550.mp4 -n 3
  eval_repro.py --backend triton --clip ../solarwm-triton-pipeline/test/--Rwt95Vkw0_0003750_0005550/video.mp4 -n 6 --out /tmp/zxc
  eval_repro.py --compare-only --out /tmp/zxc        # just re-report an existing sweep
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from compare_poses import load, path_len, rot_angles, umeyama_sim3  # noqa: E402

DEFAULT_REF = _REPO / "test_data" / "ref_Rwt95" / "--Rwt95Vkw0_0003750_0005550"
DEFAULT_OUT = _REPO / "test" / "runs" / "repro"
TRITON_REPO = _REPO.parent / "solarwm-triton-pipeline"
PY = Path("/root/solarwm-env/bin/python")


# --------------------------------------------------------------------------- #
def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[eval] $ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {cmd[0]}")


def one_clip_dir(root: Path) -> Path:
    """The single <clip> window dir under a run root.

    With ``--spec none`` the whole clip is one window, so the run root IS the window
    dir; a spec that splits windows nests them one level down. Accept either.
    """
    if (root / "poses.npy").exists():
        return root
    cands = sorted(p for p in root.glob("*") if p.is_dir() and (p / "poses.npy").exists())
    if len(cands) != 1:
        raise SystemExit(f"{root}: expected 1 window dir with poses.npy, found {len(cands)}")
    return cands[0]


# --------------------------------------------------------------------------- #
def run_local(args, stem: str) -> list[Path]:
    """run1 = full pose_videos.py (keeps its _work depth), run2..N = rerun_vipe.py."""
    outs = [args.out / f"{stem}_run{i}" for i in range(1, args.runs + 1)]
    run(["scripts/pose_videos.py", args.clip, "--spec", "none", "--out", outs[0],
         "--keep-work", "--gpus", args.gpus], cwd=_REPO)

    win = one_clip_dir(outs[0])
    depth = next(p for p in (win / "_work").rglob("depth") if (p / "fused.npy").exists())
    video = win / "video.mp4"
    print(f"[eval] shared depth: {depth}")

    for i, out in enumerate(outs[1:], start=2):
        gpu = args.gpus.split(",")[(i - 2) % len(args.gpus.split(","))]
        t0 = time.time()
        run([PY, "test/rerun_vipe.py", video, depth, out, "--gpu", gpu], cwd=_REPO)
        print(f"[eval] run{i} done in {time.time() - t0:.0f}s", flush=True)
    return outs


def run_triton(args, stem: str) -> list[Path]:
    outs = [args.out / f"{stem}_run{i}" for i in range(1, args.runs + 1)]

    def one(pair):
        i, out = pair
        out.mkdir(parents=True, exist_ok=True)
        run([PY, "test/client_demo.py", "--url", args.url, "--video", args.clip,
             "--outdir", out, "--timeout", str(args.timeout)], cwd=TRITON_REPO)
        print(f"[eval] run{i} done", flush=True)

    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for _ in ex.map(one, enumerate(outs, start=1)):
            pass
    return outs


def colocate(outs: list[Path]) -> list[Path]:
    """Normalize run outputs to <out>/poses.npy for the local backend (already there)."""
    fixed = []
    for out in outs:
        if (out / "poses.npy").exists():
            fixed.append(out)
            continue
        cands = sorted(out.glob("*_poses.npy")) or sorted(out.glob("poses.npy"))
        if not cands:
            raise SystemExit(f"{out}: no poses file produced")
        (out / "poses.npy").write_bytes(cands[0].read_bytes())
        fixed.append(out)
    return fixed


# --------------------------------------------------------------------------- #
def cluster_check(D: np.ndarray, paths: list[float], rmses: list[float]) -> str:
    """Flag a two-solution split from the run-to-run distances.

    Connected components of the graph ``d(i,j) < 0.5 * max(d)`` — a run that converged
    to the same BA solution as another sits far closer to it than to the other group
    (zx: 2.3-3.7 m within vs 6.3-9.2 m across), while a continuous jitter has no such
    gap. Falls back to path length for the group label.
    """
    n = D.shape[0]
    if n < 4:
        return "too few runs to judge clustering"
    cutoff = 0.5 * D.max()
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i):
            if D[i, j] < cutoff:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    if len(groups) < 2:
        return (f"no evidence of clustering — every run is within {cutoff:.1f} m of every "
                f"other (continuous jitter, spread {D[D > 0].min():.1f}-{D.max():.1f} m)")
    parts = []
    for idx in sorted(groups.values(), key=lambda g: np.mean([paths[i] for i in g])):
        parts.append(
            f"{len(idx)} run(s) [{'/'.join(str(i + 1) for i in idx)}] path "
            f"{min(paths[i] for i in idx):.0f}-{max(paths[i] for i in idx):.0f} m, "
            f"vs-ref med {np.nanmedian([rmses[i] for i in idx]):.2f} m")
    cross = [D[i, j] for gi, a in enumerate(groups.values())
             for b in list(groups.values())[gi + 1:] for i in a for j in b]
    return ("**BIMODAL — " + str(len(groups)) + " separate BA solutions** (cutoff "
            f"{cutoff:.1f} m):\n  " + "\n  ".join(parts) +
            f"\n  cross-group distance med {np.median(cross):.2f} m "
            f"(min {min(cross):.2f})")


def report(args, outs: list[Path]) -> None:
    ref = load(args.ref)
    Cr = ref[:, :3, 3]
    runs = [load(o) for o in outs]
    names = [o.name for o in outs]

    print(f"\nreference: {args.ref}")
    print(f"  {len(ref)} frames, path {path_len(Cr):.1f} m\n")

    print("### each run vs reference")
    hdr = (f"{'run':<26} {'centre med':>10} {'rot med':>8} {'sim3 rmse':>10} "
           f"{'scale':>7} {'path':>7}")
    print(hdr)
    print("-" * len(hdr))
    rmses, paths = [], []
    for name, o in zip(names, runs):
        if len(o) != len(ref):
            print(f"{name:<26}  frame-count mismatch: {len(o)} vs {len(ref)}")
            rmses.append(float("nan"))
            paths.append(float("nan"))
            continue
        Co = o[:, :3, 3]
        s, _, _, rmse = umeyama_sim3(Co, Cr)
        rmses.append(rmse)
        paths.append(path_len(Co))
        print(f"{name:<26} {np.median(np.linalg.norm(Co - Cr, axis=1)):>10.2f} "
              f"{np.median(rot_angles(o[:, :3, :3], ref[:, :3, :3])):>8.2f} "
              f"{rmse:>10.2f} {s:>7.3f} {paths[-1]:>7.0f}")
    ok = [r for r in rmses if not np.isnan(r)]
    if ok:
        print(f"\nvs-ref sim3 rmse: median {np.median(ok):.2f} m "
              f"(min {min(ok):.2f} / max {max(ok):.2f})")

    print("\n### run-to-run (sim3 rmse, m)")
    w = max(len(n) for n in names) + 2
    D = np.zeros((len(runs), len(runs)))
    print(" " * w + "".join(f"{n.split('_run')[-1]:>8}" for n in names))
    pairs = []
    for i, (ni, oi) in enumerate(zip(names, runs)):
        row = f"{ni:<{w}}"
        for j, oj in enumerate(runs):
            if j >= i:
                row += " " * 8
                continue
            _, _, _, d = umeyama_sim3(oi[:, :3, 3], oj[:, :3, 3])
            D[i, j] = D[j, i] = d
            pairs.append(d)
            row += f"{d:>8.2f}"
        print(row)
    if pairs:
        print(f"\nrun-to-run sim3 rmse: median {np.median(pairs):.2f} m "
              f"(min {min(pairs):.2f} / max {max(pairs):.2f}, {len(pairs)} pairs)")

    print(f"\n### clustering by run-to-run distance\n{cluster_check(D, paths, rmses)}")
    intr = []
    for o in outs:
        for cand in (o / "intrinsics.npy", *o.glob("*_intrinsics.npy")):
            if cand.exists():
                a = np.load(cand)
                intr.append((o.name, float(a[:, 0].mean())))
                break
    if intr:
        print("\n### fx mean (px)\n" + "  ".join(f"{n}: {v:.1f}" for n, v in intr))

    # criterion from doc/pose_redeploy_rwt95.md: the vs-ref range must overlap the
    # pipeline's own run-to-run range (a max-vs-max test is meaningless at n=3)
    vlo, vhi = min(ok), max(ok)
    rlo, rhi = min(pairs or [0]), max(pairs or [0])
    okc = vlo <= rhi and rlo <= vhi
    verdict = (f"vs-ref sim3 rmse spans {vlo:.2f}-{vhi:.2f} m, run-to-run spans "
               f"{rlo:.2f}-{rhi:.2f} m — ")
    verdict += ("overlapping, so the vs-ref error is within the pipeline's own spread"
                if okc else
                "disjoint, so the vs-ref error exceeds the pipeline's run-to-run spread")
    print(f"\n{'ACCEPT' if okc else 'REJECT'}: {verdict}")
    print("(criterion from doc/pose_redeploy_rwt95.md: vs-ref error must fall inside the "
          "pipeline's own run-to-run spread)")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["local", "triton"], default="local")
    ap.add_argument("--clip", type=Path, help="video file (or clip dir) to re-pose")
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF, help="reference poses dir/npy")
    ap.add_argument("-n", "--runs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="sweep root")
    ap.add_argument("--gpus", default="0", help="local: comma-separated physical GPU ids")
    ap.add_argument("--jobs", type=int, default=3, help="triton: parallel client calls")
    ap.add_argument("--url", default="http://captioning.solarwm-triton.zxc200.dev.polaris:80")
    ap.add_argument("--timeout", type=float, default=7200.0, help="triton: per-call seconds")
    ap.add_argument("--compare-only", action="store_true",
                    help="skip inference, re-report existing runs under --out")
    ap.add_argument("--force", action="store_true", help="drop existing run dirs first")
    args = ap.parse_args(argv)

    stem = "eval"
    if not args.compare_only:
        if args.clip is None:
            raise SystemExit("--clip is required unless --compare-only")
        clip = args.clip if args.clip.is_absolute() else (_REPO / args.clip).resolve()
        if not clip.exists():
            raise SystemExit(f"clip not found: {clip}")
        stem = clip.parent.name if clip.is_dir() else clip.stem
        args.clip = clip
        if args.force:
            for i in range(1, args.runs + 1):
                shutil.rmtree(args.out / f"{stem}_run{i}", ignore_errors=True)
        args.out.mkdir(parents=True, exist_ok=True)
        outs = (run_local if args.backend == "local" else run_triton)(args, stem)
    else:
        outs = sorted(p for p in args.out.glob("*_run*") if p.is_dir())
        if not outs:
            raise SystemExit(f"no *_run* dirs under {args.out}")

    outs = colocate(outs)
    print(f"\n[eval] {len(outs)} runs under {args.out}")
    report(args, outs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
