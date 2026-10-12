---
name: pose-repro-eval
description: Evaluate pose-pipeline reproducibility by running the same clip N times and comparing every run against the reference track and against each other (Sim(3) RMSE). Use when the user asks to check run-to-run stability / determinism / "跑多次误差" / "compare with self and reference" for the SolarWM pose path, on the local GPU (pose_videos.py + rerun_vipe.py) or against the production Triton service (client_demo.py).
---

# Pose reproducibility eval

Answers one question, from `doc/pose_redeploy_rwt95.md`: **is the pose pipeline's error
against the reference track consistent with how much the pipeline varies run to run, or
is that error bigger than its own noise?** Re-run the same clip N times, compare each run
to the reference and to every other run.

Everything is driven by `test/eval_repro.py`; do not hand-roll the orchestration.

## Invoke it

```bash
cd <solarwm-data-pipeline>
# local GPU: depth computed once, VIPE run N times on that same depth
/root/solarwm-env/bin/python test/eval_repro.py --backend local \
    --clip test_data/pose_rwt95_in/Rwt95Vkw0_0003750_0005550.mp4 -n 3 --gpus 0

# production Triton service (ZXC200); -n parallel client calls
/root/solarwm-env/bin/python test/eval_repro.py --backend triton \
    --clip ../solarwm-triton-pipeline/test/--Rwt95Vkw0_0003750_0005550/video.mp4 \
    -n 6 --out <solarwm-triton-pipeline>/test/ops/results/repro --jobs 3

# re-report an existing sweep without re-running inference
/root/solarwm-env/bin/python test/eval_repro.py --compare-only --out <sweep-root>
```

Defaults: `--ref test_data/ref_Rwt95/--Rwt95Vkw0_0003750_0005550`, `-n 3`,
`--out test/runs/repro`. Runs land in `<out>/<stem>_run1..N/` as `poses.npy` +
`intrinsics.npy`, so a sweep is re-reportable later.

Key flags: `--force` drops existing run dirs first; `--timeout` per Triton call (default
7200s); `--jobs` Triton parallelism; `--gpus` comma-separated physical GPU ids for local.

## Read the output

Four blocks: **each run vs reference** (centre median / rot median / Sim(3) RMSE / scale /
path), **run-to-run matrix**, **clustering**, **fx mean**, then the verdict.

- `sim3 rmse` = Umeyama Sim(3) on camera centres, in metres. This is the number that
  matters; `centre med` is unaligned and only meaningful for same-gauge tracks.
- **Verdict = interval overlap**: the vs-ref RMSE range overlapping the run-to-run RMSE
  range means the vs-ref error sits inside the pipeline's own spread (ACCEPT). Do not
  compare max-to-max — with n=3 that is noise. Report the two ranges.
- **Clustering**: connected components of the run-to-run matrix below half the max
  distance. One component = ordinary jitter. Two = the runs converged to genuinely
  different BA solutions; report each group's size, path-length range and vs-ref median.
  The ZXC200 zx backend splits 2/6 short-path (~690-758 m, fx ≈744 px, vs-ref med 4.26 m)
  vs 4/6 long-path (~1049-1323 m, fx ≈730 px, vs-ref med 5.54 m), cross-group 6.3-9.2 m.
- **fx mean** is the tell for that split: BA converges to different focal lengths, which
  is what drags scale and path length. If runs cluster, print the fx column too.

## Practical notes

- **Always use `/root/solarwm-env/bin/python`.** Bare `python3` lacks numpy/requests.
- **Triton runs take ~25-35 min each.** Launch with `run_in_background`, then wait with
  `ScheduleWakeup` (1200-1800s) rather than blocking; poll the log on wake. `--jobs 3`
  runs three calls concurrently against the service.
- **Triton response files are named after the video stem** (always `resp_video_*.npy`), so
  each run needs its own `--outdir` — `eval_repro.py` already does this. Never point two
  runs at the same outdir.
- **Local backend needs `--keep-work`** (run 1) so its `_work/<clip>/depth/` survives for
  `rerun_vipe.py`. `pose_videos.py` always recomputes depth, so sharing that dir is the
  only way to isolate VIPE's BA from the depth pass.
- A local sweep is GPU-heavy: run 1 is a full pipeline pass (~18 min here), each
  `rerun_vipe.py` ~8 min. Check `nvidia-smi` for free memory before picking `--gpus`.
- Compare the sweep against the recorded baselines in `doc/pose_redeploy_rwt95_h20.md`
  (H200 4.36 m, H20 3.65 m, ZXC200 5.33 m vs-ref medians) when judging a new machine.

## Reporting

State the answer, not the process: the vs-ref range, the run-to-run range, whether they
overlap, and — if the matrix splits — which runs form which cluster and what fc/path
differs. One short table plus two sentences. Do not write a new doc unless asked; if the
user wants it persisted, extend `doc/pose_redeploy_rwt95_h20.md`.
