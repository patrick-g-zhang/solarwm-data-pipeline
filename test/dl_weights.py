#!/usr/bin/env python3
"""Download the two pose-path model weights into weights/ (step [4]/[5] of setup_real_env.sh)."""
import os

from huggingface_hub import snapshot_download

WT = os.environ["SOLAR_WM_WEIGHTS"]
print(snapshot_download("yyfz233/Pi3", local_dir=f"{WT}/pi3"), flush=True)
print(snapshot_download("Ruicheng/moge-2-vitl-normal", local_dir=f"{WT}/moge2"), flush=True)
