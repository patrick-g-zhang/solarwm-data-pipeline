#!/bin/bash
# Compare every finished run under test/runs against the zip's reference track.
cd "$(dirname "$0")/.." || exit 1
REF=test_data/ref_Rwt95/--Rwt95Vkw0_0003750_0005550
PY=/root/solarwm-env/bin/python

runs=()
for d in test/runs/*/Rwt95Vkw0_0003750_0005550; do
    [ -f "$d/poses.npy" ] && runs+=("$d")
done
runs+=(test_data/pose_rwt95_out/Rwt95Vkw0_0003750_0005550)

echo "### vs reference"
"$PY" test/compare_poses.py "$REF" "${runs[@]}"

if [ "${#runs[@]}" -gt 2 ]; then
    echo
    echo "### run-to-run (first run as the baseline)"
    "$PY" test/compare_poses.py "${runs[@]}"
fi
