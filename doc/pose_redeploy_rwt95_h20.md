# Pose 重部署核对：`--Rwt95Vkw0_0003750_0005550`（H200 / H20 / ZXC200）

日期：2026-10-10（H20）、2026-10-11（ZXC200）。H20：sm_90，与 H200 同一编译档，脚本 `scripts/pose_videos.py`（`--spec none`），VIPE 固定在 `95a8816`。ZXC200：不是本机跑的，是 `solarwm-triton-pipeline/test/client_demo.py` 打生产 Triton 服务（紫霄 ZXC200 硬件，torch-zx 后端，同一套 VIPE 95a8816 加 zx 专用补丁）。

按 `doc/pose_redeploy_rwt95.md` 在 H200 上的同一套协议重跑：融合深度只算一次，VIPE 在同一份深度上跑三次（ZXC200 是打服务 6 次，服务每次独立重算深度+VIPE），比较相对 zip 内 reference 的误差，以及跑次之间的误差。

## 结论

**vs reference 三台机器都差不多；但 ZXC200 自己跑次之间的误差明显更大，而且不是普通抖动，是两个收敛解的聚类。**

vs reference 的 Sim(3) RMSE 中位数：H200 4.36m、H20 3.65m、ZXC200 5.33m（6次：5.72/4.12/4.40/6.42/5.37/5.29）——三者同量级，没有谁明显更差。

run-to-run：H200 中位2.97m、H20 中位3.68m，都是正常抖动。ZXC200 不是——6次里2次收敛到"短路径"（690/758m，接近参考727m，fx≈744-746px），4次收敛到"长路径"（1049-1323m，明显偏长，fx≈729-732px），两群内部RMSE只有3-4m（和H200/H20一个量级），但两群之间RMSE 6.3-9.2m。根因是VIPE BA里`scatter_add`归约在zx后端非确定、加上intrinsics钳制过宽（细节见`solarwm-triton-pipeline/SLAM_OPTIMIZATION.md`§13.4），把这个非确定性放大成了两个分开的解，而不是H200/H20那种围着一个解小幅抖动。

关键判据和原 doc 一致：**相对 reference 的误差落在自己跑次波动的同一量级内**。三台机器都满足——ZXC200 的"两群"问题体现在跑次间不稳定，不体现在 vs reference 的精度上。

## 结果

相对 zip 内 reference 的 Sim(3) 中心 RMSE（m）：

| 机器 | 几次 | 中位数 |
|---|---|---:|
| H200（原 doc） | 4.42 / 3.75 / 4.36 | 4.36 |
| H20（本次） | 3.65 / 2.83 / 5.44 | 3.65 |
| ZXC200（打生产服务6次） | 5.72 / 4.12 / 4.40 / 6.42 / 5.37 / 5.29 | 5.33 |

跑次之间的 Sim(3) 中心 RMSE（m）：

| 机器 | 数据 | 中位数 |
|---|---|---:|
| H200 | 三对：2.50 / 2.97 / 3.90 | 2.97 |
| H20 | 三对：2.83 / 3.68 / 5.11 | 3.68 |
| ZXC200 | 两群内（短路径群、长路径群各3-4m）vs 两群间（6.3-9.2m） | 群内~3.5 / 群间~7.6 |

两个区间在两台机器上都重合（H200：3.75–4.42 vs 2.50–3.90；H20：2.83–5.44 vs 2.83–5.11），即相对参考的差距没有超出同版本自己的跑次波动。

H20 三次相对 reference 的完整列（reference 路径 727.1 m）：

| run | 中心 median (m) | 旋转 median (°) | Sim(3) RMSE (m) | scale | 路径 (m) |
|---|---:|---:|---:|---:|---:|
| runA | 2.81 | 5.09 | 3.65 | 1.113 | 691 |
| runB | 1.78 | 4.31 | 2.83 | 0.964 | 749 |
| runC | 4.66 | 4.90 | 5.44 | 0.808 | 577 |

对应的 H200 数字见原 doc。其余量也都落在原 doc 的区间里：中心 median H20 1.78–4.66 对 H200 2.49–4.75；未对齐旋转 median H20 4.31–5.09 对 H200 4.47–9.09；Sim(3) 对齐后旋转 median H20 14.9–28.3° 对 H200 13–31°；路径长 H20 577–749 m 对 H200 555–703 m；融合深度 scale 中位数 4.355 对 ≈4.36。

这条 clip 本身抖，三次跑的路径长就在 577–749 m 之间摆。离散主要来自 VIPE 的 GPU BA 非逐位可复现：逐帧内参 BA 收敛到了不同焦距（fx 中位数 746.4 / 724.5 / 759.7 px），直接拖动尺度和路径长。代码里没有 seed 设定。

三次输出都满足硬约束：`frame_count == len(poses) == len(intrinsics) == 960`，全部有限。

ZXC200 六次相对 reference 的完整列（通过 `solarwm-triton-pipeline/test/client_demo.py` 打生产 Triton 服务，服务每次独立重算深度+VIPE，不是共享深度）：

| run | Sim(3) RMSE (m) | fx 均值 (px) | scale | 路径 (m) |
|---|---:|---:|---:|---:|
| A | 5.72 | 731.7 | 0.663 | 1191 |
| B | 4.12 | 743.6 | 0.681 | 690 |
| C | 4.40 | 745.7 | 0.708 | 758 |
| D | 6.42 | 731.7 | 0.558 | 1323 |
| E | 5.37 | 729.4 | 0.808 | 1049 |
| F | 5.29 | 729.9 | 0.813 | 1206 |

六次的融合深度 `scale_factors` 逐帧位级完全相同——深度融合（Pi3X+MoGe-2）这步排除嫌疑，不是离散来源。离散出在 VIPE 的 BA：fx 明显分两群（729-732px 一群、744-746px 一群），对应路径长两群（1049-1323m 一群、690-758m 一群），短路径群更接近参考的727m和fx range。根因是 BA 正规方程组装用的 `scatter_add`（`vipe/slam/maths/matrix.py`/`vector.py`）在 zx 后端归约顺序非确定，叠加 `IntrinsicsRetractor.oplus`（`retractor.py`）对 fx 的钳制过宽（`[0.1cx,10cx]`），被 Gauss-Newton 迭代放大成两个分开的解，而不是像 H200/H20 那样围着一个解小幅抖动。详见 `solarwm-triton-pipeline/SLAM_OPTIMIZATION.md` §13.4。

## 部署与偏离

这台机器是干净检出，pose 引擎从零装起：torch 2.8.0+cu126；Pi3 `9fa3ddb`、MoGe `925b8ed`、utils3d `3fab839`、VIPE `95a8816`（CUDA 扩展按 sm_90 编译，`.venv-vipe` 通过 `_parent_env.pth` 继承母环境 torch，没装第二份）；`apply_vipe_patches.sh` 报 `applied=12 already=0 MISSED=0`；numpy 1.26.4；`_check_setup()` 返回 `95a8816`。`_probe_video` 读到 960 帧 / 16 fps / 1280×720，与原 doc 一致。

与原 doc 协议的偏离：

1. **opencv 必须是 4.11.0.86，不是仓库钉的 `<4.10`。** 这是仓库的真实缺口：`setup_real_env.sh` 钉了 `opencv-python-headless<4.10`，但 VIPE 的 `vipe/streams/raw_mp4_stream.py:65` 给 `cv2.VideoCapture` 传的是 `Path` 对象，4.10 以下不接受 PathLike，直接报 `Can't convert object to 'str' for 'filename'`；而 opencv 5.x 要求 numpy≥2，撞 numpy<2 的硬约束。可用版本只剩 4.11.0.86。原机器上应是 VIPE 的依赖解析顺带拉了个无上界的 `opencv-python` 才没撞到，建议改掉这条钉法。
2. 卡是 H20 不是 H200，同为 sm_90，编译目标相同。
3. 原 doc 三次跑在三张卡上；这次 GPU1 一直被别人的任务占着 87 GB，runB/runC 并行在 GPU0 上（VIPE 阶段峰值显存约 12 GB）。
4. 一次完整跑 1061 s，原 doc 是 165 s。卡和 CPU 都在和别人的任务抢，这个数不代表 H20 吞吐。
5. runA 走完整 `scripts/pose_videos.py` 并产出融合深度；runB/runC 走 `test/rerun_vipe.py`，复用 runA 的深度目录（同一个 `fused.npy`，不是重算），只跑 VIPE 的 `infer` + 转换，各 467 s / 498 s。`pose_videos.py` 每次都重算深度、没有跳过逻辑，所以「深度算一次、VIPE 跑三次」这个协议需要这个 driver 才能复现。

## 复现

`test/compare_poses.py` 先用原 doc 的数据校验过口径：对 zip 内 reference 跑 `test_data/pose_rwt95_out/`，输出 `3.00 / 9.09 / 3.48 / 18.1 / 1.109 / 703` 和 `960 frames, path 727.1 m, bbox diag 33.6 m`，与原 doc「第一次完整跑」那一行逐值相同。

```
# runA：完整路径
scripts/pose_videos.py --spec none --out test/runs/runA ...

# runB/runC：复用 runA 的融合深度
D=test/runs/runA/Rwt95Vkw0_0003750_0005550/_work/Rwt95Vkw0_0003750_0005550/depth
test/rerun_vipe.py test/runs/runA/Rwt95Vkw0_0003750_0005550/video.mp4 "$D" \
    test/runs/runB/Rwt95Vkw0_0003750_0005550 --gpu 0

# 两张表
bash test/compare_all.sh
```

ZXC200（在 solarwm-triton-pipeline 里，打生产服务，每次独立出一份 poses）：

```
python3 test/client_demo.py --video "test/--Rwt95Vkw0_0003750_0005550/video.mp4" --outdir test/ops/results/zxc_runX
# 比较：回到 solarwm-data-pipeline
test/compare_poses.py test_data/ref_Rwt95/--Rwt95Vkw0_0003750_0005550 <zxc_runX/resp_video_poses.npy> ...
```
