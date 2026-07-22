# Challenge Deploy

这个目录现在包含两类入口：

- 真机推理入口
- HDF5 teleop 兼容数采 / 可视化入口

## 入口脚本

### 真机推理

- `run_openpi_client.py`
- `run_xvla_client.py`
- `run_openpi_sim_client.py`
- `run_motus_client.py`
- `run_dreamzero_client.py`
- `run_fastwam_client.py`

### HDF5 teleop 数采

- `run_hdf5_teleop_collect.py`
- `run_hdf5_teleop_episode_vis.py`

## 关键文档

- 部署调用链图：`docs/deploy_call_chain.md`
- HDF5 teleop 接入说明：`docs/hdf5_teleop.md`

## 顶层模块

```text
challenge_deploy/
├── clients/     # policy client abstraction and concrete OpenPI/X-VLA/OpenPI-sim/Motus/DreamZero/FastWAM clients
├── hardware/    # Piper, RealSense, runtime source, config, schemas, conversions
├── rollout/     # rollout execution, recording, metrics, train assets
├── teleop/      # HDF5 teleop collector and episode preview
├── run_openpi_client.py
├── run_xvla_client.py
├── run_openpi_sim_client.py
├── run_motus_client.py
├── run_dreamzero_client.py
├── run_fastwam_client.py
├── run_hdf5_teleop_collect.py
└── run_hdf5_teleop_episode_vis.py
```

## Gripper 编码

六个推理入口都使用显式 gripper 编码参数：

- `--state-gripper {policy,meters,old}` 控制送入 policy 的 state gripper 表示。
- `--action-gripper {policy,meters,binary,old}` 控制 policy action gripper 到 Piper 硬件开口的解释方式。
- X-VLA 默认 `--state-gripper meters --action-gripper binary`；FastWAM 默认 `policy/policy`；其他入口默认 `policy/policy`。
- 历史 gripper 数据兼容方式是同时传 `--state-gripper old --action-gripper old`；`--old_gripper` 不再提供。

## HDF5 teleop 接入原则

这次接入不是简单复刻 原始 ROS collector，而是保留其关键数据语义：

- `observation/action` 的一拍错位对齐
- RGB JPEG / depth PNG 的 HDF5 内联压缩格式
- 同时保存 `qpos/qvel/effort` 与 `eef_quaternion/eef_6d`
- `language_instruction` 的单 episode 级写法
- `episode_<idx>_running.txt` sentinel 提前终止机制
- `episode_vis.py` 的三视角拼接布局

设备层已经替换为当前项目原生实现：

- 机械臂：`piper_sdk`
- 相机：`pyrealsense2`

## 当前不包含

- ROS topic / message / roslaunch 适配层
- 历史 `dagger/`、旧 `inference/`、旧 `tools/`
- SAM3 相关代码

运行产物目录 `artifacts/` 保留，不在代码裁剪范围内。

## X-VLA 真机入口

先在 X-VLA 环境中启动 websocket policy server：

```bash
cd <xvla_repo>
<xvla_repo>/.venv/bin/python -m scripts.serve_policy \
  --model_path /path/to/your/xvla \
  --port 8000
```

再在 deploy 环境中运行独立 X-VLA client：

```bash
cd <challenge_deploy_repo>
python run_xvla_client.py \
  --train-config slai_piper_items_hand_over_place_ee20_xvla_pt_bs256_400000 \
  --host 127.0.0.1 \
  --port 8000 \
  --control-mode ee_pose
```

`--train-config` 必须是一个真实训练配置名；prompt 和训练分布图按训练配置的数据源匹配。

## FastWAM 真机入口

先在 FastWAM 环境中启动 HTTP policy server。Vid2WAM / adapter-distill 形态通常使用默认 server config：

```bash
cd <fastwam_repo>
python scripts/server.py \
  --checkpoint <upper_dir>/checkpoints/weights/step_xxx.pt \
  --port 8765
```

FastWAM baseline 权重按 PDF 中的配置启动：

```bash
cd <fastwam_repo>
python scripts/server.py \
  --config-name train \
  --task-override piper_realworld_unseen_baseline \
  --checkpoint <upper_dir>/checkpoints/weights/step_xxx.pt \
  --port 8766
```

再在 deploy 环境中运行 FastWAM client：

```bash
cd <challenge_deploy_repo>
python run_fastwam_client.py \
  --host 127.0.0.1 \
  --port 8765 \
  --prompt "grasp the towel to clean the plate"
```

FastWAM server 接收三路 RGB 图像、32D raw Piper proprio 和 `instruction`，返回 14D joint+gripper action chunk。client 默认按训练侧 0-1 gripper 开度解释 state/action，并在硬件层与米制开口互转。FastWAM 与其它 client 一样使用 `--prompt` 显式传入 instruction；`--train-config` 只作为 summary/record label，连接 baseline server 时可传 `--train-config piper_realworld_unseen_baseline --port 8766`。开启 `--record` 会保存本地相机视频、逐帧 action/state/time 和 frame1，并在 finalize 后像 Motus/DreamZero 一样尝试拉取 server 缓存的 predicted video；FastWAM server 默认不生成 predicted video，只有以 `--save-video-pred` 启动时才会进入视频生成分支。推理始终发送三路当前帧图像作为观测。

FastWAM 已内置下列 instruction 到 distribution asset 的映射；只有 `--prompt` 与这些文本匹配时，record/window 才会显示对应训练分布图：

```text
Pick the beaker, place it on the mixer, then flip the toggle switch with the other arm.
Pick the bottle then place it to the basket, carry the basket with the other arm.
Click the bell.
Pick the pipette and move it to the center-top of the beaker, use the other arm to depress the plunger.
Pick up two centrifuge tubes from the table and dock them horizontally.
Pick up the test tube and place it in the rack.
Pick up the pen, hand it over to the other arm and then place it in to the pen holder.
Open the drawer, pick the tomato with the other arm then place it in the drawer.
Grab the knob on the pan lid, lift it to open the pan, then pick the carrot with the other arm, place it in the pan, then move the lid back to the pan to close it.
Pick the cup and the bottle with the other arm, pour the water from bottle to cup.
Pick the fork and the spoon, place them next to the plate.
grasp the towel to clean the plate
Use the left arm to click the bell on the left and use the right arm to click the bell on the right.
pick up the tissue with the right arm
Pick the tomato then place it into the basket.
Pick up the test tube on the right from the rack with the right arm.
```

其中 5 个 unseen task 的首帧 distribution asset 来自 `<challenge_ws>/realsense_images`，已放到 `artifacts/train_distributions/`。
