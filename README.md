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

### HDF5 teleop 数采

- `run_hdf5_teleop_collect.py`
- `run_hdf5_teleop_episode_vis.py`

## 关键文档

- 部署调用链图：`docs/deploy_call_chain.md`
- HDF5 teleop 接入说明：`docs/hdf5_teleop.md`

## 顶层模块

```text
challenge_deploy/
├── clients/     # policy client abstraction and concrete OpenPI/X-VLA/OpenPI-sim/Motus/DreamZero clients
├── hardware/    # Piper, RealSense, runtime source, config, schemas, conversions
├── rollout/     # rollout execution, recording, metrics, train assets
├── teleop/      # HDF5 teleop collector and episode preview
├── run_openpi_client.py
├── run_xvla_client.py
├── run_openpi_sim_client.py
├── run_motus_client.py
├── run_dreamzero_client.py
├── run_hdf5_teleop_collect.py
└── run_hdf5_teleop_episode_vis.py
```

## Gripper 编码

五个推理入口都使用显式 gripper 编码参数：

- `--state-gripper {policy,meters,old}` 控制送入 policy 的 state gripper 表示。
- `--action-gripper {policy,meters,binary,old}` 控制 policy action gripper 到 Piper 硬件开口的解释方式。
- X-VLA 默认 `--state-gripper meters --action-gripper binary`；其他入口默认 `policy/policy`。
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
