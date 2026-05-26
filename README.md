# Challenge Deploy

这个目录现在包含两类入口：

- 真机推理入口
- HDF5 teleop 兼容数采 / 可视化入口

## 入口脚本

### 真机推理

- `run_openpi_clients.py`
- `run_openpi_sim_client.py`
- `run_motus_client.py`

### HDF5 teleop 数采

- `run_hdf5_teleop_collect.py`
- `run_hdf5_teleop_episode_vis.py`

## 关键文档

- 部署调用链图：`docs/deploy_call_chain.md`
- HDF5 teleop 接入说明：`docs/hdf5_teleop.md`

## 顶层模块

```text
challenge_deploy/
├── clients/     # policy client abstraction and concrete OpenPI/Motus clients
├── hardware/    # Piper, RealSense, runtime source, config, schemas, conversions
├── rollout/     # rollout execution, recording, metrics, train assets
├── teleop/      # HDF5 teleop collector and episode preview
├── run_openpi_clients.py
├── run_openpi_sim_client.py
├── run_motus_client.py
├── run_hdf5_teleop_collect.py
└── run_hdf5_teleop_episode_vis.py
```

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
