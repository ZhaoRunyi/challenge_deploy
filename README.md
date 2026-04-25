# Challenge Deploy, Without ROS

这个目录把 `/home/edemlab/challenge_ws/baselines/kai0` 里和 Agilex Piper 真机部署相关的代码整理成一份可直接运行的 ROS-free 版本。

目标很明确：

1. 尽量保留 kai0 原版的脚本命名、数据组织、控制语义、单位换算和运行习惯。
2. 删除 ROS topic、roslaunch、rospy、cv_bridge、sensor_msgs、piper_msgs 这一整层适配。
3. 用 `piper_sdk` pip 包直接访问双 Piper 机械臂。
4. 用 `pyrealsense2` 直接访问三台 RealSense。
5. 保留 raw 数据采集、OpenPi inference、DAgger collect 三条主要真机链路。

重要约定：

- `deploy/challenge_deploy/` 是共享包名。
- inference 和 DAgger 入口按部署入口处理，默认就是可以下发真实控制命令。
- 只想验证通信、拍照或读状态时，使用明确的只读工具入口，例如 `tools/probe_dual_piper.py`、`tools/capture_snapshot.py`、`inference/... --probe-only`、`dagger/collect_data.py`。
- 本次验证过程中没有执行任何会让机械臂运动的命令；这只是测试边界，不是代码默认行为。

---

## 1. 参考的 kai0 原始代码

这次只整理 kai0 里的 Agilex Piper 真机链路，不整理训练代码，也不整理 ARX 相关代码。

推理 / 部署链路：

- `baselines/kai0/train_deploy_alignment/inference/agilex/README.md`
- `baselines/kai0/train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_temporal_smoothing.py`
- `baselines/kai0/train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_sync.py`

DAgger / 采集链路：

- `baselines/kai0/train_deploy_alignment/dagger/agilex/README.md`
- `baselines/kai0/train_deploy_alignment/dagger/agilex/agilex_openpi_dagger_collect.py`
- `baselines/kai0/train_deploy_alignment/dagger/agilex/collect_data.py`

原本 ROS 外壳里真正驱动 Piper 的部分：

- `baselines/kai0/train_deploy_alignment/dagger/agilex/src/piper/scripts/piper_start_ms_node_new.py`
- `baselines/kai0/train_deploy_alignment/dagger/agilex/src/piper/scripts/piper_start_master_node.py`
- `baselines/kai0/train_deploy_alignment/dagger/agilex/src/piper/scripts/piper_start_slave_node.py`

原本 CAN 激活脚本：

- `baselines/kai0/train_deploy_alignment/dagger/agilex/can_activate.sh`
- `baselines/kai0/train_deploy_alignment/dagger/agilex/activate_can_arms.sh`

这些文件拆开后，本质上分成四层：

1. 设备层：CAN 口、RealSense、Piper 双臂。
2. ROS 适配层：ROS node、topic、message、publisher/subscriber。
3. 运行时层：三相机和双臂状态同步、observation window、temporal smoothing、DAgger 状态机。
4. 入口脚本层：inference、sync inference、DAgger collect、raw collect_data。

本目录保留第 1 / 3 / 4 层，删除第 2 层。

---

## 2. 目录结构

```text
deploy/
├── README.md
├── run_openpi_clients.py
├── configs/
│   └── dual_piper_example.yaml
├── dagger/
│   ├── agilex_openpi_dagger_collect.py
│   └── collect_data.py
├── inference/
│   ├── agilex_inference_openpi_sync.py
│   └── agilex_inference_openpi_temporal_smoothing.py
├── challenge_deploy/
│   ├── __init__.py
│   ├── buffer.py
│   ├── can_tools.py
│   ├── config.py
│   ├── constants.py
│   ├── conversions.py
│   ├── dataset.py
│   ├── observation.py
│   ├── piper.py
│   ├── policy.py
│   ├── realsense.py
│   ├── runtime.py
│   └── schemas.py
└── tools/
    ├── activate_can.py
    ├── capture_snapshot.py
    ├── list_can_ports.py
    ├── list_realsense_devices.py
    └── probe_dual_piper.py
```

`challenge_deploy/` 是公共实现层，`inference/`、`dagger/`、`tools/` 只是不同入口。

---

## 3. 原版文件到新实现的对应关系

### 3.1 temporal smoothing inference

原版：

- `baselines/kai0/train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_temporal_smoothing.py`

新实现：

- `deploy/inference/agilex_inference_openpi_temporal_smoothing.py`

对应关系：

- 原版从 ROS topic 订阅三路图像和双臂关节状态。
- 新版用 `RealSenseRig` 直接抓三路图像，用 `DualPiperSystem` 直接读双臂状态。
- 原版用 OpenPi client 推理得到 action chunk。
- 新版仍然走 OpenPi client，并保留 kai0 原本的 payload key 与相机顺序。
- 原版把 action chunk 放进 temporal smoothing buffer。
- 新版对应 `challenge_deploy/buffer.py`。
- 原版通过 ROS publisher 发回机器人。
- 新版直接调用 `piper_sdk` 的 `JointCtrl()` / `GripperCtrl()` 路径。

### 3.2 sync inference

原版：

- `baselines/kai0/train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_sync.py`

新实现：

- `deploy/inference/agilex_inference_openpi_sync.py`

这个入口不做异步 chunk buffer，而是按同步的 “采一帧 observation -> 推理 -> 执行动作” 循环运行。

### 3.3 DAgger collect

原版：

- `baselines/kai0/train_deploy_alignment/dagger/agilex/agilex_openpi_dagger_collect.py`

新实现：

- `deploy/dagger/agilex_openpi_dagger_collect.py`

保留的主要语义：

- inference 模式下由 OpenPi 控制 slave arms。
- `d` 进入 DAgger 模式。
- `space` 开始记录。
- `s` 保存当前 episode。
- `r` 返回 inference 模式。
- 保存后短时间内按 `w` 删除刚保存的 episode。
- 数据仍然保存为 HDF5，加三路相机视频。

### 3.4 raw collect_data

原版：

- `baselines/kai0/train_deploy_alignment/dagger/agilex/collect_data.py`

新实现：

- `deploy/dagger/collect_data.py`

这是这次补齐的 ROS-free raw 采集入口。它不依赖 OpenPi，不依赖键盘状态机，也不会给机械臂下发控制命令；它只按固定频率记录当前双臂状态和三路相机图像。

它保存的数据结构和 `agilex_openpi_dagger_collect.py` 使用同一个 `EpisodeCollector`：

- `/observations/qpos`
- `/observations/qvel`
- `/observations/effort`
- `/action`
- `/base_action`
- `video/<camera_name>/episode_<idx>.mp4`
- `episode_<idx>.metadata.json`

默认 `--action-source state`，也就是把当前 `qpos` 作为 action 写入。这适合 raw replay / 数据排查，因为 action 维度和 policy 输出保持一致，但不会为了采集而额外动机械臂。

如果只想要占位 action，可以用：

```bash
--action-source zeros
```

### 3.5 Piper ROS node

原版 `piper_start_ms_node_new.py` 虽然是 ROS 节点，但真正控制机械臂的部分本来就是直接调用 `piper_sdk`：

- `ConnectPort()`
- `GetArmJointMsgs()`
- `GetArmJointCtrl()`
- `GetArmGripperMsgs()`
- `GetArmEndPoseMsgs()`
- `EnableArm()`
- `MotionCtrl_1()`
- `MotionCtrl_2()`
- `JointCtrl()`
- `GripperCtrl()`
- `MasterSlaveConfig()`

新版没有重新设计控制层，而是把这些 SDK 调用从 ROS subscriber / publisher 外壳里剥出来，集中放进：

- `deploy/challenge_deploy/piper.py`

---

## 4. 共享模块说明

### 4.1 `challenge_deploy/piper.py`

核心类：

- `SinglePiperArm`
- `DualPiperSystem`

主要职责：

- 建立 CAN 端口连接。
- 读取 joint feedback、joint command feedback、gripper feedback、end pose、arm status。
- 把 Piper SDK 原始单位转换为 kai0 上层使用的 7 维单臂状态。
- 提供双臂 14 维 `qpos` / `qvel` / `effort`。
- 提供关节控制、末端位姿控制、gripper 控制、master/slave 配置接口。

单臂状态向量顺序：

```text
[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
```

双臂状态向量顺序：

```text
[left_arm_7d, right_arm_7d]
```

读状态接口：

```python
state = robot.read_state()
print(state.qpos)
```

控制接口：

```python
robot.command_joint_positions(action_14d)
```

`tools/probe_dual_piper.py` 和 `dagger/collect_data.py` 会显式使用只读连接；inference / DAgger 部署入口按真实部署入口处理。

### 4.2 `challenge_deploy/realsense.py`

核心类：

- `RealSenseRig`

主要职责：

- 枚举 RealSense 设备。
- 按序列号启动三台相机。
- warmup 后抓取 RGB 帧。
- 保存三相机快照。
- 停止 pipeline。

这里直接使用 `pyrealsense2`，不经过 ROS camera topic。

### 4.3 `challenge_deploy/runtime.py`

核心类：

- `DualPiperObservationSource`

职责是把 `DualPiperSystem` 和 `RealSenseRig` 组合成一个统一的 snapshot 来源：

```python
snapshot = source.capture_snapshot()
```

一个 snapshot 包含：

- 双臂状态。
- 三路相机 RGB 图像。
- 时间戳。
- 面向 policy 的 observation。
- 面向 dataset 的 observation。

### 4.4 `challenge_deploy/observation.py`

保留 kai0 中和 OpenPi payload 对齐的图像预处理习惯：

- `jpeg_mapping()`
- `resize_with_pad()`
- `build_policy_payload()`

相机 key 对应关系：

```text
cam_high        -> observation/image/top_head
cam_right_wrist -> observation/image/hand_right
cam_left_wrist  -> observation/image/hand_left
```

### 4.5 `challenge_deploy/buffer.py`

实现 temporal smoothing action buffer：

- 新 action chunk 到来时按 `latency_k` 裁掉前缀。
- 新旧 chunk 重叠区间做线性平滑。
- 主循环每次 `pop_next_action()` 取下一步动作。

这部分保持 kai0 原版 temporal smoothing 的运行语义。

### 4.6 `challenge_deploy/policy.py`

负责 OpenPi client 的 action 请求。

导入策略：

1. 直接导入当前环境中的 `openpi_client`。
2. 如果导入失败，明确报错，要求把 OpenPI client 安装到当前 Python 环境。

因此：

- 只做 CAN / 相机 / raw collect 测试时，不需要 OpenPi client。
- 真正跑 policy inference 时，当前环境必须可导入 `openpi_client`。

### 4.7 `challenge_deploy/dataset.py`

负责 episode 数据保存：

- episode frame 累积。
- HDF5 写入。
- MP4 视频导出。
- 删除刚保存的 episode。

HDF5 字段：

```text
/observations/qpos
/observations/qvel
/observations/effort
/action
/base_action
```

视频路径：

```text
<dataset_root>/video/cam_high/episode_<idx>.mp4
<dataset_root>/video/cam_right_wrist/episode_<idx>.mp4
<dataset_root>/video/cam_left_wrist/episode_<idx>.mp4
```

### 4.8 `challenge_deploy/can_tools.py`

负责 CAN 设备发现和激活：

- `list_can_ports()`
- `resolve_can_name()`
- `activate_can_interface()`

它对应原版 `can_activate.sh` / `activate_can_arms.sh` 的功能，但实现为 Python 工具。

---

## 5. 保留的 kai0 原版语义

### 5.1 相机顺序

保留 kai0 的三相机命名顺序：

```python
CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
```

OpenPi payload 对应：

- `cam_high` -> `top_head`
- `cam_right_wrist` -> `hand_right`
- `cam_left_wrist` -> `hand_left`

### 5.2 关节单位换算

保留 kai0 原本的换算常数：

- `0.017444 / 1000`：把 Piper joint feedback 的 `0.001 degree` 风格单位转为 rad。
- `57324.840764`：把 rad 转回 Piper SDK joint command unit。

实现位置：

- `deploy/challenge_deploy/constants.py`
- `deploy/challenge_deploy/conversions.py`

没有把这些常数替换成看起来更标准的数学常数，因为部署数据和原版 kai0 行为一致性更重要。

### 5.3 gripper 语义

保留 kai0 的第 7 维 gripper 语义：

- 读状态：`grippers_angle / 1_000_000`
- 发命令：`round(gripper * 1_000_000)`

因此单臂 action 仍然是 7 维，双臂 action 仍然是 14 维。

### 5.4 数据格式

raw collect 和 DAgger collect 共用同一套 dataset writer，保持 kai0 风格：

- 状态和动作写 HDF5。
- 图像视频单独写 MP4。
- dataset name / task name 作为目录层级。
- episode index 使用 `episode_<idx>.hdf5`。

---

## 6. 和原版不同的地方

### 6.1 删除 ROS 依赖

已删除：

- `rospy`
- `sensor_msgs`
- `piper_msgs`
- `cv_bridge`
- ROS launch
- ROS topic pub/sub

替代：

- `piper_sdk` 直接访问 CAN。
- `pyrealsense2` 直接抓 RGB。
- Python dataclass 承载运行时状态。

### 6.2 删除 `dm_env` 硬依赖

原版 collect 里用 `dm_env.TimeStep` 包 observation。

当前目标环境没有必要引入这个依赖，所以新版 `EpisodeCollector` 直接接收：

```python
collector.add_frame(observation, action)
```

数据内容保持一致，部署依赖更少。

### 6.3 视频导出使用 OpenCV

当前 venv 里有 `cv2` / `h5py`，没有必要强制引入 `av`。

因此视频导出使用：

```python
cv2.VideoWriter
```

### 6.4 运动控制入口不再用额外 CLI 开关

部署入口现在按真实部署行为处理：

- temporal smoothing inference 会在主循环里直接发送 action。
- sync inference 会在同步循环里直接发送 action。
- DAgger collect 在需要时会发送 policy action 或 master/slave 配置。

只读验证使用专门入口：

- `tools/probe_dual_piper.py`
- `tools/capture_snapshot.py`
- `inference/agilex_inference_openpi_temporal_smoothing.py --probe-only`
- `dagger/collect_data.py`

这样比“同一个部署命令有时 dry-run、有时真动”更接近 kai0 原版，也更不容易误判当前命令到底有没有控制权。

---

## 7. 配置文件

默认配置：

- `deploy/configs/dual_piper_example.yaml`

### 7.1 robot

```yaml
robot:
  left:
    can_name: can0
  right:
    can_name: can1
  master_left:
    can_name: can_left_mas
  master_right:
    can_name: can_right_mas
```

说明：

- `left/right` 是实际执行 policy action 的从臂。
- `master_left/master_right` 是 DAgger / 示教时的主臂。
- 当前机器已经确认从臂默认是 `can0/can1`。

### 7.2 cameras

```yaml
cameras:
  enabled: true
  width: 640
  height: 480
  fps: 30
  warmup_frames: 30
  serials:
    cam_high: "323422071854"
    cam_right_wrist: "335522070790"
    cam_left_wrist: "344322073012"
```

这三个序列号已经在当前机器上枚举确认过：

- `cam_high = 323422071854`
- `cam_right_wrist = 335522070790`
- `cam_left_wrist = 344322073012`

### 7.3 policy

```yaml
policy:
  host: 127.0.0.1
  port: 8000
  prompt: fold the cloth
  inference_rate: 3.0
  chunk_size: 50
  latency_k: 8
  min_smooth_steps: 8
  buffer_max_chunks: 10
```

说明：

- `host/port` 指 OpenPi server。
- `prompt` 会放进 policy payload。
- `chunk_size`、`latency_k`、`min_smooth_steps`、`buffer_max_chunks` 控制 temporal smoothing 行为。

### 7.4 runtime

```yaml
runtime:
  publish_rate: 30
  max_publish_step: 10000
  arm_steps_length: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]
  right_gripper_offset: 0.003
```

说明：

- `publish_rate` 是部署循环频率。
- `max_publish_step` 是最长执行步数。
- `arm_steps_length` 对应 kai0 原版的小步长限制。
- `right_gripper_offset` 沿用原版命名；当前实现和原版一样会对 gripper command 应用 offset 逻辑。

### 7.5 dataset

```yaml
dataset:
  dataset_dir: ./data
  dataset_name: aloha_mobile_dummy
  export_video: true
  video_fps: 30
```

说明：

- raw collect 和 DAgger collect 都使用这里的 dataset 默认值。
- CLI 里传入的 `--dataset_dir` / `--task_name` 会覆盖这些默认值。

---

## 8. CAN 和相机现场知识

这部分没有照搬 `control_your_robot` 的实现，只借用了它里面已经确认过的现场知识。

参考文件：

- `control_your_robot/example/teleop/dual_piper_arm_teleop.py`
- `control_your_robot/calib/README.md`

当前机器确认的从臂 CAN：

- `left -> can0`
- `right -> can1`

当前机器确认的 RealSense：

- `323422071854`
- `344322073012`
- `335522070790`

最终优先级：

1. 现场设备枚举结果。
2. `deploy/configs/dual_piper_example.yaml`。
3. CLI 覆盖参数。

---

## 9. 使用方式

以下命令都使用你指定的 Python：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python
```

### 9.1 列 CAN 口

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python deploy/tools/list_can_ports.py
```

### 9.2 激活 / 重命名 CAN 口

例如把某个 USB-CAN 口整理成 `can_left_slave`：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/tools/activate_can.py can_left_slave \
  --usb-bus-info 1-13:1.0 \
  --bitrate 1000000 \
  --sudo
```

### 9.3 列 RealSense 设备

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python deploy/tools/list_realsense_devices.py
```

### 9.4 拍三相机快照

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/tools/capture_snapshot.py \
  --output-dir deploy/artifacts/snapshots
```

### 9.5 只读探测双臂通信

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/tools/probe_dual_piper.py \
  --left-can can0 \
  --right-can can1 \
  --samples 3 \
  --interval 0.2
```

这条命令只连接端口并读状态，不会使能机械臂，也不会发送 joint command。

### 9.6 inference 入口只读 probe

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/inference/agilex_inference_openpi_temporal_smoothing.py \
  --probe-only
```

这条命令会：

- 连接双臂。
- 启动三相机。
- 读一个 snapshot。
- 打印初始 `qpos`。
- 打印当前相机名。
- 退出。

### 9.7 raw collect_data

采一段 raw episode：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/dagger/collect_data.py \
  --dataset_dir deploy/data/raw \
  --task_name aloha_mobile_dummy \
  --max_timesteps 500 \
  --frame_rate 30
```

指定 episode index：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/dagger/collect_data.py \
  --dataset_dir deploy/data/raw \
  --task_name aloha_mobile_dummy \
  --episode_idx 12
```

不导出视频：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/dagger/collect_data.py \
  --dataset_dir deploy/data/raw \
  --task_name aloha_mobile_dummy \
  --no-video
```

使用全零 action 占位：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/dagger/collect_data.py \
  --dataset_dir deploy/data/raw \
  --task_name aloha_mobile_dummy \
  --action-source zeros
```

raw collector 的特点：

- 不需要 OpenPi server。
- 不进入 DAgger 键盘状态机。
- 不发送机械臂运动命令。
- 每帧记录双臂 qpos/qvel/effort 和三路 RGB。
- 默认把当前 qpos 写作 action，便于后续 raw replay / 格式检查。
- `--max_timesteps` 表示要保存的 action 数，最小为 1；实际会采 `max_timesteps + 1` 帧 observation 来保持 kai0 的 episode 对齐方式。

### 9.8 temporal smoothing inference

前提：

- OpenPi server 已经启动。
- 当前环境可导入 `openpi_client`。
- 现场已经清场，允许机械臂执行 policy action。

命令：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/inference/agilex_inference_openpi_temporal_smoothing.py \
  --host <gpu_host_ip> \
  --port 8000
```

这个入口会在主循环中把 policy action 下发到双臂。

### 9.9 sync inference

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/inference/agilex_inference_openpi_sync.py \
  --host <gpu_host_ip> \
  --port 8000
```

这个入口是同步推理和同步执行，不使用 temporal smoothing buffer。

### 9.10 OpenPI Piper client

这个入口是当前推荐的 pi0 / OpenPI rollout 入口。它是顶层脚本，不放在包内执行，也不需要改 `PYTHONPATH`：

```bash
/home/edemlab/challenge_ws/baselines/openpi/.venv/bin/python \
  /home/edemlab/challenge_ws/deploy/run_openpi_clients.py \
  --train-config pi05_slai_piper_click_bell_H30_Ajointgripper_Sjointgripper_0422 \
  --ckpt-dir /home/edemlab/challenge_ws/ckpts/Pi05-SLAIPiper-ClickBell-chunk30-Ajointgripper-Sjointgripper-30000 \
  --host 127.0.0.1 \
  --port 8000 \
  --prompt "click the bell" \
  --control-mode joints \
  --record
```

默认 `--execution-mode chunk_sync`，也就是同步执行一个 chunk 后再推理下一个 chunk。kai0 风格的异步推理 + temporal chunk-wise smoothing 可以显式打开：

```bash
--execution-mode streaming
```

`--prompt` 现在可以省略。省略时会按下面顺序解析：

1. `deploy/artifacts/trainconfig_prompts.json` 里缓存过的 `train_config -> prompt`
2. `HF_LEROBOT_HOME`，若未设置则默认 `/home/edemlab/challenge_ws/data` 下、由 train config 的 `repo_id` 对应到的本地 LeRobot dataset

如果两者都没有，而且也没传 `--prompt`，入口会直接报错退出。

当打开 `--record` 时，脚本还会尝试为这个 train config 对应的数据集生成并缓存：

- `deploy/artifacts/train_distributions/<repo_id>_cam_high_first_frame_overlay.png`
  含义：整个训练集所有 episode 的第一帧 `cam_high` 透明度叠加后的分布图
- `deploy/artifacts/trainconfig_prompts.json`
  含义：`train_config -> prompt` 的缓存

record 输出现在是一个目录，而不是单个 mp4：

- `deploy/artifacts/openpi_records/<record_name>/<record_name>_videos.mp4`
- `deploy/artifacts/openpi_records/<record_name>/<record_name>_frame1.png`

其中 `_frame1.png` 是“训练分布图在上 + 初始位姿后第一帧 obs 的 `cam_high` 在下”的对比图。`Ctrl+C` 分支也会走同一套保存逻辑。

如果要做 deploy benchmark，可以保存 controller timing：

```bash
--metrics-json /home/edemlab/challenge_ws/deploy/artifacts/metrics/streaming.json
```

更完整的 kai0 deploy 改进映射和 benchmark 方案见：

- `deploy/docs/kai0_deploy_improvements.md`

如果想提前把本地 `data/` 下的有效 LeRobot dataset 全部做一遍同样的缓存，可运行：

```bash
/home/edemlab/challenge_ws/baselines/openpi/.venv/bin/python \
  /home/edemlab/challenge_ws/deploy/tools/cache_lerobot_train_assets.py
```

### 9.11 DAgger collect

不连接 OpenPi，只启动 DAgger 外壳：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/dagger/agilex_openpi_dagger_collect.py \
  --no-policy
```

正常连接 OpenPi：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/dagger/agilex_openpi_dagger_collect.py \
  --host <gpu_host_ip> \
  --port 8000
```

启用 master rig：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/dagger/agilex_openpi_dagger_collect.py \
  --enable-master-rig
```

键位：

- `d`：进入 DAgger 模式。
- `space`：开始记录。
- `s`：保存当前 episode。
- `r`：返回 inference 模式。
- 保存后短时间内按 `w`：删除刚保存的 episode。

master rig 相关路径具备调用 `MasterSlaveConfig()`、`EnableArm()`、`MotionCtrl_1()` 的能力；本次没有真实执行这些会动机械臂的路径。

---

## 10. 本次实际测试记录

测试 Python：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python
```

测试时间：

- `2026-04-15`

### 10.1 CAN 枚举

实际读到：

- `can0`
  - state: `UP`
  - bus-info: `1-6:1.0`
  - bitrate: `1000000`
- `can1`
  - state: `UP`
  - bus-info: `1-13:1.0`
  - bitrate: `1000000`

结论：

- 双从臂链路在线。
- 当前机器默认 `can0/can1` 可用。

### 10.2 RealSense 枚举

实际读到 3 台 D435：

- `323422071854`
- `344322073012`
- `335522070790`

结论：

- 三台相机都能被 `pyrealsense2` 直接访问。
- 默认配置中的序列号和现场一致。

### 10.3 双臂只读 probe

执行过：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/tools/probe_dual_piper.py \
  --left-can can0 \
  --right-can can1 \
  --samples 3 \
  --interval 0.2
```

结果摘要：

- 左臂和右臂都成功读到真实 `qpos`。
- 两侧 `feedback_hz` / `status_hz` 基本在 `200 Hz`。
- `enabled = true`。
- `ctrl_mode = 1`。
- `arm_status = 0`。
- `command_hz = 0`，符合只读 probe 预期。

最终读到的 14 维 `qpos`：

```text
[-0.010675728, -0.004500552, -0.103076596, -0.037626708, 0.35167104, 0.185115728, 0.0,
  0.015542604,  0.009088324, -0.1897035,   0.07056098,  0.317951788, -0.014879732, 0.0]
```

### 10.4 三相机抓图

执行过：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/tools/capture_snapshot.py \
  --output-dir deploy/artifacts/snapshots
```

实际保存：

- `deploy/artifacts/snapshots/20260415_162226_cam_high.png`
- `deploy/artifacts/snapshots/20260415_162226_cam_left_wrist.png`
- `deploy/artifacts/snapshots/20260415_162226_cam_right_wrist.png`

实际分辨率：

- `480 x 640 x 3`

### 10.5 inference probe

执行过：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/inference/agilex_inference_openpi_temporal_smoothing.py \
  --probe-only
```

结果：

- 正常打印初始 `qpos`。
- 正常打印相机列表。
- `DualPiperObservationSource`、`RealSenseRig`、`DualPiperSystem` 能在最终 inference 入口中组合工作。

### 10.6 DAgger 入口无 policy 启动

执行过：

```bash
/home/edemlab/challenge_ws/control_your_robot/.venv/bin/python \
  deploy/dagger/agilex_openpi_dagger_collect.py \
  --no-policy
```

结果：

- 成功启动。
- 成功打印初始 slave state。
- 进入交互等待状态后手动 `Ctrl+C` 正常退出。
- 未触发 policy action、master/slave 配置或键盘采集动作。

---

## 11. 本次没有做的实测

这些代码路径已经保留或实现，但这次没有执行，因为会涉及真实机械臂动作：

- 真实 policy action 下发。
- `command_end_pose()` 末端位姿控制。
- master arm 的 `MasterSlaveConfig(0xFA)` / teach mode 切换。
- slave arm 的 `MasterSlaveConfig(0xFC)` 跟随配置。
- 真实连接 OpenPi server 跑完整远程 policy inference。
- 真实保存 DAgger episode。

已实测的是：

- CAN 枚举。
- RealSense 枚举。
- 双臂只读状态读取。
- 三相机拍照。
- inference 入口 probe。
- DAgger 入口无 policy 启动，未触发动作路径。

---

## 12. 建议验证顺序

如果要继续推进到真实运动，建议按这个顺序：

1. 先跑 `tools/probe_dual_piper.py`，确认反馈频率和状态正常。
2. 再跑 `tools/capture_snapshot.py`，确认三路相机画面和序列号对应关系正常。
3. 跑 `dagger/collect_data.py` 采一段 raw episode，确认 HDF5 和 MP4 都能写出。
4. 跑 inference 的 `--probe-only`，确认最终 observation source 能组合起来。
5. 启动 OpenPi server 后跑 temporal smoothing inference。
6. 最后再启用 DAgger master rig，因为 master/slave 配置是这里风险最高、也是本次未真实运动测试的路径。

---

## 13. 总结

`deploy/` 现在是一套尽量贴近 kai0 原版的真机部署代码，但底层不再依赖 ROS：

- Piper 双臂通过 `piper_sdk` pip 包访问。
- RealSense 通过 `pyrealsense2` 访问。
- 共享包名是 `challenge_deploy`。
- raw collect、temporal smoothing inference、sync inference、DAgger collect 都有对应入口。
- 通信、读状态、拍照和只读入口已经在当前机器上验证过。

真实运动路径已经写入入口脚本，但本次没有执行。
