# HDF5 teleop Integration Notes

这个接入基于 `/home/edemlab/challenge_ws/data_collection.zip` 中两份核心代码：

- 原始 ROS collector
- `episode_vis.py`

目标不是重写一份“差不多能跑”的 collector，而是保留原作者在数据语义和落盘格式上的关键设计，同时把设备层替换成当前项目的：

- `piper_sdk`
- `pyrealsense2`

## 原实现里值得保留的设计

### 1. observation/action 一拍错位

原脚本会先采一帧 `FIRST observation`，之后每一拍都继续采：

- `observation_t` 来自 puppet arm + cameras
- `action_t` 实际保存的是下一拍的 master arm qpos

最终写盘时，`action[i]` 和 `observation[i]` 对齐，但 `action[i]` 来自 `frame[i + 1]` 的 master 状态。

这不是 bug，而是很有意图的行为克隆对齐方式。当前接入保留这个语义。

### 2. HDF5 内联压缩图像

原脚本不是把 RGB 图像直接存成 `(H, W, 3)` 数组，而是：

- RGB：JPEG bytes
- depth：PNG bytes
- HDF5 dataset：`vlen uint8`

这样和 `episode_vis.py` 的读取方式闭环一致，也能显著减小单 episode 体积。当前接入保留这个格式。

### 3. 同时保存 joint-space 和 EEF 派生空间

原脚本同时落盘：

- `qpos / qvel / effort`
- `eef_quaternion`
- `eef_6d`

这样后续无论训练端用关节空间还是笛卡尔空间，都不需要再从原始 HDF5 反推。当前接入保留这两组字段。

### 4. episode 内时间轴是相对首帧

原脚本把 `eef_left_time` / `eef_right_time` 写成：

- `frame_time - frame_0_time`

也就是 episode 内相对时间，而不是系统绝对时间。当前接入保留这一语义。

### 5. 单 episode 级语言字段

原脚本的 `language_instruction` 只在根级 dataset 写一次，而不是每帧重复。当前接入保留这个写法。

### 6. 可删除 sentinel 终止录制

原脚本会创建：

- `episode_<idx>_running.txt`

只要这个文件被删掉，就提前停止录制。这是很实用的人工中断机制。当前接入保留这个机制。

### 7. `episode_vis.py` 的三视角拼接方式

原可视化脚本使用：

- `cam_high` 作为主视图
- `cam_left_wrist` 与 `cam_right_wrist` 先纵向拼接，再缩小，再拼到主视图右边

当前接入保留这个视频布局。

## 新实现里的设备替换

原实现依赖 ROS topic：

- `master_arm_*`
- `puppet_arm_*`
- `camera_*`
- `PoseStamped`

当前替换为本项目原生设备层：

- master arms：`DualPiperSystem(read_only=True)` 读取 `robot.master_left/master_right`
- puppet arms：`DualPiperSystem(read_only=True)` 读取 `robot.left/right`
- cameras：`RealSenseRig`
- EEF pose：直接来自 `PiperArmState.end_pose`

## 当前接入文件

- 采集入口：`run_hdf5_teleop_collect.py`
- 可视化入口：`run_hdf5_teleop_episode_vis.py`
- 共享实现：`teleop/hdf5_teleop.py`

## 当前异步对齐逻辑

当前实现已经恢复原始 ROS collector 的 `deque + timestamp barrier` 语义，只是数据源从 ROS topic 换成了本地异步线程：

- 每个 RealSense camera 一个线程，持续 `wait_for_frames()` 并把 color/depth 分别按时间戳写入队列。
- 每个 Piper arm 一个线程，持续读取 `piper_sdk` 后台 CAN 线程缓存的状态，只在 SDK `time_stamp` 前进时入队。
- `frame_time = min(latest camera/depth timestamps)`，和原始 `get_frame()` 一致。
- 每个队列丢弃所有 `< frame_time` 的样本，然后取第一个 `>= frame_time` 的样本组成 frame。
- puppet joint + pose 队列生成 32D `observations/qpos`：每臂 `[joint6 rad, eef_pos3 m, eef_rot6d, gripper01]`。
- 下一帧 master joint + pose 队列生成同样 32D 的 `/action`。
- puppet pose 队列还生成 `eef_quaternion/eef_6d`；这里没有 ROS `PoseStamped`，pose 来自 `piper_sdk` 的 end-pose feedback。
- HDF5 额外写 `/observations/source_timestamps/*`，用于检查最终 frame 是由哪些异步源时间戳对齐出来的。
- 采集结束会在 HDF5 旁生成 `*_alignment_every<N>_frames<T>.json/png`，记录全量入队时间戳、被选中时间戳和偏移统计。

仍然保留的差异：`base_action` 固定为零向量，因为当前项目没有接入底盘里程计链路。
