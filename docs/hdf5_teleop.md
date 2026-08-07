# HDF5 Teleop 与 Rollout 数据说明

本项目有两条 HDF5 写入通路，它们复用同一套图像编码、Piper 状态转换和 reader，
但采样语义不同：

- 静态遥操：`run_hdf5_teleop_collect.py`，保留原 ROS collector 的一拍错位。
- rollout：六个 `run_*_client.py` 的 `--save`，只记录稳定控制 tick。

两者都使用本项目原生的 `piper_sdk` 与 `pyrealsense2`，不依赖 ROS topic、message
或 `PoseStamped`。

## 1. 静态遥操数据语义

### observation/action 一拍错位

collector 先采一帧 `FIRST observation`。之后每一拍继续采：

- `observation_t` 来自 slave 臂和相机；
- `action_t` 默认来自下一帧 master 臂状态。

写盘时 `action[i]` 与 `observation[i]` 具有相同 dataset index，但 action 来自
`frame[i + 1]`。这是一项保留的行为克隆对齐语义，不适用于 rollout `--save`。
`--action-from-state` 会改为使用下一帧 slave 状态，但仍保留一拍偏移。

### HDF5 布局

静态文件包含：

- `/observations/qpos`、`qpos_feedback`、`qpos_command`、`qvel`、`effort`、
  `end_pose`；
- `/observations/eef_quaternion`、`eef_left_time`、`eef_right_time`；
- 32D `/state` 与 `/action`；
- 根级单 episode `/language_instruction`；
- `/observations/source_timestamps/*`；
- `/observations/images/*` 中的 JPEG bytes，以及启用 depth 时
  `/observations/images_depth/*` 中的 PNG bytes，均为 HDF5 `vlen uint8`。

32D state/action 每臂为
`[joint6 rad, eef_pos3 m, eef_rot6d, gripper01]`。EEF 时间是相对 episode 首帧，
不是系统绝对时间。`base_action` 仍为零向量，因为项目没有底盘里程计链路。
静态文件不写 `/is_intervention`。

### CAN 拓扑

角色定义固定如下：master 是搭载示教器、由操作者拖动的臂；slave 是搭载夹爪、
执行目标的臂。

`can_topology: shared` 保持现有真机通路：同侧 master/slave 共用 CAN，collector
为同一 SDK 单例缓存创建 master-control 与 slave-feedback 两个只读逻辑视图，
遥操动作仍由固件原生 FA→FC 完成。该分支不探测角色、不写 `0x470`、不增加
USB serial gate，也不创建 host gateway。

`can_topology: isolated` 构造四台独立物理臂，先校验四路 USB serial，再通过
`hardware/factory.py`、`hardware/isolated.py` 完成 no-jump 对齐与角色
事务，通过 `hardware/linkage_gateway.py` 将每侧 master 的完整、已校验控制帧组
转发给对应 slave。项目不设置机械 qpos/qvel、关节/夹爪范围、初始差值、settle
或镜像误差阈值；任一侧通信失败都在写 slave CAN 前 fault 两侧。其他健康检查失败也会
停止 gateway，并让仍可达的 FC slave 尽力 hold；不会单侧继续采集。

### 异步时间对齐

静态 collector 保留原 ROS 版本的 `deque + timestamp barrier`：

- 每个 RealSense 相机一个线程，持续采集 color/depth 并按时间戳入队；
- 每个 Piper arm 一个线程，读取 SDK 后台 CAN 缓存，只在对应时间戳前进时入队；
- `frame_time` 取最新 camera/depth 时间戳中的最小值；
- 每个队列丢弃 `< frame_time` 的样本，再取第一个 `>= frame_time` 的样本组成帧；
- slave joint/pose 生成 observation，下一帧 master joint/pose 生成默认 action。

保存后在 HDF5 同目录生成：

- `episode_*_alignment_plot<N>_frames<T>.json`
- `episode_*_alignment_plot<N>_frames<T>.png`

它们记录入队时间戳、被选中时间戳、偏移统计和抽样可视化。

### 静态 collector 交互

静态采集要求 TTY：idle 按 `c` 开始、`q` 退出；active 按 `s` 停止，active
期间的 `q` 不会直接退出；停止后按 `c` 保存并继续，或按 `d` 丢弃并继续。
`episode_<idx>_running.txt` sentinel 被删除时也会提前停止当前 episode。

默认 dataset 根目录为 `artifacts/hdf5_data`，默认 task 为 `dummy_task`。

静态入口的 `--record/--recording` 会从保存的 episode 生成诊断视频，
`--save-sep` 额外生成逐相机视频。默认情况下这些视频与对应 HDF5 位于同一
episode 目录；`--record-dir` 可单独覆盖视频目录。HDF5 及 alignment JSON/PNG
仍保留在 `dataset-dir/task-name` 下。

## 2. Rollout `--save` 数据语义

六个 rollout runner 都把成功提交的稳定 tick 交给
`rollout/hdf5.py`：

- 只记录 `ROLLOUT_ACTIVE` 与 `INTERVENE_ACTIVE`；
- 初始插值、角色切换、对齐、hold、pause 和 `ROLLOUT_REACQUIRE` 不进入训练序列；
- 静止但成功提交的稳定 tick 仍记录；
- `/state` 来自同 tick 的 fresh slave observation；
- `/action` 是该 tick 实际选中并提交的 canonical 目标；
- `/is_intervention[t]` 与同一条 action 对齐，ROLLOUT 为 `false`，INTERVENE
  为 `true`。

未启用 `--intervention` 的 rollout 文件仍包含 `/is_intervention`，其值全为
`false`。静态 teleop 文件则完全没有该 dataset。rollout source timestamp 使用
`slave_*` 命名；reader 会把历史文件中的 `puppet_*` 同时映射为 `slave_*`，不会
重写旧文件。

`--save` 与 `--record/--recording` 互相独立：前者生成训练 HDF5，后者生成诊断
视频、动作/状态 NPZ、frame1 和相关指标。`--record-steps` 只限制诊断捕获；达到
该值后控制和 HDF5 继续，episode 边界的 `x` 也不会扩展它。`--save` 要求配置中
的全部相机持续可用。

正常停止后按 `c` 保存 HDF5、按 `d` 丢弃 HDF5；诊断产物不随 `d` 删除。
可捕获的 Ctrl-C、CAN 或相机故障会尽力保留已完成稳定 tick 的 partial episode。

## 3. 原子写入与 writer 槽

HDF5 先写到目标目录中每个 writer 独有的 `.tmp`，完整关闭后无覆盖地原子发布
目标文件。rollout 最多
同时运行两个 `spawn` writer，不设置第三条等待队列：

- 0 或 1 个 writer 活跃时，idle 可以按 `c` 开始新 episode；
- 2 个 writer 活跃时，idle 的 `c` 无效；
- idle 或退出等待期间可按 `k` 终止最老 writer，只清理其自己的 `.tmp` 并写失败记录；
- idle 按 `q` 后会等待剩余 writer 完成。

SIGKILL 或工控机突然断电仍可能丢失当前尚未提交的 episode。

## 4. 文件入口

- 静态采集：`run_hdf5_teleop_collect.py`
- episode 可视化：`run_hdf5_teleop_episode_vis.py`
- 静态共享实现：`teleop/hdf5_teleop.py`
- rollout HDF5：`rollout/hdf5.py`
- 多 episode/session：`rollout/interactive.py`、`rollout/session.py`
