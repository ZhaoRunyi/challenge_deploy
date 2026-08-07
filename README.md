# Challenge Deploy

## Python 环境

项目固定使用 Python 3.10（`pyproject.toml` 要求 `>=3.10,<3.11`）。代码与
`uv.lock` 落定后，在仓库根目录按锁文件同步环境。RealSense 是正式硬件依赖，
测试、dry-run 和真机入口使用同一个环境：

```bash
uv sync --frozen --python 3.10
uv run --frozen python -m unittest -v \
  tests.test_collect_data_robot \
  tests.test_dual_arm_rollout_robot \
  tests.test_intervene_four_arm_rollout_robot \
  tests.test_camera
uv run --frozen python -m run_hdf5_teleop_collect --help
```

`--frozen` 不会临时改写依赖解析结果。如果锁文件尚未生成，应先完成代码和
依赖声明，再由维护者执行 `uv lock --python 3.10`，复核并提交 `uv.lock`；不要在
真机运行时隐式更新环境。

所有源码入口都从仓库根目录以 `python -m run_<baseline>_client` 运行，或使用
`pyproject.toml` 中已安装的 console entry；项目不修改 `sys.path` 或 `PYTHONPATH`。

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

## 机械臂角色与 CAN 拓扑

- `master_left/master_right` 是搭载示教器、由操作者拖动的 master 臂。
- `slave_left/slave_right` 是搭载夹爪、执行目标的 slave 臂。
- `robot.left/right` 在运行时公共接口中始终表示逻辑 slave，现有 client 不需要
  感知两臂或四臂的物理构造差异。

根配置字段 `can_topology` 必须显式为下面二者之一：

| 配置 | 物理连接 | rollout 构造 | 遥操通路 |
|---|---|---|---|
| `shared` | 同侧 master/slave 并联在同一 CAN | 两个逻辑 slave | 保持已部署的固件原生 FA→FC 通路 |
| `isolated` | 四台臂各自通过 USB-to-CAN 接入工控机 | 默认只构造两台 slave | 静态遥操或 `--intervention` 时构造四臂，由工控机语义网关转发 |

`isolated` 普通 rollout 不要求 master 接口存在或上电，只控制两台 slave。
加入 `--intervention` 后才构造四臂并允许 ROLLOUT↔INTERVENE 动态切换。
`shared --intervention` 不受支持，并会在构造相机和 CAN 前失败。shared 模式不会
增加 USB serial、角色探测、`0x470` 写入或 host gateway，避免改变现有 golden
path。

`configs/dual_piper_example.yaml` 是已经逐臂复核 serial 的 isolated 配置。
原 shared 拓扑示例位于 `configs/dual_piper_shared.yaml`。

项目不配置或复述机械臂关节/夹爪范围，也不增加 qpos/qvel、初始差值、settle
或 master/slave 镜像误差阈值。物理运动限制由 Piper SDK/固件负责；host 侧只保留
帧新鲜度、CAN 帧组完整性、输入 shape/finite 校验和跨 epoch 旧动作隔离。

## 顶层模块

```text
challenge_deploy/
├── clients/     # policy client abstraction and concrete OpenPI/X-VLA/OpenPI-sim/Motus/DreamZero/FastWAM clients
├── hardware/    # topology factory, shared/isolated Piper, semantic gateway, RealSense, schemas
├── rollout/     # authority/coordinator, interactive session, HDF5, recording, metrics, train assets
├── teleop/      # HDF5 teleop collector and episode preview
├── tests/       # four focused robot/camera scenario checks
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

静态 `run_hdf5_teleop_collect.py` 不是简单复刻原始 ROS collector，而是保留其
关键数据语义：

- 默认 dataset 根目录为 `artifacts/hdf5_data`，默认 task 为 `dummy_task`

- `observation/action` 的一拍错位对齐
- RGB JPEG / depth PNG 的 HDF5 内联压缩格式
- 同时保存 `qpos/qvel/effort` 与 `eef_quaternion/eef_6d`
- `language_instruction` 的单 episode 级写法
- `episode_<idx>_running.txt` sentinel 提前终止机制
- `episode_vis.py` 的三视角拼接布局

静态 collector 默认把 `--record` 合成诊断视频及 `--save-sep` 逐相机视频写在
对应 HDF5 所在的 episode 目录；`--record-dir` 可单独覆盖视频目录。HDF5 与
alignment JSON/PNG 写在 dataset task 目录。

六个 rollout runner 的 `--save` 使用另一条稳定 tick 记录通路：只记录
`ROLLOUT_ACTIVE` 和 `INTERVENE_ACTIVE` 中实际成功提交的动作，不记录角色切换、
对齐、hold 或 reacquire。rollout HDF5 的 `/action` 是实际提交的 canonical
目标，并带有逐 tick 的 `/is_intervention`；静态 teleop 文件不写这个字段，且
只有静态 teleop 保留上面的一拍错位。

设备层已经替换为当前项目原生实现：

- 机械臂：`piper_sdk`
- 相机：`pyrealsense2`

## 当前不包含

- ROS topic / message / roslaunch 适配层
- 历史 `dagger/`、旧 `inference/`、旧 `tools/`
- SAM3 源码和权重只在未命中已有训练分布图、需要现场生成分割素材时使用；缺失
  时普通 import、dry-run、rollout 和已有素材读取不受影响。既有 submodule 元数据
  保持不变，本次环境重建不自动拉取或安装它。

运行产物目录 `artifacts/` 保留，不在代码裁剪范围内。

## Rollout 交互与保存

六个 rollout 入口都是保持硬件、相机和模型连接的多 episode TTY session：

- idle：`c` 开始新 episode，`q` 退出；有后台 HDF5 writer 时可按 `k` 终止最老
  writer。
- active：`s` 停止；启用 `--intervention` 时，`i` 进入 INTERVENE，`r` 返回
  ROLLOUT。
- 达到 `--rollout-steps`：`x` 增加 `ceil(initial_rollout_steps / 2)`，`s` 停止；
  设为 `0` 表示无限，不出现边界提示。
- 使用 `--save` 时，正常停止后 `c` 保存训练 HDF5，`d` 丢弃 HDF5。partial
  episode 会尽力自动保存已完成的稳定 tick。

HDF5 writer 最多同时运行两个 `spawn` 进程，没有第三条等待队列；两个槽都忙
时 idle 的 `c` 无效。退出时会等待 writer，期间仍可用 `k` 终止最老 writer。
每个 writer 使用同目录唯一 `.tmp`，完成后无覆盖地原子发布；终止 writer 只清理
它自己持有的临时文件，不删除未知的崩溃遗留文件。

`--record` 及其精确别名 `--recording` 生成诊断视频、动作/状态 NPZ、frame1 和
相关指标；`--save` 生成训练 HDF5，两者彼此独立。`--record-steps` 只限制当前
episode 的诊断捕获，不停止控制或 HDF5，`x` 也不会扩展它。`--save` 要求配置
中的全部相机可用，不能与 `--no-cameras` 一起使用。运行事件默认写入
`artifacts/runtime_events/` 下的 JSONL，也可用 `--event-log` 覆盖路径。

`streaming` 与 `chunk_sync` 都通过异步推理 worker，终端切换不会被网络请求
阻塞。`streaming` 按 cadence 请求并对重叠 chunk 做时延裁剪/平滑；
`chunk_sync` 等当前 action buffer 耗尽后再接收下一 chunk，不做跨 chunk 平滑。
每次控制权切换都会提升 epoch；旧 epoch 的在途结果、缓存动作和 session 状态
不会在切换后执行。

`--dry-run` 始终是单次、非交互、零 CAN/相机/模型连接的冷启动检查。仓库内可
确定 schema 的 runner 会完整校验维度；OpenPI、Motus、DreamZero 等依赖外部训练
配置的 runner 不会在 dry-run 中导入或读取该配置，而会在 JSON plan 中明确标记
`external-config-deferred`。这只表示本地 CLI、拓扑和执行 wiring 已通过；外部模型
schema 仍必须在真实运行的无运动 preflight 中通过，不能把 deferred 当作真机放行。

## 安全边界

可捕获的软件、相机或 CAN 故障会停止新动作和网关，并让仍可达的 FC 臂尽力按
当前反馈 hold。已经丢失 CAN 的臂无法再接收 hold；机械臂或电机断电时也无法靠
软件保证位置。INTERVENE 中的 FA master 是可回拖输入臂，工控机或控制进程
消失后不能保证其刚性保持。真机测试必须机械支撑、先单臂/单对空载，再逐步扩展
到双对和实际负载。

## 四项场景检测

测试目录严格只保留四个入口：

- `test_collect_data_robot.py`：isolated collect-data 四臂静态遥操机械通路；
- `test_dual_arm_rollout_robot.py`：isolated 普通 rollout，只构造和控制两台 slave；
- `test_intervene_four_arm_rollout_robot.py`：isolated 四臂
  ROLLOUT→INTERVENE→ROLLOUT、网关和故障收口；
- `test_camera.py`：RealSense 枚举、并发采集、时间戳和失败清理。

前三项只检测机械臂，不构造相机、模型 client、HDF5 writer 或终端 UI。公共行为
通过 production factory、facade 和 authority controller 复用，不为测试复制一套
控制实现。运行全部检测：

```bash
uv run --frozen python -m unittest -v \
  tests.test_collect_data_robot \
  tests.test_dual_arm_rollout_robot \
  tests.test_intervene_four_arm_rollout_robot \
  tests.test_camera
```

当前没有 shared 拓扑检测；这是明确的待补项，不能把 isolated 检测结果当作
shared golden-path 验收。真机运行仍须机械支撑，并使用 AgileX 官方工具单独完成
固件检查；本项目不会自动查询、升级或刷写固件。

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
uv run --frozen python -m run_xvla_client \
  --config configs/dual_piper_shared.yaml \
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
uv run --frozen python -m run_fastwam_client \
  --config configs/dual_piper_shared.yaml \
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

其中 5 个 unseen task 的首帧 distribution asset 应从
`<challenge_ws>/realsense_images` 人工复核后复制到
`artifacts/train_distributions/`；这些机器相关资产不随本仓库提交，缺失时只跳过
distribution 显示，不影响控制通路。
