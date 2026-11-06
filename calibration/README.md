# 粗对齐与三相机标定

正式范围：历史姿态粗对齐、实时图像比较、人工采集和离线求解。渲染暂不集成，只保留 [渲染记录](RENDER_RECORD.md)。以下从仓库根目录执行。

## 安装与设备配置

使用部署程序同一个 Python 3.10 环境，依赖统一维护在根目录 pyproject.toml 的 calibration 依赖组。不要另外创建标定专用环境。窗口需要系统 Tk 和本机桌面；Ubuntu 可安装 python3-tk，Conda Python 需在同一环境安装 tk。RealSense/CAN 驱动、设备权限由操作者配置。

```bash
# 先激活运行 challenge_deploy 的现有环境；以下从仓库根目录运行。
python -m pip install -e '.[calibration]'
python -m pip check
python -c "import cv2, tkinter, scipy, pyrealsense2, piper_sdk; assert hasattr(cv2, 'aruco')"
P=artifacts/calibration/platform_1
RUN="$P/local_runs/setup_2"
mkdir -p "$RUN"
cp "$P/hardware.yaml" "$RUN/hardware.yaml"
CFG="$RUN/hardware.yaml"
```

全新机器可先用 `python3.10 -m venv .venv` 和 `source .venv/bin/activate` 建立统一环境，部署与标定均使用它。只安装 opencv-python，不并装 opencv-python-headless 或 contrib 变体；旧环境若混装，应先卸载这些 OpenCV 包，再重新安装上述依赖。platform_1/requirements.txt 仅用于把独立离线样本包交给没有本仓库的用户，不是部署环境的第二套安装入口。

编辑 CFG 的 CAN 名称和相机序列号，不能照搬本机设备身份。local_runs 被 Git 忽略；每次采集使用新输出目录。

## 1. 粗对齐

coarse_reference 是餐具整理 episode 0 / frame 434 的历史三路图片和 observation.state；reference 则是本机已确认的照片和后读取反馈，二者不可互换。

```bash
python -m calibration.live_overlay --config "$CFG" \
  --reference "$P/coarse_reference" --output "$RUN/live"
```

窗口默认最大化，滑块调透明度，F11 全屏，s 保存，Esc 退出。只读相机，不移动机械臂。

以下两条默认仅预检查：

```bash
python -m calibration.set_reference_pose --config "$CFG" \
  --reference "$P/coarse_reference" --output "$RUN/joints"
python -m calibration.set_reference_grippers --config "$CFG" \
  --reference "$P/coarse_reference" --output "$RUN/grippers" --raw-to-meters 0.1
```

实际运动须由现场操作者核对机器人型号、关节映射、空间和线缆，清空运动范围并准备急停，再加 --execute。历史姿态不保证另一套机器的运动路径安全。程序不自动使能或解除急停。关节默认速度上限 0.15 rad/s，历史实机使用 0.06 rad/s，默认速度尚未实机验收。比例 0.1 仅适用本参考，夹爪目标左 61.9 mm、右 47.6 mm。

## 2. 打印与固定

打印 `$P/board/apriltag_a4.pdf`，A4、100%，关闭页面适配。实测最外黑框 100 mm，不包括白边；AprilTag 36h11 / ID 0，贴在平整硬板上。

| 相机 | 模式 | 固定方式 | 输出 |
|---|---|---|---|
| cam_high | eye-to-hand | 板刚性固定在左臂所选参考上 | T_base_camera |
| cam_left_wrist | eye-in-hand | 板固定在桌面，整轮不移动 | T_reference_camera |
| cam_right_wrist | eye-in-hand | 同上 | T_reference_camera |

相机必须相对所选参考刚体固定。eef 是 SDK 反馈，link1 至 link6 是 FK，不能随意互换。T_A_B 将 B 变换至 A，单位米；相机坐标为 OpenCV 右、下、前。

## 3. 采集

先关闭叠图窗口，避免抢占相机。采集工具只读，不控制运动。操作者摆位并等待静止，l 暂存、s 保存、q 丢弃暂存、x 退出。窗口显示 Tag 和通过 PnP 检查的坐标轴。

```bash
python -m calibration.cli capture --config "$CFG" --board "$P/board/board.json" \
  --camera cam_high --arm left --reference eef --mode eye-to-hand \
  --output "$RUN/high" --window
python -m calibration.cli capture --config "$CFG" --board "$P/board/board.json" \
  --camera cam_left_wrist --arm left --reference eef --mode eye-in-hand \
  --output "$RUN/left" --window
python -m calibration.cli capture --config "$CFG" --board "$P/board/board.json" \
  --camera cam_right_wrist --arm right --reference eef --mode eye-in-hand \
  --output "$RUN/right" --window
```

逐条运行，每轮建议 15–25 个姿态、至少 10 个，覆盖不同区域和距离，绕至少两轴旋转，不能只平移。整轮不得改变安装关系。保存三路 raw/rect 图、SDK 内参、双臂关节/EEF/FK；保留原始数据，不自动删除离群点。

## 4. 求解

通用 PARK 手眼求解支持高相机与腕相机：

```bash
python -m calibration.cli solve "$RUN/high"
python -m calibration.cli solve "$RUN/left"
python -m calibration.cli solve "$RUN/right"
```

15 mm / 3 度是内部一致性筛查，不是实际精度保证。固定的最佳候选重投影算法另外提供独立脚本：

```bash
python "$P/solve.py"
python "$P/solve.py" --session "$RUN/high" --output "$RUN/high_refined.json"
```

第一条校验并复算包内 10 组；第二条处理新采集的高相机/左臂 EEF 会话。此优化脚本暂不支持腕相机，腕相机使用通用求解。不得把 platform_1 外参直接用于其他机器。

## 5. 验收数据

取下板、不动相机，在 2–3 个不同且轮廓清楚的安全姿态保存干净样本：

```bash
python -m calibration.cli capture --config "$CFG" --board "$P/board/board.json" \
  --clean --camera cam_high --arm left --reference eef --mode eye-to-hand \
  --output "$RUN/clean" --window
```

渲染接口待单独设计，不承诺克隆后即可渲染。已有叠图仅用于定性参考。

## 版本范围与检查

```bash
python -m pytest tests/test_calibration.py tests/test_platform1.py -q
```

Git 纳入通用非渲染模块、测试，以及 platform_1 的打印板、粗对齐参考、硬件模板、10 组样本、内参和结果。采集依赖仓库已有的硬件读取模块。旧实验与新会话输出不入库、不删除；渲染脚本暂留本机。
