# platform_1

本机平台 1 的可迁移标定包。冻结日期：2026-11-06。

## 范围与状态

- 仅 `cam_high` 外参：左臂基座下的相机位姿 `T_base_camera`；米，OpenCV 光学坐标（右、下、前）。
- 三台相机的彩色流内参均已实际读取；**左右腕外参尚未标定**。
- 当前最佳算法候选，不是已完成独立精度验收的正式真值。
- 原始会话 `high_20261106_161025` 的第 **3、4、5、6、7、10、11、12、17、20** 组，共 10 组；目录依次命名为 `sample_01` 至 `sample_10`，原名映射保存在 `manifest.json` 的 `sample_sources` 中。
- 样本曾按前 20 组拟合残差筛选，因此不存在严格独立的测试集。不要将较小的训练残差解释为真实外参精度。

## 内容

| 路径 | 内容 |
|---|---|
| `samples/` | 10 组完整原始样本：三路 raw/rect 图、关节/EEF/FK、检测角点及采集时内参 |
| `intrinsics/realsense_color_640x480_30fps.json` | 2026-11-06 17:23:59（北京时间）实际读回的三台彩色流内参、序列号、固件版本 |
| `solve.py` | 独立、离线、纯计算标定脚本，不依赖仓库模块、相机、机械臂、Docker 或网络 |
| `read_intrinsics.py` | 独立内参读取脚本，仅访问相机，不连接机械臂 |
| `results/cam_high.json` | 固定算法从这 10 组样本重新求解的外参及质量指标 |
| `results/optimization_comparison.json` | 原实验算法优化前后指标；非入包样本无法仅凭本包重算 |
| `results/cam_high_overlay.png` | 当前候选与确认对齐照片的仿真叠图 |
| `reference/` | 叠图所用真机照片、内参和后读取的机器人反馈；两者不是同步采集 |
| `coarse_reference/` | 粗对齐用历史三路图与 observation.state，含固定 HF revision |
| `board/` | AprilTag A4 打印文件及几何定义 |
| `hardware.yaml` | 本机硬件配置模板；新机器需修改 CAN 和序列号 |
| `manifest.json` | 来源、原样本编号及文件 SHA-256 清单 |

## 固定算法

从样本的 `T_camera_board` 和真实 EEF 反馈，以 OpenCV PARK、DANIILIDIS 两种方法生成初值；同时优化相机外参 X 和末端到标记变换 Y：

```text
T_camera_tag(i) = inverse(X) @ T_base_eef(i) @ Y
```

以四角点重投影误差为目标，使用 SciPy `least_squares`，`soft_l1` 损失、1 px 尺度；固定 K 和 100 mm 标记尺寸，选择收敛候选中代价较小者。检查变换、姿态多样性、正深度和雅可比秩。不采用手调偏移，不混入旧平台/上游外参。

当前训练角点 RMS：**1.689056 px**。原实验中前段未用于本次拟合的 10 组 RMS 为 4.289764 px，但这些样本参与过前置筛选，不是严格独立测试集。

## 离线复现

Python 3.10。可将整个 `platform_1` 文件夹复制到其他机器：

```bash
python -m pip install -r platform_1/requirements.txt
python platform_1/solve.py
```

执行前自动校验包内文件 SHA-256。默认输出到包内 `local_runs/<时间>/result.json`，不会覆盖冻结结果；也可用 `--output /新路径/result.json`。本脚本没有任何真机运动接口。

重新读取内参需要额外安装 `pyrealsense2==2.59.1.11053`、连接对应设备、关闭其他占用相机的进程：

```bash
python platform_1/read_intrinsics.py --output platform_1/local_runs/intrinsics_new.json
```

序列号固定在脚本中，输出必须使用新文件；不要直接覆写本包已冻结的内参。深度内参、深度到彩色外参不属于本次输出。

## 完整流程与渲染记录

其他机器的粗对齐、人工采集、求解流程见 [工作流](../../../calibration/README.md)。新高相机会话可运行 `solve.py --session /采集目录 --output /新结果.json`；仍限定左臂 EEF、高相机。

渲染辅助代码暂不纳入，已有输出及环境记录见 [渲染记录](../../../calibration/RENDER_RECORD.md)。照片对应反馈在照片之后读取，叠图只用于定性对照，不代表独立验收通过。

## Git 与本机清理

- `.gitignore` 只放行本包明确列出的源码、样本、内参、结果与参考目录；`local_runs/` 自动忽略。
- calibration 下旧采集、历史数据缓存、手调试验、旧算法比较、失败渲染全部继续留在本机，不随 Git 上传。
- 本次不删除旧数据，避免破坏报告引用及尚需审查的异常样本。审查后可优先删除失败渲染空目录和可重新下载的缓存，再考虑归档完整旧会话；不要只因残差超标就删除原始样本。
- 本包的文件变更需重新审查并更新清单；不要用跳过校验来掩盖修改。
