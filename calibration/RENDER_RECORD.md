# 渲染实验记录（不作为可迁移渲染接口）

本轮不纳入渲染辅助逻辑。`render.py`、`docker_render.py`、renderer smoke 脚本仅留本机且被 Git 忽略；不影响非渲染标定流程。后续如何最小接入需要单独设计和验证。

## 已有输入与输出

以下相对于 `artifacts/calibration/platform_1/`：

- 输入图：`reference/cam_high_rect.png`。
- 机器人状态、相机内参：`reference/sample.json`；图像来源：`reference/snapshot.json`。反馈在照片之后读取，非同步数据，仅用于定性对照。
- 外参：`results/cam_high.json`，高相机相对左臂基座，米、OpenCV 光学坐标。
- 已有叠图：`results/cam_high_overlay.png`。
- 实际读回的关节、内参、相机坐标转换、可见像素：`results/render_report.json`。
- 当次 robot-only 场景配置：`results/calibrated_sim_config.json`。这不是完整任务配置，不应直接覆盖任务文件。

上述文件纳入 manifest 哈希清单。报告中的额外图片文件名是原实验输出名，不表示本包包含所有中间图。高相机机械臂可见像素为 47,490；两路内参读回与请求一致。结果不是独立的标定精度验收。

## 原实验环境与调用方式

- 宿主工作目录：`/home/edemlab/challenge_ws/challenge_deploy`。
- Docker 容器：`embodichain`。
- 宿主 `../embodichain_ws` 挂载为 `/root/workspace`。
- 容器 Python：`/opt/rsc-render-env/bin/python`，独立环境使用 DexSim 0.4.3；原 py310 环境为 0.4.2。
- 外部辅助脚本：`/root/workspace/calib_real_vs_sim/render_robot_calib_overlay.py`。
- 外部基础场景：`/root/workspace/Embodied_Challenge/configs/pour_water/gym_config.json`。
- 原始实验输出：`artifacts/calibration/high_20261106_161025/reprojection_refinement/render/`。

以下是本机调用方式记录，不是另一台机器克隆后即可执行的承诺；需要上述外部代码、资源与环境：

```bash
P=artifacts/calibration/platform_1
python -m calibration.docker_render \
  --sample "$P/reference" --results "$P/results/cam_high.json" \
  --output "$P/local_runs/render_new" --alpha 0.5
```

本记录不把当前外部仓库 HEAD 冒充历史执行版本；当次依赖未完整锁定，因此不能保证像素级重现。保留的场景配置和读回报告是当次行为依据。

## 后续最小接口需保留的约束

独立渲染入口只接收 sample、外参和输出目录，仿真引擎/模型留在仿真项目。按模型控制组映射关节，不猜交错索引；夹爪全开口除二对应双指位移；超限应报错，不能静默截断。以基座实际变换拼接世界位姿，OpenCV 到 OpenGL 翻转 Y/Z；读取实际 K 和 qpos 检验应用情况。禁止隐式缩放真机图。

接入前还需确定仿真依赖版本、资源定位、错误处理及独立多姿态验收。本轮不实施这些渲染改造。
