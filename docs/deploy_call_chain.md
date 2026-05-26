# Deploy Call Chain

这个目录现在只保留三条最外层真机部署入口：

- `run_openpi_clients.py`
- `run_openpi_sim_client.py`
- `run_motus_client.py`

它们共享同一套真机运行时骨架：

- `hardware/config.py`
- `hardware/runtime.py`
- `hardware/piper.py`
- `hardware/realsense.py`
- `hardware/schemas.py`
- `hardware/conversions.py`
- `hardware/constants.py`

## 1. 总体调用链

```mermaid
flowchart TD
    A[run_openpi_clients.py] --> R
    B[run_openpi_sim_client.py] --> R
    C[run_motus_client.py] --> R

    R[runner: parse args / load config / init runtime] --> CFG[hardware/config.py]
    R --> OBS[hardware/runtime.py<br/>DualPiperObservationSource]
    R --> ROBOT[hardware/piper.py<br/>DualPiperSystem]
    R --> CAM[hardware/realsense.py<br/>RealSenseRig]

    OBS --> SNAP[RobotSnapshot]
    CAM --> SNAP
    ROBOT --> SNAP

    A --> OA[clients/openpi.py]
    B --> SA[clients/openpi_sim.py]
    C --> MA[clients/motus.py]

    OA --> WS1[openpi_client.websocket_client_policy]
    SA --> WS1
    MA --> WS2[Motus websocket_client_policy]

    WS1 --> S1[OpenPI server]
    WS2 --> S2[Motus server]

    A --> ROLL[rollout/execution.py]
    C --> ROLL
    B --> SIMROLL[run_openpi_sim_client.py internal rollout]

    ROLL --> CMD[client.decode_action / client.command_action]
    SIMROLL --> CMD

    CMD --> ARM[SinglePiperArm.command_joint_positions<br/>or command_end_pose]
    ARM --> SDK[piper_sdk.C_PiperInterface]
    SDK --> CAN[CAN bus / Piper hardware]
```

## 2. OpenPI 真机链

```mermaid
flowchart TD
    A[run_openpi_clients.py] --> A1[load_piper_policy_spec]
    A --> A2[prepare_train_assets / resolve_prompt]
    A --> A3[OpenPiPiperClient]
    A --> A4[_make_runtime]

    A4 --> A5[DualPiperSystem]
    A4 --> A6[RealSenseRig]
    A4 --> A7[DualPiperObservationSource]

    A --> A8[robot.connect]
    A --> A9[cameras.start]
    A --> A10[wait_until_ready]
    A --> A11[robot.enable]
    A --> A12[robot.move_to_joint_positions INIT_JOINTS]
    A --> A13[first_obs_snapshot = source.capture_snapshot]

    A13 --> A14{execution_mode}
    A14 -->|chunk_sync| A15[openpi_rollout.run_chunk_sync_rollout]
    A14 -->|streaming| A16[openpi_rollout.run_temporal_smoothing_rollout]

    A15 --> A17[source.capture_snapshot]
    A16 --> A17
    A17 --> A18[OpenPiPiperClient.build_payload]
    A18 --> A19[build_full_piper_state]
    A18 --> A20[build_policy_payload]
    A20 --> A21[OpenPI websocket client]
    A21 --> A22[OpenPI serve_policy.py]
    A22 --> A23[SLAIPiperInputs]
    A23 --> A24[policy outputs action chunk]

    A24 --> A25[OpenPiPiperClient.decode_action]
    A25 --> A26{control_mode}
    A26 -->|joints| A27[left/right arm.command_joint_positions]
    A26 -->|ee_pose| A28[left/right arm.command_end_pose]

    A27 --> A29[piper_sdk JointCtrl + GripperCtrl]
    A28 --> A30[piper_sdk EndPoseCtrl + GripperCtrl]
```

## 3. OpenPI Sim 真机链

```mermaid
flowchart TD
    B[run_openpi_sim_client.py] --> B1[load_openpi_sim_policy_spec]
    B --> B2[resolve_prompt]
    B --> B3[OpenPiSimPiperClient]
    B --> B4[_make_runtime]

    B4 --> B5[DualPiperSystem]
    B4 --> B6[RealSenseRig]
    B4 --> B7[DualPiperObservationSource]

    B --> B8[robot.enable]
    B --> B9[robot.move_to_joint_positions initial_joints]
    B --> B10[first_obs_snapshot with gripper backfill]
    B --> B11[run_openpi_sim_client.run_chunk_sync_rollout]

    B11 --> B12[source.capture_snapshot]
    B12 --> B13[_snapshot_with_grippers]
    B13 --> B14[OpenPiSimPiperClient.build_payload]
    B14 --> B15[build_configured_piper_state fixed 14D]
    B14 --> B16[224x224 resized three-view images]
    B16 --> B17[OpenPI websocket client]
    B17 --> B18[OpenPI sim server]
    B18 --> B19[EmbodiChain action chunk]

    B19 --> B20[OpenPiSimPiperClient.decode_action]
    B20 --> B21[sim_gripper_to_piper]
    B21 --> B22[left/right arm.command_joint_positions]
    B22 --> B23[piper_sdk JointCtrl + GripperCtrl]
```

## 4. Motus 真机链

```mermaid
flowchart TD
    C[run_motus_client.py] --> C1[load_motus_policy_spec]
    C --> C2[MotusPiperClient]
    C --> C3[get_server_metadata / resolve prompt]
    C --> C4[_make_runtime]

    C4 --> C5[DualPiperSystem]
    C4 --> C6[RealSenseRig]
    C4 --> C7[DualPiperObservationSource]

    C --> C8[robot.enable]
    C --> C9[robot.move_to_joint_positions INIT_JOINTS]
    C --> C10{execution_mode}
    C10 -->|chunk_sync| C11[openpi_rollout.run_chunk_sync_rollout]
    C10 -->|streaming| C12[openpi_rollout.run_temporal_smoothing_rollout]

    C11 --> C13[source.capture_snapshot]
    C12 --> C13
    C13 --> C14[MotusPiperClient.build_payload]
    C14 --> C15[build_policy_frame T-shape image]
    C14 --> C16[build_normalized_policy_state]
    C16 --> C17[normalize by stat.json]
    C17 --> C18[Motus websocket client]
    C18 --> C19[MotusRemotePolicy]
    C19 --> C20[normalized action chunk]

    C20 --> C21[client denormalize actions]
    C21 --> C22[MotusPiperClient.decode_action]
    C22 --> C23{binary gripper transition?}
    C23 -->|yes| C24[_command_transition_step]
    C23 -->|no| C25[_command_decoded]

    C24 --> C26[arm.command_joint_positions / command_end_pose]
    C25 --> C26
    C26 --> C27[piper_sdk JointCtrl or EndPoseCtrl + GripperCtrl]
```

## 5. 共享硬件层

```mermaid
flowchart TD
    S[DualPiperObservationSource.capture_snapshot] --> I[RealSenseRig.capture]
    S --> J[DualPiperSystem.read_state]

    J --> L[left SinglePiperArm.read_state]
    J --> R[right SinglePiperArm.read_state]

    L --> SDK1[C_PiperInterface.GetArmJointMsgs]
    L --> SDK2[C_PiperInterface.GetArmGripperMsgs]
    L --> SDK3[C_PiperInterface.GetArmEndPoseMsgs]
    L --> SDK4[C_PiperInterface.GetArmStatus]

    R --> SDK1
    R --> SDK2
    R --> SDK3
    R --> SDK4

    C[command_action] --> Q1[SinglePiperArm.command_joint_positions]
    C --> Q2[SinglePiperArm.command_end_pose]

    Q1 --> QC1[MotionCtrl_2]
    Q1 --> QC2[JointCtrl]
    Q1 --> QC3[GripperCtrl]

    Q2 --> QC4[MotionCtrl_2]
    Q2 --> QC5[EndPoseCtrl]
    Q2 --> QC6[GripperCtrl]
```

## 6. 当前保留范围

当前 `challenge_deploy/` 只围绕下面这些文件保留：

- `run_openpi_clients.py`
- `run_openpi_sim_client.py`
- `run_motus_client.py`
- `configs/dual_piper_example.yaml`
- `docs/deploy_call_chain.md`
- `rollout/buffer.py`
- `hardware/config.py`
- `hardware/constants.py`
- `hardware/conversions.py`
- `rollout/assets.py`
- `clients/motus.py`
- `clients/openpi.py`
- `rollout/execution.py`
- `clients/openpi_sim.py`
- `hardware/piper.py`
- `hardware/realsense.py`
- `rollout/recording.py`
- `rollout/metrics.py`
- `hardware/runtime.py`
- `hardware/schemas.py`
