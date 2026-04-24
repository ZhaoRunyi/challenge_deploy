# kai0 Deploy Improvements And Benchmark Plan

This note maps the deploy-side improvements in `/home/edemlab/challenge_ws/baselines/kai0`
to the ROS-free implementation in `/home/edemlab/challenge_ws/deploy`.

Paper reference:

- https://arxiv.org/pdf/2602.09021

The paper's deploy-relevant pillar is Train-Deploy Alignment (TDA). In runtime code,
the important part is not model arithmetic or stage advantage training, but reducing
the mismatch between model action chunks and physical execution.

## 1. Improvements In kai0

### 1.1 Temporal Chunk-Wise Smoothing

Original code:

- `baselines/kai0/train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_temporal_smoothing.py`
- `baselines/kai0/train_deploy_alignment/dagger/agilex/agilex_openpi_dagger_collect.py`
- `baselines/kai0/train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_rtc.py`

Core mechanism:

- Run inference in a background thread.
- Execute actions in the control thread at a fixed publish rate.
- When a new chunk arrives, drop up to `latency_k` already-stale prefix actions.
- Blend the remaining old execution tail and new chunk over their overlap.
- Keep the last action as a fallback so a new chunk can be smoothly bridged even after the old queue is exhausted.

Why it matters:

- A chunked policy predicts an open-loop future from an old observation.
- Real deployment has server latency, camera latency, CAN command latency, and physical lag.
- Naively executing one full chunk before requesting the next causes pauses at chunk boundaries.
- Replacing the current chunk immediately can cause action discontinuities.
- Temporal chunk-wise smoothing keeps throughput high while reducing jerk and boundary jumps.

Current migration:

- Already existed as `challenge_deploy.buffer.StreamActionBuffer`.
- Now reused by `challenge_deploy.openpi_rollout.run_temporal_smoothing_rollout`.
- `deploy/run_openpi_clients.py` exposes `--execution-mode streaming`, which uses async inference plus temporal smoothing.
- Blocking behavior is the default as `--execution-mode chunk_sync`.

### 1.2 Non-Blocking Inference

Original code:

- `inference_fn_non_blocking_fast()` in kai0 temporal smoothing scripts.

Core mechanism:

- Policy inference does not block the action publish loop.
- The action loop consumes the smoothed buffer.
- The inference loop continuously refreshes the buffer using the newest observation.

Why it matters:

- If inference takes 200-800 ms, blocking at chunk boundaries produces visible stalls.
- Async inference converts latency into a buffer-update problem instead of a robot-stop problem.

Current migration:

- Implemented in `challenge_deploy.openpi_rollout.run_temporal_smoothing_rollout`.
- Camera and robot snapshot access is protected by a local lock so the inference thread and record/control path do not call RealSense concurrently.

### 1.3 Latency Compensation

Original code:

- `StreamActionBuffer.integrate_new_chunk(actions, max_k=args.latency_k, min_m=args.min_smooth_steps)`.

Core mechanism:

- `self.k` counts how many steps have been executed from the current chunk.
- A newly inferred chunk drops `min(self.k, latency_k)` front actions before blending.

Why it matters:

- Front actions in a freshly returned chunk correspond to time that has already passed during inference.
- Executing them late makes the robot chase stale model intentions.

Current migration:

- `--latency-k`, `--min-smooth-steps`, and `--buffer-max-chunks` are available in `run_openpi_clients.py`.
- Defaults come from `deploy/configs/dual_piper_example.yaml`.

### 1.4 Temporal Ensembling / Naive Async Baselines

Original code:

- `baselines/kai0/train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_temporal_ensembling.py`

Core mechanisms:

- `naive_async`: switch to newest chunk with minimal smoothing.
- `temporal_ensembling`: aggregate multiple predictions for the same timestep using exponential weights.

Why it matters:

- These are useful baselines for benchmarking.
- The paper reports temporal chunk-wise smoothing as stronger than temporal ensembling and RTC in most tested settings.

Current migration:

- Not migrated into the generic OpenPI Piper client.
- Reason: the main deploy path should use the paper's preferred method first, and we already preserve `chunk_sync` as a simple baseline.
- If needed, these baselines should be added as extra buffer classes in `challenge_deploy.buffer`, not inside `openpi_client.py`.

### 1.5 RTC

Original code:

- `baselines/kai0/train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_rtc.py`
- `baselines/kai0/src/openpi/models/pi0_rtc.py`

Core mechanism:

- The client sends `prev_action_chunk`, estimated inference delay, and execute horizon to an RTC-capable policy server.
- The server must run a `Pi0RTCConfig` model.

Why it matters:

- RTC shifts some latency handling into model-side action generation.
- It is conceptually orthogonal to temporal smoothing.

Current migration:

- Not migrated for the current generic client.
- Reason: the current checkpoint flow uses normal OpenPI policy serving; RTC requires an RTC config and compatible checkpoint/runtime.
- This should stay a separate optional path if added later.

### 1.6 Heuristic DAgger And Raw Collection

Original code:

- `baselines/kai0/train_deploy_alignment/dagger/agilex/agilex_openpi_dagger_collect.py`
- `baselines/kai0/train_deploy_alignment/dagger/agilex/collect_data.py`

Core mechanism:

- Run policy in the loop.
- Enter DAgger mode for human correction.
- Save intervention/recovery episodes.
- Optionally delete bad saves immediately.

Why it matters:

- This expands training data toward deployment failures and recovery states.

Current migration:

- ROS-free raw collection exists in `deploy/dagger/collect_data.py`.
- ROS-free DAgger shell exists in `deploy/dagger/agilex_openpi_dagger_collect.py`.
- These are intentionally separate from the generic OpenPI rollout client.

### 1.7 Spatio-Temporal Augmentation

Original code:

- `baselines/kai0/train_deploy_alignment/data_augment/time_scaling.py`
- `baselines/kai0/train_deploy_alignment/data_augment/space_mirroring.py`

Core mechanism:

- Time scaling keeps every Nth frame to simulate faster trajectories.
- Space mirroring swaps left/right arm vectors and flips images/videos.

Why it matters:

- It expands `Ptrain` around expected deployment variation.
- It is a training-data improvement, not a runtime controller.

Current migration:

- Not moved into `deploy/challenge_deploy`.
- Reason: it belongs with dataset preparation, not the runtime deploy package.

## 2. Benchmark Design

### 2.1 Runtime Controller Benchmark

Compare:

- `chunk_sync`: old blocking rollout.
- `streaming`: async inference + temporal chunk-wise smoothing.
- Optional future baselines: `naive_async`, `temporal_ensembling`, RTC.

Metrics:

- `success_rate`: task-level success over repeated trials.
- `retry_cost`: number of retries/interventions before success.
- `time_to_success_s`: wall-clock task duration.
- `policy_throughput_hz`: inferred chunks per second.
- `command_period_p95_s`: p95 command loop period; should stay near `1 / fps`.
- `empty_action_polls`: how often control loop had no action available.
- `boundary_jump`: L2 norm between the last action before a chunk update and first action after update.
- `action_velocity` and `action_jerk`: finite differences of commanded actions.
- `tracking_error`: commanded joint vector minus measured qpos when using joint control.

Current code support:

- `run_openpi_clients.py` has `--metrics-json`.
- `challenge_deploy.openpi_rollout.RolloutMetrics` records inference timing, command period, command duration, chunk count, empty polls, and inference errors.
- `--record` video already plots action/state curves for visual debugging.

Example:

```bash
/home/edemlab/challenge_ws/baselines/openpi/.venv/bin/python \
  /home/edemlab/challenge_ws/deploy/run_openpi_clients.py \
  --train-config pi05_slai_piper_click_bell_H30_Ajointgripper_Sjointgripper_0422 \
  --ckpt-dir /home/edemlab/challenge_ws/ckpts/Pi05-SLAIPiper-ClickBell-chunk30-Ajointgripper-Sjointgripper-30000 \
  --host 127.0.0.1 \
  --port 8000 \
  --prompt "click the bell" \
  --control-mode joints \
  --execution-mode streaming \
  --metrics-json /home/edemlab/challenge_ws/deploy/artifacts/metrics/streaming.json \
  --record
```

Baseline:

```bash
/home/edemlab/challenge_ws/baselines/openpi/.venv/bin/python \
  /home/edemlab/challenge_ws/deploy/run_openpi_clients.py \
  --train-config pi05_slai_piper_click_bell_H30_Ajointgripper_Sjointgripper_0422 \
  --ckpt-dir /home/edemlab/challenge_ws/ckpts/Pi05-SLAIPiper-ClickBell-chunk30-Ajointgripper-Sjointgripper-30000 \
  --host 127.0.0.1 \
  --port 8000 \
  --prompt "click the bell" \
  --control-mode joints \
  --execution-mode chunk_sync \
  --metrics-json /home/edemlab/challenge_ws/deploy/artifacts/metrics/chunk_sync.json \
  --record
```

### 2.2 No-Motion / Dry Benchmark

Use:

```bash
--dry-run
```

This validates:

- train config parsing
- server connectivity
- payload construction
- action shape and decoding
- camera/state snapshot

It does not benchmark the runtime controller because the action loop is not active.

### 2.3 Hardware Stress Benchmark

For controller timing without evaluating task success:

- Use a safe workspace.
- Set `--rollout-steps` small, e.g. 100-300.
- Compare `streaming` and `chunk_sync` with the same `--fps`, checkpoint, prompt, and initial pose.
- Save metrics and record videos.

Useful acceptance criteria:

- `streaming.command_period_seconds.p95` close to `1 / fps`.
- `streaming.empty_action_polls` small after first chunk.
- Fewer visible discontinuities in action/state plots at chunk boundaries.
- No repeated inference errors.

## 3. Migration Status Summary

Migrated into generic OpenPI deploy:

- Train config space parsing.
- Generic action/state/image spaces.
- Joint and EE-pose action decoding.
- Recording of camera/action/state.
- Initial pose move.
- Async inference + temporal chunk-wise smoothing.
- Latency prefix trim.
- Metrics export.

Already present elsewhere in deploy:

- ROS-free temporal smoothing inference script.
- ROS-free DAgger shell.
- ROS-free raw collector.

Not migrated by design:

- RTC, because it requires RTC server/model config.
- Temporal ensembling, because it is mainly a baseline and the paper favors temporal chunk-wise smoothing.
- Data augmentation, because it belongs in dataset conversion/training utilities, not runtime deployment.
