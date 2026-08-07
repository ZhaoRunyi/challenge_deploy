from __future__ import annotations

from dataclasses import dataclass, field
import select
import sys
import termios
import time
import tty
from typing import Any, Callable

from .authority import ExecutionState
from .execution import RolloutMetrics
from .hdf5 import RolloutHDF5WriterPool
from .session import (
    EpisodeStepLimit,
    SessionCommand,
    SessionScreen,
    command_for_key,
    prompt_for_screen,
)


KeyReader = Callable[[], str | None]
EpisodeStartCallback = Callable[[int], None]
EpisodeFinishCallback = Callable[[int, str, bool], Any]
WriterResultCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class InteractiveEpisodeResult:
    episode_number: int
    stable_steps: int
    stop_reason: str
    partial: bool
    saved: bool


@dataclass(slots=True)
class InteractiveSessionResult:
    metrics: RolloutMetrics
    episodes: list[InteractiveEpisodeResult] = field(default_factory=list)
    writer_results: list[dict[str, Any]] = field(default_factory=list)
    rollout_session_ids: list[str] = field(default_factory=list)


def read_terminal_key() -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    return sys.stdin.read(1).lower()


def _drive_until_state(
    coordinator: Any,
    expected: set[ExecutionState],
    *,
    fps: float,
) -> Any:
    while True:
        tick_started_s = time.monotonic()
        tick = coordinator.step()
        if tick.state is ExecutionState.FAULT or tick.state in expected:
            return tick
        _sleep_tick(tick_started_s, fps)


def _sleep_tick(started_s: float, fps: float) -> None:
    if fps <= 0.0:
        return
    remaining_s = 1.0 / fps - (time.monotonic() - started_s)
    if remaining_s > 0.0:
        time.sleep(remaining_s)


def _wait_for_command(
    *,
    screen: SessionScreen,
    intervention_enabled: bool,
    writer_slots_available: Callable[[], bool],
    key_reader: KeyReader,
    writer_pool: RolloutHDF5WriterPool | None,
    on_writer_result: WriterResultCallback | None,
) -> SessionCommand:
    last_prompt: str | None = None
    while True:
        if writer_pool is not None:
            for result in writer_pool.poll():
                if on_writer_result is not None:
                    on_writer_result(result)
        slots_available = writer_slots_available()
        writer_active_count = writer_pool.active_count if writer_pool is not None else 0
        prompt = prompt_for_screen(
            screen,
            intervention_enabled=intervention_enabled,
            writer_slots_available=slots_available,
            writer_active_count=writer_active_count,
        )
        if prompt != last_prompt:
            print(prompt, flush=True)
            last_prompt = prompt
        command = command_for_key(
            key_reader(),
            screen=screen,
            intervention_enabled=intervention_enabled,
            writer_slots_available=slots_available,
        )
        if command is SessionCommand.KILL_OLDEST_WRITER and writer_pool is not None:
            failure = writer_pool.terminate_oldest()
            if failure is not None and on_writer_result is not None:
                on_writer_result(failure)
            continue
        if command is not SessionCommand.NONE:
            return command
        time.sleep(0.05)


def _wait_for_writers(
    *,
    writer_pool: RolloutHDF5WriterPool,
    key_reader: KeyReader,
    on_writer_result: WriterResultCallback,
) -> None:
    """Drain outstanding saves while still allowing explicit oldest-first kills."""

    last_active_count: int | None = None
    while True:
        for writer_result in writer_pool.poll():
            on_writer_result(writer_result)
        active_count = writer_pool.active_count
        if active_count == 0:
            return
        if active_count != last_active_count:
            print(
                f"Waiting for {active_count} HDF5 save(s); press k to terminate the oldest.",
                flush=True,
            )
            last_active_count = active_count
        if (key_reader() or "").lower() == "k":
            failure = writer_pool.terminate_oldest()
            if failure is not None:
                on_writer_result(failure)
            last_active_count = None
            continue
        time.sleep(0.05)


def _stop_episode_safely(
    coordinator: Any,
    *,
    fps: float,
    reason: str,
    already_paused: bool = False,
) -> Any:
    if not already_paused:
        coordinator.request_pause(reason=f"hold before stopping episode: {reason}")
        tick = _drive_until_state(
            coordinator,
            {ExecutionState.EPISODE_PAUSED},
            fps=fps,
        )
        if tick.state is ExecutionState.FAULT:
            return tick
    coordinator.request_stop(reason=reason)
    return _drive_until_state(coordinator, {ExecutionState.IDLE}, fps=fps)


def run_interactive_session(
    *,
    coordinator: Any,
    rollout_steps: int,
    fps: float,
    intervention_enabled: bool,
    save_hdf5: bool,
    writer_pool: RolloutHDF5WriterPool | None = None,
    key_reader: KeyReader = read_terminal_key,
    on_episode_start: EpisodeStartCallback | None = None,
    on_episode_finish: EpisodeFinishCallback | None = None,
    on_writer_result: WriterResultCallback | None = None,
    require_tty: bool = True,
) -> InteractiveSessionResult:
    """Keep hardware/model resources connected across a multi-episode session."""

    terminal_settings = None
    metrics = RolloutMetrics(execution_mode=coordinator.execution_mode)
    result = InteractiveSessionResult(metrics=metrics)
    episode_number = 0
    last_command_started_s: float | None = None
    active_episode_number: int | None = None
    active_step_limit: EpisodeStepLimit | None = None
    active_intervention = False
    episode_starting = False

    def writer_slots_available() -> bool:
        return writer_pool is None or writer_pool.can_start_episode()

    def report_writer_result(writer_result: dict[str, Any]) -> None:
        result.writer_results.append(writer_result)
        if on_writer_result is not None:
            on_writer_result(writer_result)

    def finish_episode(
        *,
        current_episode_number: int,
        step_limit: EpisodeStepLimit,
        stop_reason: str,
        partial: bool,
    ) -> None:
        nonlocal active_episode_number, active_step_limit, active_intervention
        save_episode = bool(save_hdf5 and partial)
        if save_hdf5 and not partial:
            save_command = _wait_for_command(
                screen=SessionScreen.SAVE_DECISION,
                intervention_enabled=intervention_enabled,
                writer_slots_available=writer_slots_available,
                key_reader=key_reader,
                writer_pool=writer_pool,
                on_writer_result=report_writer_result,
            )
            save_episode = save_command is SessionCommand.SAVE

        episode_payload = None
        if on_episode_finish is not None:
            episode_payload = on_episode_finish(
                current_episode_number,
                stop_reason,
                partial,
            )
        saved = False
        if save_episode and episode_payload is not None:
            if writer_pool is None:
                raise RuntimeError("HDF5 writer pool is unavailable")
            try:
                writer_pool.submit(episode_payload)
                saved = True
            except Exception as exc:
                report_writer_result(
                    {
                        "ok": False,
                        "error": repr(exc),
                        "output_path": str(getattr(episode_payload, "output_path", "")),
                    }
                )
        payload_steps = getattr(episode_payload, "steps", None)
        stable_steps = (
            len(payload_steps)
            if payload_steps is not None
            else step_limit.executed_steps
        )
        result.episodes.append(
            InteractiveEpisodeResult(
                episode_number=current_episode_number,
                stable_steps=stable_steps,
                stop_reason=stop_reason,
                partial=partial,
                saved=saved,
            )
        )
        active_episode_number = None
        active_step_limit = None
        active_intervention = False

    def finish_partial_episode(stop_reason: str) -> str:
        if active_episode_number is not None and active_step_limit is not None:
            try:
                _stop_episode_safely(
                    coordinator,
                    fps=fps,
                    reason=stop_reason,
                )
            except Exception as stop_error:
                stop_reason = f"{stop_reason}; stop_failed={stop_error!r}"
            finish_episode(
                current_episode_number=active_episode_number,
                step_limit=active_step_limit,
                stop_reason=stop_reason,
                partial=True,
            )
        elif episode_starting:
            try:
                _stop_episode_safely(
                    coordinator,
                    fps=fps,
                    reason=stop_reason,
                )
            except Exception as stop_error:
                stop_reason = f"{stop_reason}; stop_failed={stop_error!r}"
        return stop_reason

    try:
        if rollout_steps < 0:
            raise ValueError("rollout_steps must be non-negative")
        if fps < 0.0:
            raise ValueError("fps must be non-negative")
        if save_hdf5 and writer_pool is None:
            raise ValueError("save_hdf5 requires a RolloutHDF5WriterPool")
        if require_tty and not sys.stdin.isatty():
            raise RuntimeError(
                "interactive rollout requires a TTY; use --dry-run for wiring checks"
            )
        if require_tty:
            terminal_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())

        coordinator.request_start(reason="interactive session startup")
        startup_tick = _drive_until_state(coordinator, {ExecutionState.IDLE}, fps=fps)
        if startup_tick.state is ExecutionState.FAULT:
            metrics.stop_reason = f"FAULT: {startup_tick.fault!r}"
            return result

        while True:
            command = _wait_for_command(
                screen=SessionScreen.IDLE,
                intervention_enabled=intervention_enabled,
                writer_slots_available=writer_slots_available,
                key_reader=key_reader,
                writer_pool=writer_pool,
                on_writer_result=report_writer_result,
            )
            if command is SessionCommand.QUIT:
                break
            if command is not SessionCommand.START:
                continue

            episode_number += 1
            last_command_started_s = None
            step_limit = EpisodeStepLimit(rollout_steps)
            active_intervention = False
            episode_starting = True
            if on_episode_start is not None:
                on_episode_start(episode_number)
            episode_starting = False
            # A camera preflight or initial alignment failure occurs before an
            # episode can contain a stable training tick.  Mark the episode
            # active only after that startup callback succeeds, otherwise the
            # exception path would try to finish an HDF5 collector that never
            # began and mask the original startup failure.
            active_episode_number = episode_number
            active_step_limit = step_limit
            coordinator.request_rollout(reason=f"start episode {episode_number}")
            stop_reason = "stopped_by_user"
            partial = False
            last_active_prompt: str | None = None

            while True:
                tick_started_s = time.monotonic()
                active_screen = (
                    SessionScreen.INTERVENE
                    if active_intervention
                    else SessionScreen.ROLLOUT
                )
                active_prompt = prompt_for_screen(
                    active_screen,
                    intervention_enabled=intervention_enabled,
                )
                if active_prompt != last_active_prompt:
                    print(active_prompt, flush=True)
                    last_active_prompt = active_prompt
                active_command = command_for_key(
                    key_reader(),
                    screen=active_screen,
                    intervention_enabled=intervention_enabled,
                )
                if active_command is SessionCommand.ENTER_INTERVENE:
                    coordinator.request_intervention(reason="terminal i")
                elif active_command is SessionCommand.ENTER_ROLLOUT:
                    coordinator.request_rollout(reason="terminal r")
                elif active_command is SessionCommand.STOP:
                    tick = _stop_episode_safely(
                        coordinator,
                        fps=fps,
                        reason="terminal s",
                    )
                    if tick.state is ExecutionState.FAULT:
                        partial = True
                        stop_reason = f"FAULT: {tick.fault!r}"
                    break

                tick = coordinator.step()
                active_intervention = tick.state is ExecutionState.INTERVENE_ACTIVE
                for inference_seconds in tick.inference_seconds:
                    metrics.record_inference(inference_seconds)
                for inference_error in tick.inference_errors:
                    metrics.record_inference_error(RuntimeError(inference_error))
                if tick.command_committed:
                    period_seconds = (
                        None
                        if last_command_started_s is None
                        else tick_started_s - last_command_started_s
                    )
                    last_command_started_s = tick_started_s
                    metrics.record_command(
                        period_seconds=period_seconds,
                        command_seconds=tick.command_seconds or 0.0,
                    )
                    if step_limit.record_stable_step():
                        coordinator.request_pause(reason="episode step limit reached")
                        paused_tick = _drive_until_state(
                            coordinator,
                            {ExecutionState.EPISODE_PAUSED},
                            fps=fps,
                        )
                        if paused_tick.state is ExecutionState.FAULT:
                            partial = True
                            stop_reason = f"FAULT: {paused_tick.fault!r}"
                            break
                        boundary_command = _wait_for_command(
                            screen=SessionScreen.STEP_LIMIT,
                            intervention_enabled=intervention_enabled,
                            writer_slots_available=lambda: True,
                            key_reader=key_reader,
                            writer_pool=writer_pool,
                            on_writer_result=report_writer_result,
                        )
                        if boundary_command is SessionCommand.EXTEND:
                            step_limit.extend()
                            if active_intervention:
                                coordinator.request_intervention(
                                    reason="resume INTERVENE after episode extension"
                                )
                            else:
                                coordinator.request_rollout(
                                    reason="resume ROLLOUT after episode extension"
                                )
                        else:
                            tick = _stop_episode_safely(
                                coordinator,
                                fps=fps,
                                reason="episode stopped at step limit",
                                already_paused=True,
                            )
                            if tick.state is ExecutionState.FAULT:
                                partial = True
                                stop_reason = f"FAULT: {tick.fault!r}"
                            else:
                                stop_reason = "step_limit_stopped"
                            break
                elif tick.state in (
                    ExecutionState.ROLLOUT_REACQUIRE,
                    ExecutionState.ROLLOUT_ACTIVE,
                ):
                    metrics.empty_action_polls += 1
                if tick.state is ExecutionState.FAULT:
                    partial = True
                    stop_reason = f"FAULT: {tick.fault!r}"
                    metrics.stop_reason = stop_reason
                    break
                _sleep_tick(tick_started_s, fps)

            finish_episode(
                current_episode_number=episode_number,
                step_limit=step_limit,
                stop_reason=stop_reason,
                partial=partial,
            )
            if partial:
                break
    except KeyboardInterrupt as exc:
        metrics.mark_interrupted(repr(exc))
        failure_reason = (
            f"interrupted during episode startup: {exc!r}"
            if episode_starting
            else f"interrupted: {exc!r}"
        )
        completed_reason = finish_partial_episode(failure_reason)
        if episode_starting and completed_reason != failure_reason:
            metrics.stop_reason = completed_reason
    except Exception as exc:
        metrics.stop_reason = f"runtime_error: {exc!r}"
        metrics.stop_reason = finish_partial_episode(metrics.stop_reason)
        raise
    finally:
        try:
            try:
                coordinator.close()
            except Exception as exc:
                print(f"Failed to close rollout coordinator cleanly: {exc}", flush=True)
            if writer_pool is not None:
                try:
                    _wait_for_writers(
                        writer_pool=writer_pool,
                        key_reader=key_reader,
                        on_writer_result=report_writer_result,
                    )
                    writer_results = writer_pool.close(terminate=False)
                except KeyboardInterrupt:
                    try:
                        writer_results = writer_pool.close(terminate=True)
                    except BaseException as exc:
                        writer_results = [{"ok": False, "error": repr(exc)}]
                except Exception as exc:
                    writer_results = [{"ok": False, "error": repr(exc)}]
                for writer_result in writer_results:
                    try:
                        report_writer_result(writer_result)
                    except Exception as exc:
                        print(f"Failed to report HDF5 writer result: {exc}", flush=True)
        finally:
            if terminal_settings is not None:
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    terminal_settings,
                )
    return result


__all__ = [
    "InteractiveEpisodeResult",
    "InteractiveSessionResult",
    "read_terminal_key",
    "run_interactive_session",
]
