from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .hdf5 import extend_rollout_limit


class SessionScreen(str, Enum):
    IDLE = "idle"
    ROLLOUT = "rollout"
    INTERVENE = "intervene"
    STEP_LIMIT = "step_limit"
    SAVE_DECISION = "save_decision"


class SessionCommand(str, Enum):
    NONE = "none"
    START = "start"
    QUIT = "quit"
    STOP = "stop"
    ENTER_INTERVENE = "enter_intervene"
    ENTER_ROLLOUT = "enter_rollout"
    EXTEND = "extend"
    SAVE = "save"
    DISCARD = "discard"
    KILL_OLDEST_WRITER = "kill_oldest_writer"


@dataclass(slots=True)
class EpisodeStepLimit:
    initial_steps: int
    current_limit: int = 0
    executed_steps: int = 0

    def __post_init__(self) -> None:
        if self.initial_steps < 0:
            raise ValueError("initial_steps must be non-negative")
        self.current_limit = int(self.initial_steps)

    @property
    def unlimited(self) -> bool:
        return self.initial_steps == 0

    @property
    def reached(self) -> bool:
        return not self.unlimited and self.executed_steps >= self.current_limit

    def record_stable_step(self) -> bool:
        self.executed_steps += 1
        return self.reached

    def extend(self) -> int:
        if self.unlimited:
            raise ValueError("an unlimited episode cannot be extended")
        self.current_limit = extend_rollout_limit(self.initial_steps, self.current_limit)
        return self.current_limit


def command_for_key(
    key: str | None,
    *,
    screen: SessionScreen,
    intervention_enabled: bool,
    writer_slots_available: bool = True,
) -> SessionCommand:
    """Translate one terminal key without performing hardware or save actions."""

    normalized = "" if key is None else key.lower()
    if normalized == "k" and screen is SessionScreen.IDLE:
        return SessionCommand.KILL_OLDEST_WRITER
    if screen is SessionScreen.IDLE:
        if normalized == "q":
            return SessionCommand.QUIT
        if normalized == "c" and writer_slots_available:
            return SessionCommand.START
        return SessionCommand.NONE
    if screen is SessionScreen.SAVE_DECISION:
        if normalized == "c":
            return SessionCommand.SAVE
        if normalized == "d":
            return SessionCommand.DISCARD
        return SessionCommand.NONE
    if screen is SessionScreen.STEP_LIMIT:
        if normalized == "x":
            return SessionCommand.EXTEND
        if normalized == "s":
            return SessionCommand.STOP
        return SessionCommand.NONE
    if screen in (SessionScreen.ROLLOUT, SessionScreen.INTERVENE):
        if normalized == "s":
            return SessionCommand.STOP
        if intervention_enabled and normalized == "i":
            return SessionCommand.ENTER_INTERVENE
        if intervention_enabled and normalized == "r":
            return SessionCommand.ENTER_ROLLOUT
    return SessionCommand.NONE


def prompt_for_screen(
    screen: SessionScreen,
    *,
    intervention_enabled: bool,
    writer_slots_available: bool = True,
    writer_active_count: int | None = None,
) -> str:
    if screen is SessionScreen.IDLE:
        if writer_active_count == 0:
            return "Idle: press c to start, or q to quit."
        if writer_slots_available:
            return "Idle: press c to start, k to terminate the oldest save, or q to quit."
        return "Idle: both save slots are busy; press k to terminate the oldest save, or q to quit."
    if screen is SessionScreen.SAVE_DECISION:
        return "Episode stopped: press c to save HDF5, or d to discard HDF5."
    if screen is SessionScreen.STEP_LIMIT:
        return "Episode step limit reached: press x to extend, or s to stop."
    if intervention_enabled:
        return "Active: press i for INTERVENE, r for ROLLOUT, or s to stop."
    return "ROLLOUT active: press s to stop."


__all__ = [
    "EpisodeStepLimit",
    "SessionCommand",
    "SessionScreen",
    "command_for_key",
    "prompt_for_screen",
]
