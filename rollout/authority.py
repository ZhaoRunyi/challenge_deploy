from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import threading
from typing import Any, Iterator, Protocol
from uuid import uuid4

import numpy as np


class ExecutionState(str, Enum):
    """Stable and transitional states of one rollout/intervention supervisor."""

    STARTING = "STARTING"
    IDLE = "IDLE"
    ROLLOUT_REACQUIRE = "ROLLOUT_REACQUIRE"
    ROLLOUT_ACTIVE = "ROLLOUT_ACTIVE"
    TO_INTERVENE = "TO_INTERVENE"
    INTERVENE_ACTIVE = "INTERVENE_ACTIVE"
    TO_ROLLOUT = "TO_ROLLOUT"
    EPISODE_PAUSED = "EPISODE_PAUSED"
    FAULT = "FAULT"


ROLLOUT_STATES = frozenset(
    (ExecutionState.ROLLOUT_REACQUIRE, ExecutionState.ROLLOUT_ACTIVE)
)


class SupervisorEventKind(str, Enum):
    START = "start"
    ROLLOUT = "rollout"
    INTERVENE = "intervene"
    PAUSE = "pause"
    STOP = "stop"
    FAULT = "fault"


@dataclass(frozen=True, slots=True)
class SupervisorEvent:
    kind: SupervisorEventKind
    reason: str = ""
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class RolloutActionEnvelope:
    """A policy chunk tied to exactly one observation and authority epoch."""

    epoch: int
    request_id: int
    observation_seq: int
    session_id: str | None
    actions: np.ndarray

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float64).copy()
        actions.setflags(write=False)
        object.__setattr__(self, "actions", actions)


@dataclass(frozen=True, slots=True)
class InterventionActionSample:
    """The latest complete intervention target committed by the gateway.

    A stationary tick reuses the last target actually submitted to both slaves
    with ``is_new_dispatch=False``.  It never infers an action from ordinary
    slave feedback and never claims a synthetic frame sequence.
    """

    action_state: Any
    committed_monotonic_s: float
    frame_group_seq: int
    generation: int
    is_new_dispatch: bool = True


class AuthorityHardwareController(Protocol):
    """Hardware transaction boundary used only by the supervisor thread.

    An isolated Piper adapter owns the SDK role writes, no-jump seeding,
    alignment, and semantic gateway.  This protocol intentionally contains no
    Piper SDK calls and in particular no trajectory-recording/teaching command.
    """

    def start_authority(self) -> Any: ...

    def require_healthy(self) -> None: ...

    def enter_rollout(self) -> Any: ...

    def enter_intervention(self) -> Any: ...

    def resume_intervention(self) -> Any: ...

    def pause_intervention(self) -> Any: ...

    def pause_episode(self) -> None: ...

    def hold_position(self) -> None: ...

    def sample_intervention_action_state(self) -> InterventionActionSample | None: ...


@dataclass(frozen=True, slots=True)
class AuthoritySnapshot:
    state: ExecutionState
    epoch: int
    session_id: str | None
    fault: BaseException | None


@dataclass(frozen=True, slots=True)
class AuthorityTransition:
    previous: ExecutionState
    current: ExecutionState
    epoch: int
    session_id: str | None
    reason: str


_ALLOWED_TRANSITIONS: dict[ExecutionState, frozenset[ExecutionState]] = {
    ExecutionState.STARTING: frozenset((ExecutionState.IDLE, ExecutionState.FAULT)),
    ExecutionState.IDLE: frozenset(
        (
            ExecutionState.TO_ROLLOUT,
            ExecutionState.TO_INTERVENE,
            ExecutionState.EPISODE_PAUSED,
            ExecutionState.FAULT,
        )
    ),
    ExecutionState.ROLLOUT_REACQUIRE: frozenset(
        (
            ExecutionState.ROLLOUT_ACTIVE,
            ExecutionState.TO_INTERVENE,
            ExecutionState.EPISODE_PAUSED,
            ExecutionState.IDLE,
            ExecutionState.FAULT,
        )
    ),
    ExecutionState.ROLLOUT_ACTIVE: frozenset(
        (
            ExecutionState.ROLLOUT_REACQUIRE,
            ExecutionState.TO_INTERVENE,
            ExecutionState.EPISODE_PAUSED,
            ExecutionState.IDLE,
            ExecutionState.FAULT,
        )
    ),
    ExecutionState.TO_INTERVENE: frozenset((ExecutionState.INTERVENE_ACTIVE, ExecutionState.FAULT)),
    ExecutionState.INTERVENE_ACTIVE: frozenset(
        (
            ExecutionState.TO_ROLLOUT,
            ExecutionState.EPISODE_PAUSED,
            ExecutionState.IDLE,
            ExecutionState.FAULT,
        )
    ),
    ExecutionState.TO_ROLLOUT: frozenset((ExecutionState.ROLLOUT_REACQUIRE, ExecutionState.FAULT)),
    ExecutionState.EPISODE_PAUSED: frozenset(
        (
            ExecutionState.TO_ROLLOUT,
            ExecutionState.TO_INTERVENE,
            ExecutionState.IDLE,
            ExecutionState.FAULT,
        )
    ),
    ExecutionState.FAULT: frozenset(),
}


class ExecutionAuthority:
    """Thread-safe event inbox with one state-mutating supervisor.

    Producers may only post intent.  The first thread calling
    :meth:`bind_supervisor` becomes the sole thread allowed to change state.
    Rollout commands use :meth:`rollout_command_gate` as their final linearized
    epoch check, so a toggle and a command have an unambiguous ordering.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: deque[SupervisorEvent] = deque()
        self._state = ExecutionState.STARTING
        self._epoch = 0
        self._session_id: str | None = None
        self._fault: BaseException | None = None
        self._supervisor_ident: int | None = None

    @property
    def state(self) -> ExecutionState:
        with self._lock:
            return self._state

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    def snapshot(self) -> AuthoritySnapshot:
        with self._lock:
            return AuthoritySnapshot(
                state=self._state,
                epoch=self._epoch,
                session_id=self._session_id,
                fault=self._fault,
            )

    def bind_supervisor(self) -> None:
        ident = threading.get_ident()
        with self._lock:
            if self._supervisor_ident is None:
                self._supervisor_ident = ident
            elif self._supervisor_ident != ident:
                raise RuntimeError("execution authority may only be driven by its bound supervisor thread")

    def _require_supervisor_locked(self) -> None:
        if self._supervisor_ident != threading.get_ident():
            raise RuntimeError("only the bound supervisor thread may mutate execution authority")

    def post(
        self,
        kind: SupervisorEventKind,
        *,
        reason: str = "",
        error: BaseException | None = None,
    ) -> None:
        event_kind = kind
        with self._lock:
            if any(event.kind is event_kind for event in self._events):
                return
            event = SupervisorEvent(event_kind, reason=reason, error=error)
            if event_kind is SupervisorEventKind.FAULT:
                self._events.appendleft(event)
            elif event_kind in (SupervisorEventKind.PAUSE, SupervisorEventKind.STOP):
                superseded = {
                    SupervisorEventKind.ROLLOUT,
                    SupervisorEventKind.INTERVENE,
                }
                if event_kind is SupervisorEventKind.STOP:
                    superseded.add(SupervisorEventKind.PAUSE)
                self._events = deque(
                    pending for pending in self._events if pending.kind not in superseded
                )
                insert_at = 1 if self._events and self._events[0].kind is SupervisorEventKind.FAULT else 0
                self._events.insert(insert_at, event)
            else:
                self._events.append(event)

    def drain_events(self, *, max_events: int | None = None) -> tuple[SupervisorEvent, ...]:
        with self._lock:
            self._require_supervisor_locked()
            count = len(self._events) if max_events is None else min(len(self._events), max_events)
            return tuple(self._events.popleft() for _ in range(count))

    def new_session_id(self, epoch: int) -> str:
        return f"rollout-{int(epoch)}-{uuid4().hex}"

    def transition(
        self,
        state: ExecutionState,
        *,
        reason: str,
        bump_epoch: bool = False,
        session_id: str | None = None,
        fault: BaseException | None = None,
    ) -> AuthorityTransition:
        with self._lock:
            self._require_supervisor_locked()
            previous = self._state
            if state is previous:
                raise ValueError(f"duplicate authority transition to {state.value}")
            if state not in _ALLOWED_TRANSITIONS[previous]:
                raise ValueError(f"invalid authority transition {previous.value} -> {state.value}")
            if bump_epoch:
                self._epoch += 1
            self._state = state
            self._session_id = session_id
            if state is ExecutionState.FAULT:
                self._fault = fault or RuntimeError(reason)
            return AuthorityTransition(
                previous=previous,
                current=state,
                epoch=self._epoch,
                session_id=self._session_id,
                reason=reason,
            )

    def accepts_rollout_response(self, envelope: RolloutActionEnvelope) -> bool:
        with self._lock:
            return (
                self._state in ROLLOUT_STATES
                and envelope.epoch == self._epoch
                and envelope.session_id == self._session_id
            )

    @contextmanager
    def rollout_command_gate(self, envelope: RolloutActionEnvelope) -> Iterator[bool]:
        """Hold the authority lock across the final check and hardware command."""

        with self._lock:
            self._require_supervisor_locked()
            allowed = (
                self._state in ROLLOUT_STATES
                and envelope.epoch == self._epoch
                and envelope.session_id == self._session_id
                and not any(
                    event.kind
                    in {
                        SupervisorEventKind.INTERVENE,
                        SupervisorEventKind.PAUSE,
                        SupervisorEventKind.STOP,
                        SupervisorEventKind.FAULT,
                    }
                    for event in self._events
                )
            )
            yield allowed
