from __future__ import annotations

from collections import deque
import threading

import numpy as np

from .authority import RolloutActionEnvelope


class StreamActionBuffer:
    """Action chunk buffer for temporal smoothing."""

    def __init__(
        self,
        max_chunks: int = 10,
        decay_alpha: float = 0.25,
        state_dim: int = 14,
        smooth_method: str = "temporal",
    ) -> None:
        self.lock = threading.Lock()
        self.cur_chunk = deque()
        self._cur_envelopes: deque[RolloutActionEnvelope | None] = deque()
        self.k = 0
        self.last_action: np.ndarray | None = None
        self._active_epoch: int | None = None

    def _integrate_locked(
        self,
        actions_chunk: np.ndarray,
        *,
        max_k: int,
        min_m: int,
        blend: bool,
    ) -> bool:
        if actions_chunk is None or len(actions_chunk) == 0:
            return False
        max_k = max(0, int(max_k))
        min_m = max(1, int(min_m))
        drop_n = min(self.k, max_k)
        if drop_n >= len(actions_chunk):
            return False
        new_chunk = [np.asarray(a, dtype=float).copy() for a in actions_chunk[drop_n:]]
        if not blend:
            self.cur_chunk = deque(new_chunk, maxlen=None)
            self.last_action = None
            self.k = 0
            return True

        if len(self.cur_chunk) == 0 and self.last_action is not None:
            old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
            self.last_action = None
        else:
            old_list = list(self.cur_chunk)
            if len(old_list) > 0 and len(old_list) < min_m:
                tail = np.asarray(old_list[-1], dtype=float).copy()
                old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
            elif len(old_list) == 0:
                self.cur_chunk = deque(new_chunk, maxlen=None)
                self.k = 0
                return True

        new_list = list(new_chunk)
        overlap_len = min(len(old_list), len(new_list))
        if overlap_len <= 0:
            self.cur_chunk = deque(new_list, maxlen=None)
            self.k = 0
            return True

        if len(old_list) > len(new_list):
            old_list = old_list[: len(new_list)]
            overlap_len = len(new_list)

        if overlap_len == 1:
            w_old = np.array([1.0], dtype=float)
        else:
            w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
        w_new = 1.0 - w_old

        smoothed = [
            (w_old[i] * np.asarray(old_list[i], dtype=float))
            + (w_new[i] * np.asarray(new_list[i], dtype=float))
            for i in range(overlap_len)
        ]
        combined = smoothed + new_list[overlap_len:]
        self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
        self.k = 0
        return True

    def activate_epoch(self, epoch: int, *, clear: bool = True) -> None:
        """Fence the buffer to one authority epoch.

        Clearing is the safe default and deliberately resets ``last_action`` so
        the first action after reacquisition is never blended with an old epoch.
        """

        with self.lock:
            self._active_epoch = int(epoch)
            if clear:
                self._clear_locked()

    def integrate_envelope(
        self,
        envelope: RolloutActionEnvelope,
        *,
        max_k: int,
        min_m: int = 8,
        blend: bool = True,
    ) -> bool:
        """Integrate a response only if its epoch is still active."""

        with self.lock:
            if self._active_epoch is None or int(envelope.epoch) != self._active_epoch:
                return False
            integrated = self._integrate_locked(
                envelope.actions,
                max_k=max_k,
                min_m=min_m,
                blend=blend,
            )
            if integrated:
                self._cur_envelopes = deque((envelope for _ in self.cur_chunk), maxlen=None)
            return integrated

    def _pop_locked(self) -> tuple[np.ndarray, RolloutActionEnvelope | None] | None:
        if len(self.cur_chunk) == 0:
            return None
        if len(self.cur_chunk) == 1:
            self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
        act = np.asarray(self.cur_chunk.popleft(), dtype=float)
        envelope = self._cur_envelopes.popleft() if self._cur_envelopes else None
        self.k += 1
        return act, envelope

    def pop_next_enveloped_action(
        self,
        *,
        expected_epoch: int,
    ) -> tuple[np.ndarray, RolloutActionEnvelope] | None:
        """Pop with a second epoch check immediately before command dispatch."""

        with self.lock:
            if self._active_epoch != int(expected_epoch):
                return None
            popped = self._pop_locked()
            if popped is None:
                return None
            action, envelope = popped
            if envelope is None or int(envelope.epoch) != int(expected_epoch):
                self._clear_locked()
                return None
            return action, envelope

    def has_any(self, *, expected_epoch: int | None = None) -> bool:
        with self.lock:
            if expected_epoch is not None and self._active_epoch != int(expected_epoch):
                return False
            return len(self.cur_chunk) > 0

    def _clear_locked(self) -> None:
        self.cur_chunk.clear()
        self._cur_envelopes.clear()
        self.last_action = None
        self.k = 0
