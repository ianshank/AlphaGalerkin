"""LambdaSchedulingGame — a RefinementGame over λ-window sample allocation.

State is packed into ``RefinementState.values`` as an ``(K, 3)`` array of
``[lo, hi, n]`` rows (one per active window). ``apply_action`` is a **pure,
deterministic** function of the state — no RNG, no MD, no I/O — advancing the
sufficient statistics ``σ_i = c_i / sqrt(n_i)``. All stochasticity lives in the
outer loop. This is what keeps node identity ≡ action sequence, so the MCTS
engine needs no chance nodes.

Two action families over a fixed index range ``[0, 2*max_windows)``:

* ``allocate(i)`` (``i < max_windows``) — add ``batch_samples`` to window ``i``.
  Monotone: it can only *decrease* the total standard error.
* ``split(i)`` (``i >= max_windows``) — split window ``i`` at its midpoint,
  dividing its samples between the children by ``sample_split_credit``. With the
  default credit 0.5 this **conserves** samples and therefore *increases* the
  total variance immediately (two children at n/2 each) — the non-monotone move
  whose payoff is only realised by subsequent allocation near the hardness peak.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.refinement.game import RefinementGame
from src.refinement.state import RefinementState

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.thermo.config import SchedulingParams
    from src.thermo.surrogate import VarianceSurrogate

_LO, _HI, _N = 0, 1, 2

# Converts a wall-clock cost (ns) into the error-reduction units of the reward so
# the reduction and cost terms are commensurate. Named, not a bare literal.
COST_TO_ERROR_SCALE = 1e-4

# Minimum samples a child window keeps after a split. A split conserves samples
# (each child gets ``n * sample_split_credit``); this floor only bites for a
# nearly-empty parent and guards against a zero-sample (infinite-variance) child.
MIN_SPLIT_CHILD_SAMPLES = 1.0


def total_stderr(windows: NDArray[np.float64], surrogate: VarianceSurrogate) -> float:
    """ΔG standard error ``sqrt(Σ c_i**2 / n_i)`` over the windows under a surrogate."""
    var = 0.0
    for lo, hi, n in windows:
        if n <= 0:
            continue
        c = surrogate.variance_coeff(float(lo), float(hi))
        var += (c * c) / float(n)
    return math.sqrt(var)


def variance_contributions(
    windows: NDArray[np.float64], surrogate: VarianceSurrogate
) -> NDArray[np.float64]:
    """Per-window variance contribution ``c_i**2 / n_i`` (the greedy signal)."""
    out = np.zeros(windows.shape[0], dtype=np.float64)
    for i, (lo, hi, n) in enumerate(windows):
        if n <= 0:
            out[i] = np.inf
            continue
        c = surrogate.variance_coeff(float(lo), float(hi))
        out[i] = (c * c) / float(n)
    return out


class LambdaSchedulingGame(RefinementGame):
    """Sequential λ-window sample-allocation game driven by a variance surrogate."""

    def __init__(self, params: SchedulingParams, surrogate: VarianceSurrogate) -> None:
        self.params = params
        self.surrogate = surrogate

    # ------------------------------------------------------------------ #
    # RefinementGame interface                                            #
    # ------------------------------------------------------------------ #

    @property
    def action_space_size(self) -> int:
        # allocate(0..max-1) + split(max..2*max-1)
        return 2 * self.params.max_windows

    def get_initial_state(self) -> RefinementState:
        k = self.params.n_initial_windows
        edges = np.linspace(0.0, 1.0, k + 1)
        windows = np.zeros((k, 3), dtype=np.float64)
        windows[:, _LO] = edges[:-1]
        windows[:, _HI] = edges[1:]
        windows[:, _N] = float(self.params.batch_samples)
        return self._state_from_windows(windows, step=0)

    def get_valid_actions(self, state: RefinementState) -> list[int]:
        if self.is_terminal(state):
            return []
        windows = self._windows(state)
        k = windows.shape[0]
        maxw = self.params.max_windows
        actions: list[int] = []
        if self._can_allocate(state):
            actions.extend(range(k))
        if self.params.allow_split and k < maxw:
            for i in range(k):
                width = windows[i, _HI] - windows[i, _LO]
                if width / 2.0 >= self.params.min_window_width:
                    actions.append(maxw + i)
        return sorted(actions)

    def apply_action(self, state: RefinementState, action: int) -> RefinementState:
        windows = self._windows(state).copy()
        maxw = self.params.max_windows

        if action < maxw:
            windows[action, _N] += float(self.params.batch_samples)
        else:
            i = action - maxw
            lo, hi, n = windows[i]
            mid = 0.5 * (lo + hi)
            child_n = max(float(n) * self.params.sample_split_credit, MIN_SPLIT_CHILD_SAMPLES)
            left = np.array([lo, mid, child_n], dtype=np.float64)
            right = np.array([mid, hi, child_n], dtype=np.float64)
            windows = np.vstack([windows[:i], left, right, windows[i + 1 :]])

        return self._state_from_windows(windows, step=state.step + 1)

    def is_terminal(self, state: RefinementState) -> bool:
        if state.step >= self.params.max_steps:
            return True
        if state.error_estimate <= self.params.error_tolerance:
            return True
        # Out of sample budget and splitting alone cannot reduce variance.
        return not self._can_allocate(state)

    def get_reward(self, state: RefinementState, prev_state: RefinementState) -> float:
        reduction = prev_state.error_estimate - state.error_estimate
        # A split adds a window (allocate never does), so the window-count delta
        # identifies the move type exactly — independent of the sample-accounting
        # side-effect a dof-delta would rely on.
        is_split = state.values.shape[0] > prev_state.values.shape[0]
        cost = self.params.split_cost_ns if is_split else self.params.batch_cost_ns
        return reduction - COST_TO_ERROR_SCALE * cost

    def get_winner(self, state: RefinementState) -> int:
        # Neutral (0) unless the schedule actually converged. Returning -1 for a
        # non-converged terminal would be ~1e3× the per-edge shaped reward and
        # would swamp the intermediate-reward signal MCTS optimises (the shaped
        # variance reduction), confounding the comparison. See the spec.
        return 1 if state.error_estimate <= self.params.error_tolerance else 0

    def to_tensor(self, state: RefinementState) -> NDArray[np.float32]:
        return state.values.astype(np.float32).reshape(-1)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _windows(self, state: RefinementState) -> NDArray[np.float64]:
        return state.values.astype(np.float64).reshape(-1, 3)

    def _samples_used(self, state: RefinementState) -> int:
        return int(round(float(state.values.astype(np.float64)[:, _N].sum())))

    def _can_allocate(self, state: RefinementState) -> bool:
        return self._samples_used(state) + self.params.batch_samples <= (self.params.sample_budget)

    def _state_from_windows(self, windows: NDArray[np.float64], step: int) -> RefinementState:
        err = total_stderr(windows, self.surrogate)
        indicators = variance_contributions(windows, self.surrogate)
        samples_used = int(round(float(windows[:, _N].sum())))
        return RefinementState(
            values=windows.astype(np.float32),
            indicators=indicators.astype(np.float32),
            error_estimate=float(err),
            dof=samples_used,
            step=step,
            budget_remaining=float(self.params.sample_budget - samples_used),
        )
