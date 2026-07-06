"""L-shaped Poisson AMR framed as an MCTS refinement game.

This module frames adaptive mesh refinement on the standard L-shaped
Poisson benchmark (reentrant-corner singularity ``u = r^(2/3) sin(2θ/3)``)
as a sequential decision game so the AlphaGalerkin ``MCTS`` engine can drive
the refinement *policy*. Each action refines one candidate element (the
tensor-product x/y edge pair); :meth:`LShapeAMRGame.apply_action` performs a
**real** masked finite-difference solve (injected via ``solve_fn``) and
reports the true L2 error against the exact solution together with the
active (in-domain) degree-of-freedom count.

Because the solve, the residual error estimator, the geometry mask and the
DOF accounting are all supplied by the caller, the *only* thing that differs
between this MCTS arm and the classical Dörfler-marking arm in
:mod:`src.research.lshape_amr_compare` is the marking policy — an honest,
controlled head-to-head. See ``specs/lshape_amr_compare.spec.md``.

Design notes
------------
* The tensor-product grid ``(xs, ys)`` lives on the game instance (mirroring
  :class:`~src.pde.games.mesh_refinement.MeshRefinementGame`, which holds its
  ``Mesh``); :meth:`clone` deep-copies it so sibling MCTS simulations never
  interfere.
* MCTS takes a leaf's value from ``evaluator.evaluate(state).value``. The
  bundled :class:`EncodedValueEvaluator` reads a normalised
  *lower-error-per-DOF-is-better* scalar that :meth:`to_tensor` writes into
  the state encoding, so MCTS performs genuine multi-step lookahead over an
  efficiency objective (no trained network required, ``src/mcts`` untouched).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch
from numpy.typing import NDArray

from src.mcts.evaluator import EvaluationResult
from src.pde.game import GamePhase, PDEGame, PDEResult, PDEState

if TYPE_CHECKING:
    from src.pde.config import PDEGameConfig
    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# Named numerical constants (no magic numbers)                                 #
# --------------------------------------------------------------------------- #

# Steepness of the tanh squashing used when encoding the leaf value. Larger
# values saturate faster; surfaced as a constructor arg (``value_scale``).
DEFAULT_VALUE_SCALE: float = 4.0

# Floor applied to error-per-DOF before the log ratio, avoiding log(0)/inf.
EPD_FLOOR: float = 1e-15

# Minimum spacing below which two grid coordinates are treated as identical,
# so a re-refined edge does not insert a duplicate node.
DEFAULT_MERGE_TOL: float = 1e-12


@dataclass
class GridSolveResult:
    """Outcome of one masked tensor-product solve.

    Attributes:
        solution: Solution values over the full grid, shape ``(n_nodes,)``.
        grid: Node coordinates, shape ``(n_nodes, 2)``.
        l2_error: RMS L2 error against the exact solution over in-domain nodes.
        n_dof: Active (in-domain) node count — the matched DOF measure.
        indicators: Per-element residual indicators, shape
            ``(n_elem_x, n_elem_y)``; out-of-domain elements are zero.

    """

    solution: NDArray[np.float64]
    grid: NDArray[np.float64]
    l2_error: float
    n_dof: int
    indicators: NDArray[np.float64]


# A solve function maps a tensor-product grid ``(xs, ys)`` to a
# :class:`GridSolveResult`. Injected so this game stays independent of the
# concrete finite-difference backend (which lives in ``src.research``).
GridSolveFn = Callable[[NDArray[np.float64], NDArray[np.float64]], GridSolveResult]


class EncodedValueEvaluator:
    """MCTS evaluator: uniform policy prior + leaf value from the encoding.

    Structurally satisfies the ``src.mcts.evaluator.Evaluator`` Protocol
    (duck-typed — no subclassing, ``src/mcts`` untouched). The policy prior is
    uniform over legal actions; the leaf *value* is read from the first element
    of the state encoding, which :meth:`LShapeAMRGame.to_tensor` sets to a
    normalised *lower-error-per-DOF-is-better* scalar in ``[-1, 1]``.
    """

    def __init__(self, n_actions: int) -> None:
        """Initialise the evaluator.

        Args:
            n_actions: Size of the (fixed) action space.

        """
        if n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {n_actions}")
        self.n_actions = n_actions

    def evaluate(
        self,
        state: NDArray[np.float32],
        legal_actions: list[int],
    ) -> EvaluationResult:
        """Return a uniform policy prior and the encoded leaf value.

        Args:
            state: State encoding; ``state.reshape(-1)[0]`` carries the value.
            legal_actions: Legal action indices at this state.

        Returns:
            Policy over the full action space and a clamped scalar value.

        """
        policy = np.zeros(self.n_actions, dtype=np.float32)
        if legal_actions:
            uniform = 1.0 / len(legal_actions)
            for action in legal_actions:
                policy[action] = uniform
        flat = np.asarray(state, dtype=np.float32).reshape(-1)
        value = float(flat[0]) if flat.size else 0.0
        value = max(-1.0, min(1.0, value))
        return EvaluationResult(policy=policy, value=value)

    def evaluate_batch(
        self,
        states: list[NDArray[np.float32]],
        legal_actions_batch: list[list[int]],
    ) -> list[EvaluationResult]:
        """Evaluate a batch by delegating to :meth:`evaluate`."""
        return [self.evaluate(s, la) for s, la in zip(states, legal_actions_batch, strict=False)]


class LShapeAMRGame(PDEGame):
    """Adaptive mesh refinement on the L-shaped Poisson benchmark as a game.

    One action refines one candidate element (ranked by residual indicator) by
    bisecting its x- and y-edges; ``apply_action`` re-solves with the injected
    masked solver and reports the true L2 error and active-DOF count.
    """

    name = "lshape_amr"
    description = "L-shaped Poisson adaptive mesh refinement (MCTS policy)"

    def __init__(
        self,
        pde_operator: PDEOperator,
        config: PDEGameConfig,
        *,
        solve_fn: GridSolveFn,
        initial_side: int,
        n_candidate_elements: int,
        value_scale: float = DEFAULT_VALUE_SCALE,
        merge_tol: float = DEFAULT_MERGE_TOL,
    ) -> None:
        """Initialise the game.

        Args:
            pde_operator: The (L-shaped Poisson) operator; supplies the domain
                bounding box via ``domain_min`` / ``domain_max``.
            config: Standard :class:`~src.pde.config.PDEGameConfig` supplying
                budget / tolerance / winner thresholds.
            solve_fn: Masked tensor-product solver (injected).
            initial_side: Number of *elements* per axis on the coarse grid
                (grid has ``initial_side + 1`` nodes per axis).
            n_candidate_elements: Fixed action-space size — the number of
                top-ranked (by indicator) refinable elements MCTS may choose
                between at each step.
            value_scale: tanh steepness for the encoded leaf value.
            merge_tol: Minimum node spacing (duplicate-insertion guard).

        Raises:
            ValueError: If ``initial_side`` or ``n_candidate_elements`` < 1.

        """
        super().__init__(pde_operator, config)
        if initial_side < 1:
            raise ValueError(f"initial_side must be >= 1, got {initial_side}")
        if n_candidate_elements < 1:
            raise ValueError(f"n_candidate_elements must be >= 1, got {n_candidate_elements}")
        self._solve_fn = solve_fn
        self._initial_side = initial_side
        self._n_candidate_elements = n_candidate_elements
        self._value_scale = value_scale
        self._merge_tol = merge_tol

        lo = np.asarray(pde_operator.domain_min, dtype=np.float64)
        hi = np.asarray(pde_operator.domain_max, dtype=np.float64)
        self._x_lo, self._x_hi = float(lo[0]), float(hi[0])
        self._y_lo, self._y_hi = float(lo[1]), float(hi[1])

        # Per-episode grid state (mutated by apply_action; deep-copied by clone).
        self._xs: NDArray[np.float64] = np.linspace(
            self._x_lo, self._x_hi, initial_side + 1, dtype=np.float64
        )
        self._ys: NDArray[np.float64] = np.linspace(
            self._y_lo, self._y_hi, initial_side + 1, dtype=np.float64
        )
        self._last_indicators: NDArray[np.float64] = np.zeros((1, 1), dtype=np.float64)
        self._initial_epd: float | None = None

    # ------------------------------------------------------------------ #
    # Required PDEGame properties                                         #
    # ------------------------------------------------------------------ #

    @property
    def action_space_size(self) -> int:
        """Fixed number of candidate elements MCTS may refine."""
        return self._n_candidate_elements

    @property
    def state_channels(self) -> int:
        """Single channel: the encoded leaf value scalar."""
        return 1

    # ------------------------------------------------------------------ #
    # Grid <-> state helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _error_per_dof(error: float, dof: int) -> float:
        """Error-per-DOF efficiency metric with a positive floor."""
        return max(error, EPD_FLOOR) / max(dof, 1)

    def _make_state(self, solve: GridSolveResult, step: int, history: list[int]) -> PDEState:
        """Build a :class:`PDEState` from a solve result."""
        residuals = self._last_indicators.reshape(-1).astype(np.float32)
        # residuals array is per-element; pad/truncate to n_nodes for the
        # dataclass contract (only error_estimate/dof drive the game logic).
        n_nodes = solve.grid.shape[0]
        if residuals.size < n_nodes:
            residuals = np.concatenate(
                [residuals, np.zeros(n_nodes - residuals.size, dtype=np.float32)]
            )
        else:
            residuals = residuals[:n_nodes]
        return PDEState(
            coords=solve.grid.astype(np.float32),
            solution=solve.solution.astype(np.float32),
            residuals=residuals,
            error_estimate=float(solve.l2_error),
            dof=int(solve.n_dof),
            step=step,
            budget_remaining=float(max(self.config.max_dof - solve.n_dof, 0)),
            phase=GamePhase.INITIAL if step == 0 else GamePhase.REFINING,
            history=list(history),
        )

    def _solve_current(self) -> GridSolveResult:
        """Solve on the current grid and cache the indicators."""
        result = self._solve_fn(self._xs, self._ys)
        self._last_indicators = np.asarray(result.indicators, dtype=np.float64)
        return result

    # ------------------------------------------------------------------ #
    # Game lifecycle                                                      #
    # ------------------------------------------------------------------ #

    def get_initial_state(self) -> PDEState:
        """Solve the coarse grid and record the initial error-per-DOF."""
        self._xs = np.linspace(self._x_lo, self._x_hi, self._initial_side + 1, dtype=np.float64)
        self._ys = np.linspace(self._y_lo, self._y_hi, self._initial_side + 1, dtype=np.float64)
        solve = self._solve_current()
        self._initial_epd = self._error_per_dof(solve.l2_error, solve.n_dof)
        logger.debug(
            "lshape_amr_initial_state",
            n_dof=solve.n_dof,
            l2_error=solve.l2_error,
            initial_epd=self._initial_epd,
        )
        return self._make_state(solve, step=0, history=[])

    def _ranked_elements(self) -> list[tuple[int, int]]:
        """Return refinable elements sorted by indicator, descending.

        Only in-domain elements (strictly positive indicator) whose x- or y-edge
        is still wider than the bisection tolerance are candidates. Excluding
        edges already at ``merge_tol`` avoids no-op refinements that would burn a
        game step (and could otherwise re-select the same saturated element).
        """
        ind = self._last_indicators
        flat = ind.reshape(-1)
        order = np.argsort(flat)[::-1]
        ny_elem = ind.shape[1]
        min_width = 2.0 * self._merge_tol
        ranked: list[tuple[int, int]] = []
        for flat_idx in order:
            if flat[flat_idx] <= 0.0:
                break
            ix = int(flat_idx) // ny_elem
            iy = int(flat_idx) % ny_elem
            can_bisect_x = (self._xs[ix + 1] - self._xs[ix]) >= min_width
            can_bisect_y = (self._ys[iy + 1] - self._ys[iy]) >= min_width
            if can_bisect_x or can_bisect_y:
                ranked.append((ix, iy))
        return ranked

    def get_valid_actions(self, state: PDEState) -> list[int]:
        """Legal actions: the top-``k`` refinable elements by indicator."""
        n_candidates = min(len(self._ranked_elements()), self._n_candidate_elements)
        return list(range(n_candidates))

    def get_action_mask(self, state: PDEState) -> NDArray[np.bool_]:
        """Boolean mask of legal actions over the fixed action space."""
        mask = np.zeros(self._n_candidate_elements, dtype=bool)
        valid = self.get_valid_actions(state)
        if valid:
            mask[valid] = True
        return mask

    @staticmethod
    def _bisect_edge(axis: NDArray[np.float64], edge: int, tol: float) -> NDArray[np.float64]:
        """Insert the midpoint of element ``edge`` into a 1-D grid axis."""
        if edge < 0 or edge >= len(axis) - 1:
            return axis
        mid = 0.5 * (axis[edge] + axis[edge + 1])
        if np.any(np.abs(axis - mid) < tol):
            return axis
        return np.sort(np.append(axis, mid))

    def apply_action(self, state: PDEState, action: int) -> PDEState:
        """Refine the ``action``-th ranked element and re-solve.

        Args:
            state: Current state (kept for interface parity; the authoritative
                grid lives on the game instance).
            action: Index into the ranked candidate list.

        Returns:
            The new :class:`PDEState` after refinement and re-solve.

        Raises:
            ValueError: If ``action`` is not currently legal.

        """
        ranked = self._ranked_elements()
        n_candidates = min(len(ranked), self._n_candidate_elements)
        if action < 0 or action >= n_candidates:
            raise ValueError(f"Illegal action {action}; {n_candidates} candidates available")
        ix, iy = ranked[action]
        self._xs = self._bisect_edge(self._xs, ix, self._merge_tol)
        self._ys = self._bisect_edge(self._ys, iy, self._merge_tol)
        solve = self._solve_current()
        return self._make_state(solve, step=state.step + 1, history=[*state.history, action])

    def get_reward(self, state: PDEState, prev_state: PDEState) -> float:
        """Reward = reduction in error-per-DOF from ``prev_state`` to ``state``."""
        epd_prev = self._error_per_dof(prev_state.error_estimate, prev_state.dof)
        epd_cur = self._error_per_dof(state.error_estimate, state.dof)
        return float(self.config.reward_per_error_reduction * (epd_prev - epd_cur))

    def is_terminal(self, state: PDEState) -> bool:
        """Terminal on DOF budget, step cap, tolerance, or no legal actions."""
        if state.dof >= self.config.max_dof:
            return True
        if state.step >= self.config.max_steps:
            return True
        if state.error_estimate < self.config.error_tolerance:
            return True
        return not self.get_valid_actions(state)

    def _termination_reason(self, state: PDEState) -> str:
        """Classify *why* an episode ended, mirroring :meth:`is_terminal`.

        Distinguishes the terminal causes (converged / max DOF / max steps /
        no legal actions) rather than collapsing them into a single
        "budget_exhausted", matching the fidelity of the sibling
        mesh_refinement / basis_selection games. Returns "running" for a
        non-terminal state.
        """
        if state.error_estimate < self.config.error_tolerance:
            return "converged"
        if state.dof >= self.config.max_dof:
            return "max_dof"
        if state.step >= self.config.max_steps:
            return "max_steps"
        if not self.get_valid_actions(state):
            return "no_legal_actions"
        return "running"

    def get_result(self, state: PDEState, error_history: list[float]) -> PDEResult:
        """Assemble a :class:`PDEResult` from the terminal state and history."""
        initial_error = error_history[0] if error_history else state.error_estimate
        best_error = min(error_history) if error_history else state.error_estimate
        avg_error = float(np.mean(error_history)) if error_history else state.error_estimate
        reduction = initial_error - state.error_estimate
        return PDEResult(
            final_error=state.error_estimate,
            final_dof=state.dof,
            n_steps=state.step,
            converged=state.error_estimate < self.config.error_tolerance,
            l2_error=state.error_estimate,
            h1_error=state.error_estimate,
            linf_error=state.error_estimate,
            residual_norm=float(np.sqrt(np.mean(state.residuals**2)))
            if state.residuals.size
            else 0.0,
            error_reduction_rate=reduction / max(state.step, 1),
            dof_efficiency=reduction / max(state.dof, 1),
            compute_efficiency=reduction / max(state.step, 1),
            initial_error=initial_error,
            best_error=best_error,
            average_error=avg_error,
            error_history=list(error_history),
            termination_reason=self._termination_reason(state),
            budget_used=float(state.dof),
        )

    def compute_exact_error(self, state: PDEState) -> dict[str, float]:
        """Return the error metrics carried on the state."""
        return {
            "l2": float(state.error_estimate),
            "h1": float(state.error_estimate),
            "linf": float(state.error_estimate),
            "residual": float(np.sqrt(np.mean(state.residuals**2)))
            if state.residuals.size
            else 0.0,
        }

    def to_tensor(self, state: PDEState) -> torch.Tensor:
        """Encode the normalised *lower-error-per-DOF-is-better* leaf value.

        The scalar is ``tanh(value_scale * log(epd0 / epd))`` clamped to
        ``[-1, 1]`` (positive when the current error-per-DOF beats the initial
        grid). Shape ``(1, 1, 1)`` — read by :class:`EncodedValueEvaluator`.
        """
        epd0 = (
            self._initial_epd
            if self._initial_epd is not None
            else self._error_per_dof(state.error_estimate, state.dof)
        )
        epd = self._error_per_dof(state.error_estimate, state.dof)
        ratio = max(epd0, EPD_FLOOR) / max(epd, EPD_FLOOR)
        value = float(np.tanh(self._value_scale * np.log(ratio)))
        return torch.tensor([[[value]]], dtype=torch.float32)

    def clone(self) -> LShapeAMRGame:
        """Deep-copy the per-episode grid so MCTS siblings stay isolated."""
        cloned = LShapeAMRGame(
            self.pde_operator,
            self.config,
            solve_fn=self._solve_fn,
            initial_side=self._initial_side,
            n_candidate_elements=self._n_candidate_elements,
            value_scale=self._value_scale,
            merge_tol=self._merge_tol,
        )
        cloned._xs = self._xs.copy()
        cloned._ys = self._ys.copy()
        cloned._last_indicators = self._last_indicators.copy()
        cloned._initial_epd = self._initial_epd
        return cloned
