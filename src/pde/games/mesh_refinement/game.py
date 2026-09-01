"""MCTS-facing mesh refinement game built on the pure ``Mesh`` data structure.

This module implements a PDEGame where:
- State: Current mesh + solution quality indicators
- Actions: Refine specific elements (h or p refinement)
- Reward: Error reduction per DOF added
- Terminal: Error < tolerance or DOF budget exhausted

MCTS can look ahead multiple refinement steps to find optimal
refinement sequences, outperforming single-step error indicators.

See ``mesh.py`` in this package for the domain-free ``Mesh``/``MeshElement``/
``ActionKind`` data structure this game drives.
"""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from torch import Tensor

from src.pde.config import MeshRefinementConfig, PDEGameConfig
from src.pde.game import GamePhase, PDEGame, PDEState
from src.pde.games.mesh_refinement.mesh import ActionKind, Mesh, MeshElement
from src.pde.reward import log_reward

if TYPE_CHECKING:
    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)


class MeshRefinementGame(PDEGame):
    """Mesh refinement game for adaptive methods.

    The agent decides which elements to refine and how,
    building an optimal mesh for the given PDE.
    """

    name = "mesh_refinement"
    description = "Adaptive mesh refinement game"

    def __init__(
        self,
        pde_operator: PDEOperator,
        config: PDEGameConfig,
    ) -> None:
        """Initialize mesh refinement game.

        Args:
            pde_operator: PDE operator to solve.
            config: Game configuration.

        """
        super().__init__(pde_operator, config)

        self.mesh_config = config.mesh_config or MeshRefinementConfig(name="default_mesh")

        # Initialize mesh
        self.mesh = Mesh(
            domain_min=np.array(pde_operator.config.domain_min, dtype=np.float32),
            domain_max=np.array(pde_operator.config.domain_max, dtype=np.float32),
            initial_resolution=self.mesh_config.initial_resolution,
            hp_switchover_level=self.mesh_config.hp_switchover_level,
        )

        # Action space: refine element i with strategy s
        # For simplicity, use h-refinement only
        self._refinement_strategy = self.mesh_config.refinement_strategy

        # Interpolator cache for ``_interpolate_solution``.
        #
        # ``LinearNDInterpolator(coords, …).__init__`` runs a Delaunay
        # triangulation on the input points (Qhull); this dominates the
        # interpolation cost. During an MCTS expansion the *same*
        # ``old_state`` is fed to ``apply_action`` for many candidate
        # actions, so caching by object identity (``is``) avoids the
        # repeated Qhull pass without any change to the public state
        # type.  The cache stores at most one (state, interpolator) pair
        # per game instance and is naturally invalidated whenever a new
        # state object is observed.
        self._cached_interp_state: PDEState | None = None
        self._cached_interp_linear: Any | None = None

    def clone(self) -> PDEGame:
        """MCTS-safe clone with an independent mesh.

        ``apply_action`` mutates ``self.mesh`` in-place (both refine and
        coarsen edit the element tree), so sibling MCTS simulations must
        not share it. The expensive immutables — ``pde_operator``,
        ``config``, ``mesh_config`` — are shared by reference; only the
        mutable mesh tree is deep-copied.
        """
        cls = type(self)
        cloned = cls.__new__(cls)
        cloned.pde_operator = self.pde_operator
        cloned.config = self.config
        cloned.mesh_config = self.mesh_config
        cloned._refinement_strategy = self._refinement_strategy
        cloned.mesh = copy.deepcopy(self.mesh)
        # Each clone starts with a fresh interpolator cache: the cloned game
        # will see different ``old_state`` objects than the source.
        cloned._cached_interp_state = None
        cloned._cached_interp_linear = None
        logger.debug(
            "mesh_game_cloned",
            n_elements=cloned.mesh.n_elements,
            n_leaves=len(cloned.mesh.leaf_elements),
        )
        return cloned

    @property
    def _coarsen_enabled(self) -> bool:
        """Whether the coarsen half of the action space is active."""
        return self.mesh_config.allow_coarsening

    @property
    def _refine_slot_count(self) -> int:
        """Effective width of the refine half of the action space.

        This is the *single source of truth* for the refine/coarsen
        partition point. ``action_space_size``, ``_decode_action``,
        ``get_valid_actions``, and ``get_action_mask`` all derive their
        slot count from this property so the partition stays consistent
        when ``max_elements`` is smaller than ``n_candidate_elements``
        (e.g. low-dim / low-level configs).
        """
        dim = self.mesh.dim
        max_elements = (
            self.mesh_config.initial_resolution**dim
            * (2**dim) ** self.mesh_config.max_refinement_level
        )
        return min(max_elements, self.mesh_config.n_candidate_elements)

    @property
    def action_space_size(self) -> int:
        """Number of possible actions.

        When ``allow_coarsening`` is set the action space is partitioned
        into two equal halves of width :attr:`_refine_slot_count`: the
        low half refines, the high half coarsens.
        """
        n = self._refine_slot_count
        return n * 2 if self._coarsen_enabled else n

    @property
    def state_channels(self) -> int:
        """Neural network input channels."""
        return 5  # solution, residual, error, refinement level, size

    def get_initial_state(self) -> PDEState:
        """Create initial state with coarse mesh.

        Returns:
            Initial PDEState.

        """
        # Reset mesh to initial state
        self.mesh = Mesh(
            domain_min=np.array(self.pde_operator.config.domain_min, dtype=np.float32),
            domain_max=np.array(self.pde_operator.config.domain_max, dtype=np.float32),
            initial_resolution=self.mesh_config.initial_resolution,
            hp_switchover_level=self.mesh_config.hp_switchover_level,
        )

        # Get element centers for collocation
        coords = self.mesh.get_element_centers()
        n_points = len(coords)

        # Initial solution (zero)
        solution = np.zeros(n_points, dtype=np.float32)

        # Compute residual
        source = self.pde_operator.source_term(coords)
        if isinstance(source, Tensor):
            source = source.numpy()
        residuals = -source.astype(np.float32)

        # Initial error
        error = float(np.sqrt(np.mean(residuals**2)))

        # Mesh info
        mesh_levels = np.array([e.level for e in self.mesh.leaf_elements], dtype=np.int32)

        return PDEState(
            coords=coords,
            solution=solution,
            residuals=residuals,
            mesh_levels=mesh_levels,
            error_estimate=error,
            dof=self.mesh.n_dof,
            step=0,
            budget_remaining=self.config.computational_budget,
            phase=GamePhase.INITIAL,
            history=[],
        )

    def _decode_action(self, action: int) -> tuple[ActionKind, int]:
        """Split a flat action into (kind, leaf_index).

        When coarsening is enabled the action space is::

            [0, n)          -> REFINE leaf_elements[action]
            [n, 2n)         -> COARSEN leaf_elements[action - n]

        where ``n = _refine_slot_count`` (the effective slot width —
        matches the bound used by ``action_space_size``, never the raw
        ``n_candidate_elements`` config value). When coarsening is
        disabled the upper half is absent and every action decodes as a
        refinement.

        Args:
            action: Flat action index.

        Returns:
            Pair of the decoded action kind and the leaf-order index.

        Raises:
            ValueError: If the action is out of range for the current
                action space.

        """
        slots = self._refine_slot_count
        if action < 0 or action >= self.action_space_size:
            raise ValueError(f"Invalid action: {action} not in [0, {self.action_space_size})")
        if not self._coarsen_enabled or action < slots:
            return ActionKind.REFINE, action
        return ActionKind.COARSEN, action - slots

    def _refine_eligible(self, element: MeshElement) -> bool:
        """Whether a leaf element can be refined under current config limits."""
        if element.level >= self.mesh_config.max_refinement_level:
            return False
        if element.size < self.mesh_config.min_element_size:
            return False
        if element.polynomial_degree >= self.mesh_config.max_polynomial_degree:
            return False
        return True

    def get_valid_actions(self, state: PDEState) -> list[int]:
        """Get valid refinement (and optionally coarsening) actions.

        The partition point is driven by :attr:`_refine_slot_count` so
        emitted indices are always within ``action_space_size``.

        Coarsen actions are deduplicated by parent: every child in a
        coarsenable sibling group triggers the same parent collapse, so
        exposing one action per parent (rather than one per child)
        keeps the MCTS branching factor minimal without losing
        expressivity.

        Args:
            state: Current state.

        Returns:
            List of valid flat action indices. When ``allow_coarsening``
            is enabled, indices below :attr:`_refine_slot_count` are
            refine actions and indices above are coarsen actions.

        """
        slots = self._refine_slot_count
        leaves = self.mesh.leaf_elements

        refine_actions: list[int] = []
        for i, element in enumerate(leaves):
            if i >= slots:
                break
            if self._refine_eligible(element):
                refine_actions.append(i)

        if not self._coarsen_enabled:
            return refine_actions

        coarsen_actions: list[int] = []
        seen_parents: set[int] = set()
        for i, element in enumerate(leaves):
            if i >= slots:
                break
            if element.parent is None or element.parent in seen_parents:
                continue
            if self.mesh.can_coarsen_element(element.index):
                coarsen_actions.append(slots + i)
                seen_parents.add(element.parent)

        return refine_actions + coarsen_actions

    def get_action_mask(self, state: PDEState) -> NDArray[np.bool_]:
        """Get boolean mask for valid actions.

        Args:
            state: Current state.

        Returns:
            Boolean mask of length ``action_space_size``.

        """
        mask = np.zeros(self.action_space_size, dtype=bool)
        valid = self.get_valid_actions(state)
        for idx in valid:
            if idx < self.action_space_size:
                mask[idx] = True
        return mask

    def apply_action(self, state: PDEState, action: int) -> PDEState:
        """Apply a refinement or coarsening action.

        Args:
            state: Current state.
            action: Flat action index; see :meth:`_decode_action`.

        Returns:
            New state after the action.

        """
        kind, leaf_idx = self._decode_action(action)

        leaf_elements = self.mesh.leaf_elements
        if leaf_idx >= len(leaf_elements):
            raise ValueError(f"Invalid action: {action} (leaf {leaf_idx} missing)")

        element = leaf_elements[leaf_idx]

        if kind is ActionKind.REFINE:
            self.mesh.refine_element(
                element.index,
                self._refinement_strategy,
            )
        else:
            # ActionKind.COARSEN: raise a clean error if the action space
            # happened to expose a slot whose leaf is no longer coarsenable
            # (e.g. a sibling was refined further between mask evaluation
            # and dispatch). This preserves the "invalid action" contract.
            if not self.mesh.can_coarsen_element(element.index):
                raise ValueError(
                    f"Invalid coarsen action {action}: element {element.index} "
                    "is not in a fully-leaf sibling group."
                )
            self.mesh.coarsen_element(element.index)

        # Rebuild state
        coords = self.mesh.get_element_centers()
        n_points = len(coords)

        # Solve on new mesh (simplified: interpolate old solution)
        if len(state.solution) == n_points:
            solution = state.solution.copy()
        else:
            # Interpolate from old mesh
            solution = self._interpolate_solution(state, coords)

        # Compute residual
        residual_result = self.pde_operator.residual(
            torch.from_numpy(solution),
            torch.from_numpy(coords),
            compute_derivatives=False,
        )
        if isinstance(residual_result.values, Tensor):
            residuals = residual_result.values.numpy().astype(np.float32)
        else:
            residuals = residual_result.values.astype(np.float32)

        # Compute error
        error = float(np.sqrt(np.mean(residuals**2)))

        # Mesh info
        mesh_levels = np.array([e.level for e in self.mesh.leaf_elements], dtype=np.int32)

        # Cost is config-driven (``cost_per_dof``), mirroring the reward
        # path's ``cost = self.config.cost_per_dof * dof_added`` in
        # ``get_reward`` below, instead of the flat unit cost of 1 this used
        # previously (which was decoupled from ``cost_per_dof`` entirely).
        #
        # The direction of the change is dimension-dependent, not a fixed
        # ratio: ``n_dof`` sums ``(p+1)**dim`` per element, so one 2D h-refine
        # at p=1 adds 12 DOF -> cost 0.12 (cheaper than the old 1.0), while a
        # 3D h-refine at p=3 adds 448 DOF -> cost 4.48 (more expensive). Budget
        # exhaustion is unreachable at every shipped config either way, since
        # ``max_steps`` (<=100) caps the episode long before
        # ``computational_budget`` (>=1e4) is spent.
        #
        # DOF can decrease under coarsening, so ``dof_added`` -- and thus
        # ``cost`` -- may be negative, matching the reward path's treatment of
        # coarsening as a partial refund. Note this gives up the strict
        # monotonicity the old ``- 1`` had: a refine/coarsen oscillation can
        # hold ``budget_remaining`` steady, so it is no longer a liveness
        # bound. ``max_steps`` remains the actual episode bound.
        new_dof = self.mesh.n_dof
        dof_added = new_dof - state.dof
        cost = self.config.cost_per_dof * dof_added

        new_state = PDEState(
            coords=coords,
            solution=solution,
            residuals=residuals,
            mesh_levels=mesh_levels,
            error_estimate=error,
            dof=new_dof,
            step=state.step + 1,
            budget_remaining=state.budget_remaining - cost,
            phase=state.phase,
            history=[*state.history, action],
        )

        # Update phase
        if new_state.error_estimate < self.config.error_tolerance:
            new_state.phase = GamePhase.CONVERGED
        elif new_state.budget_remaining <= 0 or new_state.dof > self.config.max_dof:
            new_state.phase = GamePhase.BUDGET_EXHAUSTED

        return new_state

    def _interpolate_solution(
        self,
        old_state: PDEState,
        new_coords: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Interpolate the old solution onto the refined mesh.

        Uses piecewise-linear interpolation (barycentric in 2D+, linear in
        1D) over the previous mesh's collocation points. Points that fall
        outside the convex hull of the old coords fall back to nearest-
        neighbor so the returned array is always well-defined.

        A proper Galerkin :math:`L^2`-projection onto the refined trial
        space would require assembling the refined mass matrix; that
        refinement is tracked as a future deliverable alongside the FEM
        baseline (see ``docs/doe_genesis/mdp_specification.md § 4``).

        Args:
            old_state: Previous state.
            new_coords: New coordinate points; shape ``(n_new, dim)``.

        Returns:
            Interpolated solution at ``new_coords`` with ``np.float32`` dtype.

        """
        old_coords = np.asarray(old_state.coords, dtype=np.float64)
        new_coords_f64 = np.asarray(new_coords, dtype=np.float64)
        old_solution = np.asarray(old_state.solution, dtype=np.float64)

        # Degenerate case: nothing to interpolate from.
        if old_coords.size == 0:
            return np.zeros(len(new_coords_f64), dtype=np.float32)

        if old_coords.ndim == 1 or old_coords.shape[1] == 1:
            # 1-D: sort by x and use numpy linear interpolation; out-of-range
            # points are clamped to the edge values (extrapolation would be
            # worse than a constant for a collocation mesh).
            xs = old_coords.reshape(-1)
            order = np.argsort(xs)
            xs_sorted = xs[order]
            ys_sorted = old_solution[order]
            queries = new_coords_f64.reshape(-1)
            interpolated = np.interp(queries, xs_sorted, ys_sorted)
            return interpolated.astype(np.float32)

        # 2-D and higher: linear interpolation with nearest-neighbor fallback.
        # ``LinearNDInterpolator.__init__`` runs Delaunay triangulation on
        # the source points (Qhull), so we cache the interpolator by source
        # state identity to avoid re-triangulating across the candidate
        # actions of a single MCTS expansion. ``NearestNDInterpolator`` is
        # only built lazily, on the cold path where points fall outside the
        # convex hull.
        nearest: NearestNDInterpolator | None = None
        cache_hit = (
            self._cached_interp_state is old_state and self._cached_interp_linear is not None
        )
        if cache_hit:
            linear = self._cached_interp_linear
        else:
            t0 = time.perf_counter()
            try:
                linear = LinearNDInterpolator(old_coords, old_solution, fill_value=np.nan)
            except Exception as exc:
                # Degenerate triangulation (e.g. collinear source points).
                # Covered by tests/pde/test_mesh_refinement.py
                # ``TestMeshRefinementGameInterpolation::
                # test_degenerate_triangulation_*``.
                logger.warning(
                    "interpolator_build_failed",
                    n_points=len(old_coords),
                    error=str(exc),
                )
                linear = None
            else:
                self._cached_interp_state = old_state
                self._cached_interp_linear = linear
            logger.debug(
                "interpolator_built",
                n_points=len(old_coords),
                build_time_ms=(time.perf_counter() - t0) * 1e3,
            )

        if linear is None:
            values = np.full(len(new_coords_f64), np.nan)
        else:
            values = linear(new_coords_f64)

        missing = np.isnan(values)
        n_missing = int(missing.sum())
        if n_missing:
            nearest = NearestNDInterpolator(old_coords, old_solution)
            values[missing] = nearest(new_coords_f64[missing])
            logger.debug(
                "interpolation_nn_fallback",
                n_query_points=len(new_coords_f64),
                n_missing=n_missing,
                fraction=float(n_missing) / float(len(new_coords_f64)),
                cache_hit=cache_hit,
            )

        return values.astype(np.float32)

    def get_reward(self, state: PDEState, prev_state: PDEState) -> float:
        """Compute reward for refinement action.

        Two forms are supported, selected by ``PDEGameConfig.reward_form``:

        * ``"linear"`` (default): error-reduction reward minus DOF cost
          plus an efficiency bonus plus terminal bonus.
        * ``"log"``: the DOE Genesis proposal reward
          ``-alpha * log(error) - beta * log(cost)`` with ``cost = state.dof``,
          plus the terminal bonus.

        Args:
            state: New state.
            prev_state: Previous state.

        Returns:
            Reward value.

        """
        if self.config.reward_form == "log":
            reward = log_reward(
                error=state.error_estimate,
                cost=float(state.dof),
                alpha=self.config.log_reward_alpha,
                beta=self.config.log_reward_beta,
                epsilon=self.config.log_reward_epsilon,
            )
            if state.error_estimate < self.config.error_tolerance:
                reward += self.config.terminal_bonus

            logger.debug(
                "reward_computed",
                form="log",
                error=state.error_estimate,
                dof=state.dof,
                total_reward=reward,
            )
            return reward

        # Linear form (historical default).
        error_reduction = prev_state.error_estimate - state.error_estimate

        dof_added = state.dof - prev_state.dof
        cost = self.config.cost_per_dof * dof_added

        efficiency_threshold = self.mesh_config.efficiency_threshold
        efficiency_multiplier = self.mesh_config.efficiency_multiplier

        if dof_added > 0:
            efficiency = error_reduction / dof_added
            efficiency_bonus = max(0, efficiency - efficiency_threshold) * efficiency_multiplier
        else:
            efficiency_bonus = 0.0

        reward = self.config.reward_per_error_reduction * error_reduction - cost + efficiency_bonus

        if state.error_estimate < self.config.error_tolerance:
            reward += self.config.terminal_bonus

        logger.debug(
            "reward_computed",
            error_reduction=error_reduction,
            dof_added=dof_added,
            efficiency_bonus=efficiency_bonus,
            total_reward=reward,
        )

        return reward

    def is_terminal(self, state: PDEState) -> bool:
        """Check if game has ended.

        Args:
            state: Current state.

        Returns:
            True if terminal.

        """
        if state.error_estimate < self.config.error_tolerance:
            return True
        if state.dof > self.config.max_dof:
            return True
        if state.budget_remaining <= 0:
            return True
        if state.step >= self.config.max_steps:
            return True
        return len(self.get_valid_actions(state)) == 0

    def _capacity_reason(self, state: PDEState) -> str | None:
        """Report ``"max_dof"`` once the DOF cap is exceeded.

        Uses the same strict ``>`` as :meth:`is_terminal`: a state sitting
        exactly on ``max_dof`` is at capacity but not yet over it.
        """
        if state.dof > self.config.max_dof:
            return "max_dof"
        return None

    def compute_exact_error(self, state: PDEState) -> dict[str, float]:
        """Compute error metrics.

        Args:
            state: Current state.

        Returns:
            Error dictionary.

        """
        # Get exact solution if available
        exact = self.pde_operator.exact_solution(state.coords)

        if exact is not None:
            if isinstance(exact, Tensor):
                exact = exact.numpy()
            l2_error = float(np.sqrt(np.mean((state.solution - exact) ** 2)))
            linf_error = float(np.max(np.abs(state.solution - exact)))
        else:
            l2_error = float(np.sqrt(np.mean(state.residuals**2)))
            linf_error = float(np.max(np.abs(state.residuals)))

        h1_error = l2_error  # Approximation
        residual_norm = float(np.sqrt(np.mean(state.residuals**2)))

        return {
            "l2": l2_error,
            "h1": h1_error,
            "linf": linf_error,
            "residual": residual_norm,
        }

    def to_tensor(self, state: PDEState) -> Float[Tensor, ...]:
        """Convert state to neural network input.

        Args:
            state: PDE state.

        Returns:
            Tensor encoding with shape:
            - 1D: (channels, resolution)
            - 2D: (channels, height, width)
            - 3D: (channels, depth, height, width)

        """
        from scipy.interpolate import griddata

        grid_size = self.mesh_config.initial_resolution
        dim = self.mesh.dim

        # Generate grid points based on dimension
        domain_min = self.pde_operator.domain_min
        domain_max = self.pde_operator.domain_max

        axes = [np.linspace(domain_min[d], domain_max[d], grid_size) for d in range(dim)]
        grids = np.meshgrid(*axes, indexing="ij")
        grid_points = np.stack([g.flatten() for g in grids], axis=-1)

        # Grid shape for reshaping
        grid_shape = tuple([grid_size] * dim)

        # Interpolate solution
        solution_grid = griddata(
            state.coords, state.solution, grid_points, method="linear", fill_value=0
        ).reshape(grid_shape)

        # Interpolate residuals
        residual_grid = griddata(
            state.coords, np.abs(state.residuals), grid_points, method="linear", fill_value=0
        ).reshape(grid_shape)

        # Refinement level indicator
        if state.mesh_levels is not None:
            level_grid = griddata(
                state.coords,
                state.mesh_levels.astype(np.float32),
                grid_points,
                method="nearest",
                fill_value=0,
            ).reshape(grid_shape)
        else:
            level_grid = np.zeros(grid_shape)

        # Build tensor with shape (channels, *grid_shape)
        tensor_shape = (self.state_channels, *grid_shape)
        tensor = torch.zeros(tensor_shape)
        tensor[0] = torch.from_numpy(solution_grid.astype(np.float32))
        tensor[1] = torch.from_numpy(residual_grid.astype(np.float32))
        tensor[2] = torch.from_numpy(level_grid.astype(np.float32))
        # Additional channels could include element sizes, polynomial degrees, etc.

        return tensor

    def action_to_string(self, action: int) -> str:
        """Convert action to string.

        Args:
            action: Action index.

        Returns:
            Action description.

        """
        if action < 0 or action >= self.action_space_size:
            return f"invalid_action_{action}"
        kind, leaf_idx = self._decode_action(action)
        if leaf_idx >= len(self.mesh.leaf_elements):
            return f"invalid_action_{action}"
        element = self.mesh.leaf_elements[leaf_idx]
        verb = "refine_element" if kind is ActionKind.REFINE else "coarsen_element"
        return f"{verb}({leaf_idx}, level={element.level})"
