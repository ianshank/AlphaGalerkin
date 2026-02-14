"""Temperature schedule for action selection."""
from __future__ import annotations

import numpy as np
import structlog

from src.alphagalerkin.core.types import TemperatureScheduleType

logger = structlog.get_logger("mcts.temperature")


class TemperatureSchedule:
    """Anneals temperature over episode steps.

    The temperature controls the stochasticity of action
    selection from MCTS visit counts:

    * High temperature (close to 1): actions sampled nearly
      proportional to visit counts -- more exploration.
    * Low temperature (close to 0): converges to deterministic
      argmax -- pure exploitation.

    Args:
        schedule_type: Shape of the annealing curve.
        initial: Starting temperature (must be > 0).
        final: Final temperature after *decay_steps*.
        decay_steps: Number of steps over which to decay.

    """

    def __init__(
        self,
        schedule_type: TemperatureScheduleType = (
            TemperatureScheduleType.LINEAR
        ),
        initial: float = 1.0,
        final: float = 0.1,
        decay_steps: int = 30,
    ) -> None:
        if initial <= 0:
            msg = f"initial must be positive, got {initial}"
            raise ValueError(msg)
        if final <= 0:
            msg = f"final must be positive, got {final}"
            raise ValueError(msg)
        if decay_steps < 1:
            msg = f"decay_steps must be >= 1, got {decay_steps}"
            raise ValueError(msg)

        self._type = schedule_type
        self._initial = initial
        self._final = final
        self._decay_steps = decay_steps

    # ---------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------

    @property
    def schedule_type(self) -> TemperatureScheduleType:
        """The annealing schedule shape."""
        return self._type

    @property
    def initial(self) -> float:
        """Starting temperature."""
        return self._initial

    @property
    def final(self) -> float:
        """Final temperature."""
        return self._final

    @property
    def decay_steps(self) -> int:
        """Number of annealing steps."""
        return self._decay_steps

    # ---------------------------------------------------------------
    # Temperature computation
    # ---------------------------------------------------------------

    def get_temperature(self, step: int) -> float:
        """Return the temperature at *step*.

        Args:
            step: Current episode step (0-indexed).

        Returns:
            Temperature value, clamped to ``[final, initial]``.

        """
        if step >= self._decay_steps:
            return self._final

        progress = step / max(1, self._decay_steps)

        if self._type == TemperatureScheduleType.LINEAR:
            return (
                self._initial
                + (self._final - self._initial) * progress
            )

        if self._type == TemperatureScheduleType.EXPONENTIAL:
            ratio = self._final / max(self._initial, 1e-10)
            return self._initial * (ratio ** progress)

        if self._type == TemperatureScheduleType.STEP:
            return self._initial

        if self._type == TemperatureScheduleType.CONSTANT:
            return self._initial

        return self._initial  # pragma: no cover

    # ---------------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------------

    def select_action_with_temperature(
        self,
        visit_counts: dict,
        temperature: float,
        rng: np.random.Generator | None = None,
    ) -> object:
        """Select an action from visit counts and temperature.

        With temperature > 0 the action is sampled proportional
        to ``N(a) ^ (1 / T)``.  With temperature near 0 the
        action with the highest count is picked deterministically.

        Args:
            visit_counts: Mapping action -> visit count.
            temperature: Current temperature.
            rng: Optional NumPy random generator.

        Returns:
            The selected action (same type as the dict keys).

        """
        if rng is None:
            rng = np.random.default_rng()

        actions = list(visit_counts.keys())
        counts = np.array(
            [float(visit_counts[a]) for a in actions],
        )

        if temperature < 1e-8:
            # Deterministic argmax
            best_idx = int(np.argmax(counts))
            return actions[best_idx]

        # Temperature-scaled probabilities
        scaled = counts ** (1.0 / temperature)
        total = scaled.sum()
        if total == 0:
            probs = np.ones(len(actions)) / len(actions)
        else:
            probs = scaled / total

        idx = int(rng.choice(len(actions), p=probs))
        return actions[idx]
