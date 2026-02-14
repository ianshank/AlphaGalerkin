"""PDE-aware curriculum that generates progressively harder problems.

Unlike the base :class:`CurriculumManager` which only adjusts
environment parameters (DOF budget, step limit), this manager generates
actual PDE problems with increasing difficulty:

- Higher frequency source terms
- Anisotropic diffusion coefficients
- Point singularities
- Multi-scale features
- Complex boundary conditions

The "opponent" in AlphaGalerkin is the PDE itself, and this module
implements the progressive generation of harder test cases for
curriculum learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger("training.pde_curriculum")


class DifficultyDimension(str, Enum):
    """Dimensions along which PDE difficulty can increase."""

    MESH_SIZE = "mesh_size"
    """More elements in initial mesh."""

    FREQUENCY = "frequency"
    """Higher frequency in source term."""

    ANISOTROPY = "anisotropy"
    """Anisotropic diffusion coefficients."""

    NONLINEARITY = "nonlinearity"
    """Degree of nonlinearity."""

    MULTI_SCALE = "multi_scale"
    """Multi-scale features in solution."""

    SINGULARITY = "singularity"
    """Point or edge singularities."""

    BOUNDARY_COMPLEXITY = "boundary_complexity"
    """Complex boundary conditions."""


@dataclass
class PDEDifficultyConfig:
    """Configuration for one difficulty level.

    Attributes
    ----------
    level:
        Integer difficulty level (0-based).
    mesh_resolution:
        Elements per side in the initial mesh.
    source_frequency:
        Frequency multiplier for the source term.
    diffusion_coefficient:
        Base diffusion coefficient.
    anisotropy_ratio:
        Ratio of x-direction to y-direction diffusion.
        1.0 means isotropic.
    singularity_strength:
        Strength of a point singularity at the origin.
        0.0 means no singularity, 1.0 means a strong one.
    boundary_type:
        Type of boundary conditions (``"dirichlet"``,
        ``"neumann"``, or ``"mixed"``).
    description:
        Human-readable description of this difficulty level.

    """

    level: int = 0
    mesh_resolution: int = 4
    source_frequency: float = 1.0
    diffusion_coefficient: float = 1.0
    anisotropy_ratio: float = 1.0
    singularity_strength: float = 0.0
    boundary_type: str = "dirichlet"
    description: str = ""


# -------------------------------------------------------------------
# Predefined difficulty progressions
# -------------------------------------------------------------------

DIFFICULTY_PROGRESSIONS: dict[str, list[PDEDifficultyConfig]] = {
    "gradual": [
        PDEDifficultyConfig(
            level=0,
            mesh_resolution=4,
            source_frequency=1.0,
            description="Smooth, coarse mesh",
        ),
        PDEDifficultyConfig(
            level=1,
            mesh_resolution=4,
            source_frequency=2.0,
            description="Higher frequency source",
        ),
        PDEDifficultyConfig(
            level=2,
            mesh_resolution=8,
            source_frequency=2.0,
            anisotropy_ratio=2.0,
            description="Finer mesh, anisotropic",
        ),
        PDEDifficultyConfig(
            level=3,
            mesh_resolution=8,
            source_frequency=4.0,
            anisotropy_ratio=5.0,
            description="High frequency, strongly anisotropic",
        ),
        PDEDifficultyConfig(
            level=4,
            mesh_resolution=8,
            source_frequency=8.0,
            anisotropy_ratio=10.0,
            singularity_strength=0.5,
            description="Multi-scale with mild singularity",
        ),
    ],
    "aggressive": [
        PDEDifficultyConfig(
            level=0,
            mesh_resolution=4,
            source_frequency=2.0,
            description="Medium frequency start",
        ),
        PDEDifficultyConfig(
            level=1,
            mesh_resolution=8,
            source_frequency=4.0,
            anisotropy_ratio=5.0,
            description="Fast ramp to anisotropic",
        ),
        PDEDifficultyConfig(
            level=2,
            mesh_resolution=8,
            source_frequency=8.0,
            anisotropy_ratio=10.0,
            singularity_strength=0.8,
            description="High difficulty with strong singularity",
        ),
    ],
    "singularity_focused": [
        PDEDifficultyConfig(
            level=0,
            mesh_resolution=4,
            source_frequency=1.0,
            description="Smooth baseline",
        ),
        PDEDifficultyConfig(
            level=1,
            mesh_resolution=4,
            source_frequency=1.0,
            singularity_strength=0.2,
            description="Mild singularity",
        ),
        PDEDifficultyConfig(
            level=2,
            mesh_resolution=8,
            source_frequency=1.0,
            singularity_strength=0.5,
            description="Moderate singularity, finer mesh",
        ),
        PDEDifficultyConfig(
            level=3,
            mesh_resolution=8,
            source_frequency=2.0,
            singularity_strength=0.8,
            description="Strong singularity with higher frequency",
        ),
        PDEDifficultyConfig(
            level=4,
            mesh_resolution=8,
            source_frequency=4.0,
            singularity_strength=1.0,
            anisotropy_ratio=3.0,
            description="Full singularity, anisotropic, high frequency",
        ),
    ],
}


class PDECurriculumManager:
    """Curriculum manager that generates progressively harder PDEs.

    Unlike the base :class:`CurriculumManager` which only adjusts
    environment parameters (DOF budget, step limit), this manager
    generates actual PDE problems with increasing difficulty:

    - Higher frequency source terms
    - Anisotropic diffusion coefficients
    - Point singularities
    - Multi-scale features
    - Complex boundary conditions

    The curriculum advances to the next stage once the agent's
    recent performance exceeds ``advance_threshold`` over a sliding
    window of ``evaluation_window`` episodes.

    Parameters
    ----------
    progression:
        Name of a predefined difficulty progression from
        :data:`DIFFICULTY_PROGRESSIONS`.  Ignored when
        ``custom_stages`` is provided.
    custom_stages:
        Optional list of :class:`PDEDifficultyConfig` stages.
        When provided, overrides the named ``progression``.
    advance_threshold:
        Performance threshold (in ``[0, 1]``) to advance.
    evaluation_window:
        Number of recent performance samples to average.

    """

    def __init__(
        self,
        progression: str = "gradual",
        custom_stages: list[PDEDifficultyConfig] | None = None,
        advance_threshold: float = 0.8,
        evaluation_window: int = 100,
    ) -> None:
        if custom_stages is not None:
            self._stages = list(custom_stages)
        else:
            if progression not in DIFFICULTY_PROGRESSIONS:
                available = ", ".join(sorted(DIFFICULTY_PROGRESSIONS))
                msg = (
                    f"Unknown progression {progression!r}. "
                    f"Available: {available}"
                )
                raise ValueError(msg)
            self._stages = list(DIFFICULTY_PROGRESSIONS[progression])

        if not self._stages:
            msg = "At least one difficulty stage is required"
            raise ValueError(msg)

        self._advance_threshold = advance_threshold
        self._evaluation_window = evaluation_window
        self._current_stage_idx: int = 0
        self._performance_history: list[float] = []

        logger.info(
            "pde_curriculum.init",
            num_stages=len(self._stages),
            advance_threshold=advance_threshold,
            evaluation_window=evaluation_window,
            first_stage=self._stages[0].description,
        )

    # ---------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------

    @property
    def current_stage_index(self) -> int:
        """Zero-based index of the current curriculum stage."""
        return self._current_stage_idx

    @property
    def is_at_final_stage(self) -> bool:
        """Whether the curriculum has reached the last stage."""
        return self._current_stage_idx >= len(self._stages) - 1

    @property
    def num_stages(self) -> int:
        """Total number of curriculum stages."""
        return len(self._stages)

    @property
    def stages(self) -> list[PDEDifficultyConfig]:
        """All difficulty stages (read-only copy)."""
        return list(self._stages)

    # ---------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------

    def get_current_difficulty(self) -> PDEDifficultyConfig:
        """Return the difficulty config for the current stage."""
        idx = min(self._current_stage_idx, len(self._stages) - 1)
        return self._stages[idx]

    def generate_source_term(
        self,
        difficulty: PDEDifficultyConfig,
    ) -> Any:
        """Generate a source term function for the given difficulty.

        Returns a callable ``f(points: np.ndarray) -> np.ndarray``
        where ``points`` has shape ``(N, 2)`` and the result has
        shape ``(N,)``.

        Parameters
        ----------
        difficulty:
            The difficulty configuration controlling frequency
            and singularity parameters.

        Returns
        -------
        callable
            Source term function.

        """
        freq = difficulty.source_frequency
        sing_strength = difficulty.singularity_strength

        def source(points: np.ndarray) -> np.ndarray:
            x = points[:, 0]
            y = points[:, 1]
            result = np.sin(freq * np.pi * x) * np.sin(
                freq * np.pi * y
            )
            if sing_strength > 0:
                r = np.sqrt(x**2 + y**2) + 1e-10
                result = result + sing_strength / r
            return result

        return source

    def generate_diffusion_tensor(
        self,
        difficulty: PDEDifficultyConfig,
    ) -> np.ndarray:
        """Generate an anisotropic diffusion tensor.

        Returns a 2x2 diagonal matrix ``[[D_xx, 0], [0, D_yy]]``
        where the ratio ``D_xx / D_yy`` equals the configured
        ``anisotropy_ratio``.

        Parameters
        ----------
        difficulty:
            The difficulty configuration controlling diffusion
            coefficient and anisotropy ratio.

        Returns
        -------
        np.ndarray
            Shape ``(2, 2)`` diffusion tensor.

        """
        base = difficulty.diffusion_coefficient
        ratio = difficulty.anisotropy_ratio

        # D_xx * D_yy = base^2  (preserve geometric mean)
        # D_xx / D_yy = ratio
        d_xx = base * np.sqrt(ratio)
        d_yy = base / np.sqrt(ratio)

        return np.array([[d_xx, 0.0], [0.0, d_yy]])

    def to_env_overrides(
        self,
        difficulty: PDEDifficultyConfig,
    ) -> dict[str, Any]:
        """Convert difficulty config to environment overrides.

        This bridges the PDE curriculum to the existing
        :class:`CurriculumManager` interface by producing a dict
        of environment-config overrides.

        Parameters
        ----------
        difficulty:
            The difficulty configuration to convert.

        Returns
        -------
        dict[str, Any]
            Environment-config overrides including mesh resolution,
            source frequency, diffusion coefficient, anisotropy
            ratio, and singularity strength.

        """
        return {
            "initial_mesh_resolution": difficulty.mesh_resolution,
            "source_frequency": difficulty.source_frequency,
            "diffusion_coefficient": difficulty.diffusion_coefficient,
            "anisotropy_ratio": difficulty.anisotropy_ratio,
            "singularity_strength": difficulty.singularity_strength,
            "boundary_type": difficulty.boundary_type,
        }

    def update(self, performance: float) -> bool:
        """Update curriculum based on performance.

        Records the performance value, and if the running average
        over the evaluation window exceeds the advancement
        threshold, advances to the next stage.

        Parameters
        ----------
        performance:
            Performance metric from the latest evaluation
            (in ``[0, 1]``).

        Returns
        -------
        bool
            ``True`` if the curriculum advanced to a new stage.

        """
        self._performance_history.append(performance)

        window = self._evaluation_window
        recent = self._performance_history[-window:]
        avg_performance = sum(recent) / len(recent)

        if avg_performance >= self._advance_threshold and not self.is_at_final_stage:
            old_stage = self._current_stage_idx
            self._current_stage_idx += 1
            new_difficulty = self.get_current_difficulty()
            logger.info(
                "pde_curriculum.advanced",
                old_stage=old_stage,
                new_stage=self._current_stage_idx,
                avg_performance=round(avg_performance, 4),
                threshold=self._advance_threshold,
                new_description=new_difficulty.description,
                new_frequency=new_difficulty.source_frequency,
                new_anisotropy=new_difficulty.anisotropy_ratio,
            )
            return True

        return False

    def reset(self) -> None:
        """Reset the curriculum to the first stage."""
        self._current_stage_idx = 0
        self._performance_history.clear()
        logger.info("pde_curriculum.reset")
