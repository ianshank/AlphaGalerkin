"""Tests for PDE-aware curriculum learning."""

from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.training.pde_curriculum import (
    DIFFICULTY_PROGRESSIONS,
    DifficultyDimension,
    PDECurriculumManager,
    PDEDifficultyConfig,
)


class TestDifficultyProgression:
    """Tests for difficulty progression through curriculum stages."""

    def test_difficulty_progression_gradual(self) -> None:
        """The 'gradual' progression has monotonically increasing difficulty."""
        manager = PDECurriculumManager(progression="gradual")

        stages = manager.stages
        assert len(stages) == 5

        # Verify levels are sequential
        for i, stage in enumerate(stages):
            assert stage.level == i

        # Frequency should increase or stay the same
        for i in range(1, len(stages)):
            assert stages[i].source_frequency >= stages[i - 1].source_frequency

        # Descriptions should be non-empty
        for stage in stages:
            assert stage.description != ""

    def test_difficulty_progression_aggressive(self) -> None:
        """The 'aggressive' progression ramps faster."""
        manager = PDECurriculumManager(progression="aggressive")
        assert manager.num_stages == 3

    def test_difficulty_progression_singularity_focused(self) -> None:
        """The 'singularity_focused' progression emphasizes singularity."""
        manager = PDECurriculumManager(progression="singularity_focused")
        stages = manager.stages
        # Last stage should have highest singularity
        assert stages[-1].singularity_strength >= stages[0].singularity_strength

    def test_unknown_progression_raises(self) -> None:
        """An unknown progression name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown progression"):
            PDECurriculumManager(progression="nonexistent")

    def test_empty_custom_stages_raises(self) -> None:
        """Empty custom stages raise ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            PDECurriculumManager(custom_stages=[])


class TestSourceTermGeneration:
    """Tests for source term function generation."""

    def test_source_term_generation(self) -> None:
        """Source term is a callable returning correct shape."""
        manager = PDECurriculumManager(progression="gradual")
        difficulty = manager.get_current_difficulty()
        source = manager.generate_source_term(difficulty)

        points = np.array([[0.5, 0.5], [0.25, 0.75], [0.0, 0.0]])
        result = source(points)

        assert result.shape == (3,)
        assert np.isfinite(result).all()

    def test_source_term_frequency_effect(self) -> None:
        """Higher frequency produces more oscillation."""
        manager = PDECurriculumManager(progression="gradual")

        low_freq = PDEDifficultyConfig(source_frequency=1.0)
        high_freq = PDEDifficultyConfig(source_frequency=4.0)

        source_low = manager.generate_source_term(low_freq)
        source_high = manager.generate_source_term(high_freq)

        # Evaluate on a grid
        x = np.linspace(0.01, 0.99, 100)
        y = np.linspace(0.01, 0.99, 100)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        vals_low = source_low(points)
        vals_high = source_high(points)

        # Higher frequency should have more sign changes
        sign_changes_low = np.sum(np.diff(np.sign(vals_low)) != 0)
        sign_changes_high = np.sum(np.diff(np.sign(vals_high)) != 0)
        assert sign_changes_high > sign_changes_low

    def test_source_term_singularity(self) -> None:
        """Singularity term adds 1/r behaviour near origin."""
        manager = PDECurriculumManager(progression="gradual")

        no_sing = PDEDifficultyConfig(singularity_strength=0.0)
        with_sing = PDEDifficultyConfig(singularity_strength=1.0)

        source_clean = manager.generate_source_term(no_sing)
        source_sing = manager.generate_source_term(with_sing)

        # Point near origin
        near_origin = np.array([[0.001, 0.001]])
        val_clean = source_clean(near_origin)
        val_sing = source_sing(near_origin)

        # Singularity term should dominate near origin
        assert abs(val_sing[0]) > abs(val_clean[0])

    def test_source_term_no_singularity_at_zero(self) -> None:
        """Without singularity, value at origin is well-behaved."""
        manager = PDECurriculumManager(progression="gradual")
        difficulty = PDEDifficultyConfig(
            singularity_strength=0.0,
            source_frequency=1.0,
        )
        source = manager.generate_source_term(difficulty)

        origin = np.array([[0.0, 0.0]])
        result = source(origin)
        assert np.isfinite(result).all()
        assert abs(result[0]) < 1e-10  # sin(0) * sin(0) = 0


class TestDiffusionTensor:
    """Tests for anisotropic diffusion tensor generation."""

    def test_diffusion_tensor_anisotropy(self) -> None:
        """Anisotropic tensor has correct ratio of D_xx / D_yy."""
        manager = PDECurriculumManager(progression="gradual")

        difficulty = PDEDifficultyConfig(
            diffusion_coefficient=1.0,
            anisotropy_ratio=4.0,
        )
        tensor = manager.generate_diffusion_tensor(difficulty)

        assert tensor.shape == (2, 2)
        # Off-diagonal should be zero
        assert tensor[0, 1] == 0.0
        assert tensor[1, 0] == 0.0
        # Ratio should match
        ratio = tensor[0, 0] / tensor[1, 1]
        assert abs(ratio - 4.0) < 1e-10

    def test_diffusion_tensor_isotropic(self) -> None:
        """Isotropic tensor (ratio=1) has equal diagonal entries."""
        manager = PDECurriculumManager(progression="gradual")

        difficulty = PDEDifficultyConfig(
            diffusion_coefficient=2.0,
            anisotropy_ratio=1.0,
        )
        tensor = manager.generate_diffusion_tensor(difficulty)

        assert abs(tensor[0, 0] - tensor[1, 1]) < 1e-10
        assert abs(tensor[0, 0] - 2.0) < 1e-10

    def test_diffusion_tensor_geometric_mean(self) -> None:
        """Geometric mean of D_xx and D_yy equals base coefficient."""
        manager = PDECurriculumManager(progression="gradual")

        base = 3.0
        difficulty = PDEDifficultyConfig(
            diffusion_coefficient=base,
            anisotropy_ratio=9.0,
        )
        tensor = manager.generate_diffusion_tensor(difficulty)

        geom_mean = np.sqrt(tensor[0, 0] * tensor[1, 1])
        assert abs(geom_mean - base) < 1e-10


class TestAdvanceThreshold:
    """Tests for curriculum advancement logic."""

    def test_advance_threshold(self) -> None:
        """Curriculum advances when windowed average exceeds threshold."""
        manager = PDECurriculumManager(
            progression="gradual",
            advance_threshold=0.8,
            evaluation_window=10,
        )
        assert manager.current_stage_index == 0

        # Feed high performance to trigger advancement
        advanced = manager.update(0.9)
        assert advanced
        assert manager.current_stage_index == 1

    def test_no_advance_below_threshold(self) -> None:
        """No advancement when performance is below threshold."""
        manager = PDECurriculumManager(
            progression="gradual",
            advance_threshold=0.8,
            evaluation_window=10,
        )
        advanced = manager.update(0.5)
        assert not advanced
        assert manager.current_stage_index == 0

    def test_no_advance_past_final_stage(self) -> None:
        """Cannot advance past the final stage."""
        stages = [
            PDEDifficultyConfig(level=0, description="first"),
            PDEDifficultyConfig(level=1, description="last"),
        ]
        manager = PDECurriculumManager(
            custom_stages=stages,
            advance_threshold=0.5,
            evaluation_window=10,
        )
        # Advance to stage 1
        manager.update(0.9)
        assert manager.current_stage_index == 1
        assert manager.is_at_final_stage

        # Try to advance again
        advanced = manager.update(0.9)
        assert not advanced
        assert manager.current_stage_index == 1

    def test_windowed_evaluation(self) -> None:
        """Average is computed over the evaluation window."""
        # Use only 2 stages so we can control advancement precisely
        stages = [
            PDEDifficultyConfig(level=0, description="first"),
            PDEDifficultyConfig(level=1, description="second"),
        ]
        manager = PDECurriculumManager(
            custom_stages=stages,
            advance_threshold=0.8,
            evaluation_window=10,
        )
        # Feed mixed performance -- average stays below 0.8
        for _ in range(5):
            manager.update(0.5)
        assert manager.current_stage_index == 0

        # Now feed high performance to push windowed average above 0.8
        for _ in range(10):
            manager.update(1.0)
        assert manager.current_stage_index == 1


class TestEnvOverrides:
    """Tests for converting difficulty to environment overrides."""

    def test_env_overrides_from_difficulty(self) -> None:
        """Overrides dict contains expected keys and values."""
        manager = PDECurriculumManager(progression="gradual")

        difficulty = PDEDifficultyConfig(
            mesh_resolution=8,
            source_frequency=4.0,
            diffusion_coefficient=2.0,
            anisotropy_ratio=5.0,
            singularity_strength=0.3,
            boundary_type="mixed",
        )
        overrides = manager.to_env_overrides(difficulty)

        assert overrides["initial_mesh_resolution"] == 8
        assert overrides["source_frequency"] == 4.0
        assert overrides["diffusion_coefficient"] == 2.0
        assert overrides["anisotropy_ratio"] == 5.0
        assert overrides["singularity_strength"] == 0.3
        assert overrides["boundary_type"] == "mixed"

    def test_env_overrides_default_values(self) -> None:
        """Default difficulty produces default override values."""
        manager = PDECurriculumManager(progression="gradual")
        difficulty = PDEDifficultyConfig()
        overrides = manager.to_env_overrides(difficulty)

        assert overrides["initial_mesh_resolution"] == 4
        assert overrides["source_frequency"] == 1.0
        assert overrides["anisotropy_ratio"] == 1.0
        assert overrides["singularity_strength"] == 0.0


class TestPDECurriculumReset:
    """Tests for curriculum reset."""

    def test_pde_curriculum_reset(self) -> None:
        """Reset restores stage index and clears history."""
        manager = PDECurriculumManager(
            progression="gradual",
            advance_threshold=0.5,
            evaluation_window=10,
        )
        manager.update(0.9)
        assert manager.current_stage_index > 0

        manager.reset()
        assert manager.current_stage_index == 0

    def test_reset_allows_re_advance(self) -> None:
        """After reset, curriculum can advance again."""
        stages = [
            PDEDifficultyConfig(level=0, description="first"),
            PDEDifficultyConfig(level=1, description="last"),
        ]
        manager = PDECurriculumManager(
            custom_stages=stages,
            advance_threshold=0.5,
            evaluation_window=10,
        )
        manager.update(0.9)
        assert manager.current_stage_index == 1

        manager.reset()
        assert manager.current_stage_index == 0

        manager.update(0.9)
        assert manager.current_stage_index == 1


class TestCustomStages:
    """Tests for custom stage configurations."""

    def test_custom_stages(self) -> None:
        """Custom stages override the named progression."""
        custom = [
            PDEDifficultyConfig(
                level=0,
                source_frequency=10.0,
                description="custom_first",
            ),
            PDEDifficultyConfig(
                level=1,
                source_frequency=20.0,
                description="custom_second",
            ),
        ]
        manager = PDECurriculumManager(custom_stages=custom)

        assert manager.num_stages == 2
        assert manager.get_current_difficulty().source_frequency == 10.0
        assert manager.get_current_difficulty().description == "custom_first"

    def test_custom_stages_advance(self) -> None:
        """Custom stages can be advanced through."""
        custom = [
            PDEDifficultyConfig(level=0, description="a"),
            PDEDifficultyConfig(level=1, description="b"),
            PDEDifficultyConfig(level=2, description="c"),
        ]
        manager = PDECurriculumManager(
            custom_stages=custom,
            advance_threshold=0.5,
            evaluation_window=10,
        )
        manager.update(0.9)
        assert manager.current_stage_index == 1
        assert manager.get_current_difficulty().description == "b"

        manager.update(0.9)
        assert manager.current_stage_index == 2
        assert manager.is_at_final_stage

    def test_custom_stages_with_all_fields(self) -> None:
        """All PDEDifficultyConfig fields can be set in custom stages."""
        custom = [
            PDEDifficultyConfig(
                level=0,
                mesh_resolution=16,
                source_frequency=5.0,
                diffusion_coefficient=0.5,
                anisotropy_ratio=3.0,
                singularity_strength=0.7,
                boundary_type="neumann",
                description="fully configured",
            ),
        ]
        manager = PDECurriculumManager(custom_stages=custom)
        diff = manager.get_current_difficulty()

        assert diff.mesh_resolution == 16
        assert diff.source_frequency == 5.0
        assert diff.diffusion_coefficient == 0.5
        assert diff.anisotropy_ratio == 3.0
        assert diff.singularity_strength == 0.7
        assert diff.boundary_type == "neumann"


class TestDifficultyDimensionEnum:
    """Tests for the DifficultyDimension enum."""

    def test_all_dimensions_exist(self) -> None:
        """All expected difficulty dimensions are defined."""
        expected = {
            "mesh_size",
            "frequency",
            "anisotropy",
            "nonlinearity",
            "multi_scale",
            "singularity",
            "boundary_complexity",
        }
        actual = {d.value for d in DifficultyDimension}
        assert actual == expected


class TestPredefinedProgressions:
    """Tests for predefined difficulty progressions."""

    def test_all_progressions_exist(self) -> None:
        """All expected progressions are defined."""
        expected = {"gradual", "aggressive", "singularity_focused"}
        assert set(DIFFICULTY_PROGRESSIONS.keys()) == expected

    def test_all_progressions_non_empty(self) -> None:
        """Every predefined progression has at least one stage."""
        for name, stages in DIFFICULTY_PROGRESSIONS.items():
            assert len(stages) > 0, f"Progression {name!r} is empty"

    def test_all_progressions_constructible(self) -> None:
        """Every predefined progression can create a manager."""
        for name in DIFFICULTY_PROGRESSIONS:
            manager = PDECurriculumManager(progression=name)
            assert manager.num_stages > 0
