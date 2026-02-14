"""Tests for the Neural Operator Architecture Search framework."""
from __future__ import annotations

import pytest

from src.alphagalerkin.planning.neural_arch_search import (
    ArchitectureState,
    LayerSpec,
    NASAction,
    NASActionType,
    NeuralOperatorNAS,
    OperatorBlockType,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def sample_layers() -> list[LayerSpec]:
    """A minimal three-layer architecture for testing."""
    return [
        LayerSpec(
            block_type=OperatorBlockType.FOURIER_LAYER,
            width=64,
            num_modes=12,
        ),
        LayerSpec(
            block_type=OperatorBlockType.GALERKIN_ATTENTION,
            width=64,
        ),
        LayerSpec(
            block_type=OperatorBlockType.MLP_LAYER,
            width=64,
        ),
    ]


@pytest.fixture()
def sample_state(sample_layers: list[LayerSpec]) -> ArchitectureState:
    """An architecture state initialised with the sample layers."""
    return ArchitectureState(
        layers=sample_layers,
        input_dim=2,
        output_dim=1,
        max_layers=10,
        max_width=256,
        min_width=16,
        max_modes=64,
    )


@pytest.fixture()
def nas() -> NeuralOperatorNAS:
    """A NAS engine configured for quick tests."""
    return NeuralOperatorNAS(
        max_layers=10,
        max_width=256,
        min_width=16,
        num_simulations=2,
        width_step=16,
        mode_step=4,
        complexity_penalty=1e-6,
    )


# ------------------------------------------------------------------
# LayerSpec tests
# ------------------------------------------------------------------


class TestLayerSpecClone:
    """LayerSpec.clone produces an independent copy."""

    def test_layer_spec_clone(self) -> None:
        layer = LayerSpec(
            block_type=OperatorBlockType.FOURIER_LAYER,
            width=128,
            num_modes=16,
            activation="relu",
            has_residual=False,
            skip_to=3,
        )
        cloned = layer.clone()

        # Must be a different object
        assert cloned is not layer

        # Values must match
        assert cloned.block_type == layer.block_type
        assert cloned.width == layer.width
        assert cloned.num_modes == layer.num_modes
        assert cloned.activation == layer.activation
        assert cloned.has_residual == layer.has_residual
        assert cloned.skip_to == layer.skip_to

        # Mutation independence
        cloned.width = 999
        assert layer.width == 128


class TestLayerSpecParamCount:
    """LayerSpec.param_count_estimate returns a positive integer."""

    def test_layer_spec_param_count(self) -> None:
        layer = LayerSpec(
            block_type=OperatorBlockType.FOURIER_LAYER,
            width=64,
            num_modes=12,
        )
        count = layer.param_count_estimate
        assert isinstance(count, int)
        assert count > 0

        # Fourier layers include mode-dependent params
        # base = 2.0 * 64 * 64 + 64 * 12 = 8192 + 768 = 8960
        assert count == 2 * 64 * 64 + 64 * 12

    def test_mlp_layer_param_count(self) -> None:
        layer = LayerSpec(
            block_type=OperatorBlockType.MLP_LAYER,
            width=32,
        )
        count = layer.param_count_estimate
        # MLP multiplier = 1.0, so 1.0 * 32 * 32 = 1024
        assert count == 1 * 32 * 32

    def test_wider_layer_has_more_params(self) -> None:
        narrow = LayerSpec(
            block_type=OperatorBlockType.MLP_LAYER,
            width=32,
        )
        wide = LayerSpec(
            block_type=OperatorBlockType.MLP_LAYER,
            width=128,
        )
        assert wide.param_count_estimate > narrow.param_count_estimate


# ------------------------------------------------------------------
# ArchitectureState tests
# ------------------------------------------------------------------


class TestArchitectureStateClone:
    """ArchitectureState.clone produces an independent copy."""

    def test_architecture_state_clone(
        self, sample_state: ArchitectureState,
    ) -> None:
        cloned = sample_state.clone()

        # Must be different objects
        assert cloned is not sample_state
        assert cloned.layers is not sample_state.layers

        # Each layer must be cloned
        for orig, cpy in zip(sample_state.layers, cloned.layers):
            assert cpy is not orig
            assert cpy.block_type == orig.block_type
            assert cpy.width == orig.width

        # Scalar values must match
        assert cloned.input_dim == sample_state.input_dim
        assert cloned.output_dim == sample_state.output_dim
        assert cloned.max_layers == sample_state.max_layers
        assert cloned.validation_error == sample_state.validation_error
        assert cloned.step == sample_state.step

    def test_clone_mutation_independence(
        self, sample_state: ArchitectureState,
    ) -> None:
        cloned = sample_state.clone()
        cloned.validation_error = 0.001
        cloned.layers[0].width = 999

        assert sample_state.validation_error == float("inf")
        assert sample_state.layers[0].width == 64


class TestArchitectureStateTotalParams:
    """ArchitectureState.total_params sums all layer estimates."""

    def test_architecture_state_total_params(
        self, sample_state: ArchitectureState,
    ) -> None:
        total = sample_state.total_params
        expected = sum(l.param_count_estimate for l in sample_state.layers)
        assert total == expected
        assert total > 0


class TestArchitectureStateDepth:
    """ArchitectureState.depth returns the layer count."""

    def test_architecture_state_depth(
        self, sample_state: ArchitectureState,
    ) -> None:
        assert sample_state.depth == 3

    def test_empty_architecture_depth(self) -> None:
        state = ArchitectureState(layers=[])
        assert state.depth == 0


# ------------------------------------------------------------------
# NeuralOperatorNAS action tests
# ------------------------------------------------------------------


class TestNASValidActions:
    """NeuralOperatorNAS.get_valid_actions returns correct actions."""

    def test_nas_valid_actions(
        self,
        nas: NeuralOperatorNAS,
        sample_state: ArchitectureState,
    ) -> None:
        actions = nas.get_valid_actions(sample_state)

        # Must always contain NO_OP
        action_types = {a.action_type for a in actions}
        assert NASActionType.NO_OP in action_types

        # With 3 layers (< 10 max), ADD_LAYER should be present
        assert NASActionType.ADD_LAYER in action_types

        # With 3 layers (> 1), REMOVE_LAYER should be present
        assert NASActionType.REMOVE_LAYER in action_types

        # Width adjustments should be present for mid-range widths
        assert NASActionType.ADJUST_WIDTH in action_types

        # The first layer is Fourier, so ADJUST_MODES should be present
        assert NASActionType.ADJUST_MODES in action_types

        # Change type and toggle residual should always be present
        assert NASActionType.CHANGE_LAYER_TYPE in action_types
        assert NASActionType.TOGGLE_RESIDUAL in action_types

    def test_no_add_at_max_layers(
        self,
        nas: NeuralOperatorNAS,
    ) -> None:
        """ADD_LAYER is excluded when at max layer count."""
        layers = [
            LayerSpec(block_type=OperatorBlockType.MLP_LAYER)
            for _ in range(10)
        ]
        state = ArchitectureState(
            layers=layers,
            max_layers=10,
            max_width=256,
            min_width=16,
        )
        actions = nas.get_valid_actions(state)
        action_types = {a.action_type for a in actions}
        assert NASActionType.ADD_LAYER not in action_types

    def test_no_remove_single_layer(
        self,
        nas: NeuralOperatorNAS,
    ) -> None:
        """REMOVE_LAYER is excluded when only 1 layer remains."""
        state = ArchitectureState(
            layers=[LayerSpec(block_type=OperatorBlockType.MLP_LAYER)],
            max_layers=10,
            max_width=256,
            min_width=16,
        )
        actions = nas.get_valid_actions(state)
        action_types = {a.action_type for a in actions}
        assert NASActionType.REMOVE_LAYER not in action_types


class TestNASAddLayer:
    """Applying ADD_LAYER increases depth."""

    def test_nas_add_layer(
        self,
        nas: NeuralOperatorNAS,
        sample_state: ArchitectureState,
    ) -> None:
        action = NASAction(
            action_type=NASActionType.ADD_LAYER,
            layer_index=sample_state.depth,
            params={"block_type": OperatorBlockType.WAVELET_LAYER.value},
        )
        new_state = nas.apply_action(sample_state, action)

        assert new_state.depth == sample_state.depth + 1
        assert new_state.step == sample_state.step + 1
        assert new_state.layers[-1].block_type == OperatorBlockType.WAVELET_LAYER
        # Original state is unmodified
        assert sample_state.depth == 3


class TestNASRemoveLayer:
    """Applying REMOVE_LAYER decreases depth."""

    def test_nas_remove_layer(
        self,
        nas: NeuralOperatorNAS,
        sample_state: ArchitectureState,
    ) -> None:
        original_depth = sample_state.depth
        action = NASAction(
            action_type=NASActionType.REMOVE_LAYER,
            layer_index=1,
        )
        new_state = nas.apply_action(sample_state, action)

        assert new_state.depth == original_depth - 1
        assert new_state.step == sample_state.step + 1
        # The Galerkin attention layer (index 1) was removed
        block_types = [l.block_type for l in new_state.layers]
        assert OperatorBlockType.GALERKIN_ATTENTION not in block_types
        # Original state is unmodified
        assert sample_state.depth == 3


class TestNASChangeType:
    """Applying CHANGE_LAYER_TYPE modifies the block type."""

    def test_nas_change_type(
        self,
        nas: NeuralOperatorNAS,
        sample_state: ArchitectureState,
    ) -> None:
        assert sample_state.layers[0].block_type == OperatorBlockType.FOURIER_LAYER

        action = NASAction(
            action_type=NASActionType.CHANGE_LAYER_TYPE,
            layer_index=0,
            params={"block_type": OperatorBlockType.DEEPONET_BRANCH.value},
        )
        new_state = nas.apply_action(sample_state, action)

        assert new_state.layers[0].block_type == OperatorBlockType.DEEPONET_BRANCH
        # Width and other properties are preserved
        assert new_state.layers[0].width == sample_state.layers[0].width
        # Original state is unmodified
        assert sample_state.layers[0].block_type == OperatorBlockType.FOURIER_LAYER


class TestNASAdjustWidth:
    """Applying ADJUST_WIDTH modifies the layer width."""

    def test_nas_adjust_width(
        self,
        nas: NeuralOperatorNAS,
        sample_state: ArchitectureState,
    ) -> None:
        original_width = sample_state.layers[0].width  # 64

        # Increase width
        action_up = NASAction(
            action_type=NASActionType.ADJUST_WIDTH,
            layer_index=0,
            params={"delta": 16},
        )
        state_up = nas.apply_action(sample_state, action_up)
        assert state_up.layers[0].width == original_width + 16

        # Decrease width
        action_down = NASAction(
            action_type=NASActionType.ADJUST_WIDTH,
            layer_index=0,
            params={"delta": -16},
        )
        state_down = nas.apply_action(sample_state, action_down)
        assert state_down.layers[0].width == original_width - 16

    def test_width_clamped_to_bounds(
        self,
        nas: NeuralOperatorNAS,
    ) -> None:
        """Width cannot exceed max_width or go below min_width."""
        state = ArchitectureState(
            layers=[
                LayerSpec(
                    block_type=OperatorBlockType.MLP_LAYER,
                    width=250,
                ),
            ],
            max_width=256,
            min_width=16,
        )
        action = NASAction(
            action_type=NASActionType.ADJUST_WIDTH,
            layer_index=0,
            params={"delta": 32},
        )
        new_state = nas.apply_action(state, action)
        assert new_state.layers[0].width <= state.max_width


class TestNASAdjustModes:
    """Applying ADJUST_MODES modifies the Fourier mode count."""

    def test_nas_adjust_modes(
        self,
        nas: NeuralOperatorNAS,
        sample_state: ArchitectureState,
    ) -> None:
        original_modes = sample_state.layers[0].num_modes  # 12

        # Increase modes
        action_up = NASAction(
            action_type=NASActionType.ADJUST_MODES,
            layer_index=0,
            params={"delta": 4},
        )
        state_up = nas.apply_action(sample_state, action_up)
        assert state_up.layers[0].num_modes == original_modes + 4

        # Decrease modes
        action_down = NASAction(
            action_type=NASActionType.ADJUST_MODES,
            layer_index=0,
            params={"delta": -4},
        )
        state_down = nas.apply_action(sample_state, action_down)
        assert state_down.layers[0].num_modes == original_modes - 4

    def test_modes_clamped_to_bounds(
        self,
        nas: NeuralOperatorNAS,
    ) -> None:
        """Modes cannot exceed max_modes or go below mode_step."""
        state = ArchitectureState(
            layers=[
                LayerSpec(
                    block_type=OperatorBlockType.FOURIER_LAYER,
                    width=64,
                    num_modes=60,
                ),
            ],
            max_modes=64,
        )
        action = NASAction(
            action_type=NASActionType.ADJUST_MODES,
            layer_index=0,
            params={"delta": 8},
        )
        new_state = nas.apply_action(state, action)
        assert new_state.layers[0].num_modes <= state.max_modes


class TestNASAddSkip:
    """Applying ADD_SKIP_CONNECTION sets skip_to on the target layer."""

    def test_nas_add_skip(
        self,
        nas: NeuralOperatorNAS,
        sample_state: ArchitectureState,
    ) -> None:
        # Connect layer 0 to layer 2 via skip connection
        action = NASAction(
            action_type=NASActionType.ADD_SKIP_CONNECTION,
            layer_index=0,
            params={"target_index": 2},
        )
        new_state = nas.apply_action(sample_state, action)

        assert new_state.layers[2].skip_to == 0
        # Original state is unmodified
        assert sample_state.layers[2].skip_to == -1


# ------------------------------------------------------------------
# NAS search tests
# ------------------------------------------------------------------


class TestNASSearchReturnsArchitecture:
    """NeuralOperatorNAS.search returns a valid architecture."""

    def test_nas_search_returns_architecture(self) -> None:
        nas = NeuralOperatorNAS(
            max_layers=6,
            max_width=128,
            min_width=16,
            num_simulations=1,
            width_step=16,
            mode_step=4,
            complexity_penalty=1e-6,
        )
        result = nas.search(max_steps=3)

        assert isinstance(result, list)
        assert len(result) > 0
        for layer in result:
            assert isinstance(layer, LayerSpec)
            assert isinstance(layer.block_type, OperatorBlockType)

    def test_search_with_eval_fn(self) -> None:
        """Search with a custom evaluation function."""
        nas = NeuralOperatorNAS(
            max_layers=6,
            max_width=128,
            min_width=16,
            num_simulations=1,
            width_step=16,
            mode_step=4,
            complexity_penalty=1e-6,
        )

        call_count = 0

        def dummy_eval(layers: list[LayerSpec]) -> tuple[float, float]:
            nonlocal call_count
            call_count += 1
            # Lower error for smaller architectures
            error = len(layers) * 0.1
            cost = sum(l.param_count_estimate for l in layers) * 1e-6
            return error, cost

        result = nas.search(eval_fn=dummy_eval, max_steps=2)
        assert isinstance(result, list)
        assert len(result) > 0
        assert call_count > 0

    def test_search_with_initial_architecture(self) -> None:
        """Search starting from a given architecture."""
        nas = NeuralOperatorNAS(
            max_layers=6,
            max_width=128,
            min_width=16,
            num_simulations=1,
            width_step=16,
            mode_step=4,
            complexity_penalty=1e-6,
        )
        initial = [
            LayerSpec(block_type=OperatorBlockType.FNET_MIXING, width=32),
        ]
        result = nas.search(initial_architecture=initial, max_steps=2)
        assert isinstance(result, list)
        assert len(result) >= 1


# ------------------------------------------------------------------
# Default initial architecture
# ------------------------------------------------------------------


class TestNASDefaultInitial:
    """NeuralOperatorNAS._default_initial_architecture creates a valid start."""

    def test_nas_default_initial(self) -> None:
        nas = NeuralOperatorNAS()
        arch = nas._default_initial_architecture()

        assert isinstance(arch, list)
        assert len(arch) == 3

        # Check expected block types
        block_types = [l.block_type for l in arch]
        assert OperatorBlockType.FOURIER_LAYER in block_types
        assert OperatorBlockType.GALERKIN_ATTENTION in block_types
        assert OperatorBlockType.MLP_LAYER in block_types

        # All layers should have valid specs
        for layer in arch:
            assert layer.width > 0
            assert isinstance(layer.block_type, OperatorBlockType)


# ------------------------------------------------------------------
# Enum completeness
# ------------------------------------------------------------------


class TestOperatorBlockTypesEnum:
    """OperatorBlockType enum has all expected members."""

    def test_operator_block_types_enum(self) -> None:
        expected = {
            "fourier_layer",
            "deeponet_branch",
            "deeponet_trunk",
            "wavelet_layer",
            "galerkin_attention",
            "fnet_mixing",
            "mlp_layer",
            "residual_block",
            "skip_connection",
        }
        actual = {member.value for member in OperatorBlockType}
        assert actual == expected

    def test_nas_action_types_enum(self) -> None:
        expected = {
            "add_layer",
            "remove_layer",
            "change_layer_type",
            "adjust_width",
            "adjust_modes",
            "add_skip_connection",
            "toggle_residual",
            "no_op",
        }
        actual = {member.value for member in NASActionType}
        assert actual == expected

    def test_operator_block_type_is_str_enum(self) -> None:
        """OperatorBlockType members are strings."""
        for member in OperatorBlockType:
            assert isinstance(member, str)
            assert isinstance(member.value, str)

    def test_nas_action_type_is_str_enum(self) -> None:
        """NASActionType members are strings."""
        for member in NASActionType:
            assert isinstance(member, str)
            assert isinstance(member.value, str)
