"""Tests that ``src/modeling/`` default arguments really come from ``src/constants.py``.

``FNetBlock``/``FNetStack``/``GalerkinFNetHybrid`` (dropout) and
``StabilityGuard`` (``beta_threshold``) had their literal defaults replaced by
imports from the central constants module. Two failure modes need catching:

1. **Wrong constant wired.** ``DEFAULT_LBB_TARGET`` (0.1) and
   ``DEFAULT_LBB_THRESHOLD`` (1e-6) are both "LBB" constants; picking the wrong
   one silently changes the stability gate by five orders of magnitude.
   ``tests/test_constants.py`` only asserts *ranges*
   (``0 < DEFAULT_LBB_THRESHOLD < 1e-3``, ``0 < DEFAULT_DROPOUT < 1``), which a
   wrong-but-plausible value passes.
2. **Silently unwired.** A revert to the literal ``0.1`` / ``1e-6`` keeps every
   value-based assertion green. The tests below therefore also assert *object
   identity* between each signature default and the constant: a literal in the
   defining module compiles to a distinct ``float`` object, so ``is`` fails
   there while the imported-constant form passes.

Validates:
    - Exact values of the two constants (the AQA pin).
    - Constructed instances use the constant's value.
    - Signature defaults are the constant object itself (wiring is live).
    - Explicit arguments still override the default.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import src.constants as C
from src.modeling.fnet import FNetBlock, FNetStack, GalerkinFNetHybrid
from src.modeling.stability import StabilityGuard


class TestConstantValuesArePinned:
    """Exact values, not just ranges — a wrong constant must fail here."""

    def test_default_dropout_value(self) -> None:
        """FNet's shared dropout default."""
        assert C.DEFAULT_DROPOUT == 0.1

    def test_default_lbb_threshold_value(self) -> None:
        """Minimum singular value for the LBB (inf-sup) stability check."""
        assert C.DEFAULT_LBB_THRESHOLD == 1e-6

    def test_lbb_threshold_and_target_are_distinct(self) -> None:
        """The two similarly-named LBB constants must not be interchangeable."""
        assert C.DEFAULT_LBB_THRESHOLD != C.DEFAULT_LBB_TARGET
        assert C.DEFAULT_LBB_THRESHOLD < C.DEFAULT_LBB_TARGET


class TestFNetDropoutWiring:
    """``src/modeling/fnet.py`` dropout defaults."""

    def test_fnet_block_default_dropout_matches_constant(self) -> None:
        """Both the residual dropout and the FFN dropouts use the constant."""
        block = FNetBlock(d_model=16)
        assert block.dropout.p == C.DEFAULT_DROPOUT
        ffn_dropouts = [m.p for m in block.ffn if isinstance(m, torch.nn.Dropout)]
        assert ffn_dropouts == [C.DEFAULT_DROPOUT, C.DEFAULT_DROPOUT]

    def test_fnet_stack_default_dropout_matches_constant(self) -> None:
        """The stack propagates the same default into every block."""
        stack = FNetStack(d_model=16, n_layers=2)
        assert [layer.dropout.p for layer in stack.layers] == [C.DEFAULT_DROPOUT] * 2

    def test_hybrid_default_dropout_matches_constant(self) -> None:
        """The Galerkin/FNet hybrid shares the same default."""
        hybrid = GalerkinFNetHybrid(d_model=16, n_heads=2)
        assert hybrid.fnet.dropout.p == C.DEFAULT_DROPOUT

    def test_explicit_dropout_overrides_the_constant(self) -> None:
        """An explicit argument still wins over the module default."""
        assert FNetBlock(d_model=16, dropout=0.42).dropout.p == pytest.approx(0.42)

    @pytest.mark.parametrize("cls", [FNetBlock, FNetStack, GalerkinFNetHybrid])
    def test_signature_default_is_the_constant_object(self, cls: type) -> None:
        """The default *is* ``C.DEFAULT_DROPOUT``, not an equal literal.

        This is what distinguishes "imported from ``src.constants``" from a
        literal ``0.1`` that happens to equal it today: a literal in
        ``fnet.py`` compiles to that module's own ``float`` object, so the
        identity check fails while the value check would not.
        """
        default = inspect.signature(cls.__init__).parameters["dropout"].default
        assert default is C.DEFAULT_DROPOUT


class TestStabilityGuardThresholdWiring:
    """``src/modeling/stability.py`` LBB threshold default."""

    def test_default_beta_threshold_matches_constant(self) -> None:
        """The guard's inf-sup floor is the central constant."""
        assert StabilityGuard().beta_threshold == C.DEFAULT_LBB_THRESHOLD

    def test_default_is_not_the_lbb_target(self) -> None:
        """Guards against wiring the wrong LBB constant."""
        assert StabilityGuard().beta_threshold != C.DEFAULT_LBB_TARGET

    def test_explicit_threshold_overrides_the_constant(self) -> None:
        """An explicit argument still wins over the module default."""
        assert StabilityGuard(beta_threshold=0.25).beta_threshold == pytest.approx(0.25)

    def test_signature_default_is_the_constant_object(self) -> None:
        """The default *is* ``C.DEFAULT_LBB_THRESHOLD``, not an equal literal."""
        default = inspect.signature(StabilityGuard.__init__).parameters["beta_threshold"].default
        assert default is C.DEFAULT_LBB_THRESHOLD
