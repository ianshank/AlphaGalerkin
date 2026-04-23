# ADR — Mouse-Droid-AGI Fusion Head Integration

- Status: Accepted
- Date: 2026-04-23
- Related plan: `docs/GALERKIN_FUSION_HEAD_PLAN.md`

## Context

`Mouse-Droid-AGI` is implementing a Galerkin-attention sensor fusion block
(`sensing/galerkin_fusion.py`) that consumes operators from this repository
via a git submodule. The 2-week plan (`docs/GALERKIN_FUSION_HEAD_PLAN.md`)
schedules the integration on a fixed timeline and relies on the
constructor signatures of a small set of public classes remaining stable
for the duration of the experiment.

This ADR records that contract so it cannot be silently broken by a
refactor on this side.

## Decision

For the duration of the integration window (planned: 2 weeks from
2026-04-23, extendable on agreement), the following classes are part of
the **stable public surface** of `src.modeling`:

| Class | Module | Importable as |
|-------|--------|---------------|
| `GalerkinAttention` | `src/modeling/attention.py` | `from src.modeling import GalerkinAttention` |
| `SoftmaxAttention` | `src/modeling/attention.py` | `from src.modeling import SoftmaxAttention` |
| `HybridAttention` | `src/modeling/attention.py` | `from src.modeling import HybridAttention` |
| `FNetBlock` | `src/modeling/fnet.py` | `from src.modeling import FNetBlock` |
| `FNetMixing` | `src/modeling/fnet.py` | `from src.modeling import FNetMixing` |
| `FNetStack` | `src/modeling/fnet.py` | `from src.modeling import FNetStack` |
| `GalerkinFNetHybrid` | `src/modeling/fnet.py` | `from src.modeling import GalerkinFNetHybrid` |
| `MultiScaleFourierFeatures` | `src/modeling/multiscale_fourier.py` | `from src.modeling import MultiScaleFourierFeatures` |
| `AdaptiveFourierFeatures` | `src/modeling/multiscale_fourier.py` | `from src.modeling import AdaptiveFourierFeatures` |
| `ProgressiveFourierFeatures` | `src/modeling/multiscale_fourier.py` | `from src.modeling import ProgressiveFourierFeatures` |
| `PositionalEncoding` | `src/modeling/multiscale_fourier.py` | `from src.modeling import PositionalEncoding` |
| `SpatialPositionalEncoding` | `src/modeling/multiscale_fourier.py` | `from src.modeling import SpatialPositionalEncoding` |
| `FourierFeaturesConfig` | `src/modeling/multiscale_fourier.py` | `from src.modeling import FourierFeaturesConfig` |
| `StabilityGuard` | `src/modeling/stability.py` | `from src.modeling import StabilityGuard` |
| `StableGalerkinInitializer` | `src/modeling/stability.py` | `from src.modeling import StableGalerkinInitializer` |
| `AlphaGalerkinModel` | `src/modeling/model.py` | `from src.modeling import AlphaGalerkinModel` |
| `ContinuousEmbedding` | `src/modeling/embeddings.py` | `from src.modeling import ContinuousEmbedding` |
| `FourierFeatures` | `src/modeling/embeddings.py` | `from src.modeling import FourierFeatures` |

The last three classes pre-date this ADR and were already exported by
`src/modeling/__init__.py`; they are included in the stable surface so
that existing downstream code is not broken by the fusion-head work.

### Stability rules

1. **Constructor signatures**: The order, names, and default values of
   existing parameters must not change. New parameters may be added with
   defaults that preserve current behaviour (additive, backwards
   compatible only).
2. **Forward signatures**: Input/output tensor shapes and dtypes for the
   `forward()` of each class must remain stable.
3. **Re-export surface**: `src/modeling/__init__.py` `__all__` must keep
   listing the classes above. Removal requires bumping this ADR.
4. **Submodule pin**: Mouse-Droid-AGI pins to a specific commit SHA on
   `main`. Promotions to a newer SHA are explicit and reviewed.

### Key signatures (frozen)

```python
GalerkinAttention(
    d_model: int,
    n_heads: int,
    d_key: int | None = None,
    d_value: int | None = None,
    dropout: float = 0.0,
    normalize_features: bool = True,
)
# forward(x: [B, n, d], return_lbb: bool = False) -> [B, n, d] or ([B, n, d], [B])

FNetBlock(
    d_model: int,
    d_ffn: int | None = None,
    dropout: float = 0.1,
    use_2d_fft: bool = True,
)
# forward(x: [B, n, d], board_size: int | None = None) -> [B, n, d]

MultiScaleFourierFeatures(
    input_dim: int,
    config: FourierFeaturesConfig | None = None,
    n_features: int = 128,
    scales: list[float] | None = None,
    learnable: bool = True,
    include_input: bool = True,
)
# forward(x: [..., d]) -> [..., features]

StabilityGuard(
    beta_threshold: float = 1e-6,
    regularization_strength: float = 0.01,
    log_interval: int = 100,
    margin_multiplier: float = 10.0,  # added 2026-04-23, default preserves prior behaviour
)
# regularization_loss(keys, multihead=False) -> scalar
# check_stability(keys, multihead=False) -> (bool, [B])
```

## Consequences

- Any AlphaGalerkin PR that touches the modules above is required to
  preserve the signatures listed here. CI enforces ruff and `mypy
  --strict` on those files; coverage on `src/modeling/` must stay
  ≥85%.
- Non-additive signature changes during the integration window require
  either (a) a coordinated PR on Mouse-Droid-AGI updating the submodule
  pin and consumer code, or (b) bumping this ADR.
- After the Day-10 decision gate (see plan §8), this ADR will be
  superseded by `ADR-post-fusion-direction.md`.

## Verification

```bash
# Re-exports stable
python -c "from src.modeling import (
    GalerkinAttention, SoftmaxAttention, HybridAttention,
    FNetBlock, FNetMixing, FNetStack, GalerkinFNetHybrid,
    MultiScaleFourierFeatures, AdaptiveFourierFeatures, ProgressiveFourierFeatures,
    PositionalEncoding, SpatialPositionalEncoding, FourierFeaturesConfig,
    StabilityGuard, StableGalerkinInitializer,
)"

# Type & lint contracts
mypy src/modeling/ --strict --ignore-missing-imports
ruff check src/modeling/

# Coverage contract
pytest tests/modeling/ --cov=src.modeling --cov-fail-under=85
```
