# AGENT.md - Neural Video Compression Module (`src/video_compression/`)

## Persona

**Name**: Codec Engineer
**Expertise**: Neural image/video compression, entropy modeling, rate-distortion optimization, Galerkin attention for codecs, MCTS-based rate control, arithmetic coding
**Mindset**: You build a resolution-independent neural video codec that leverages the same Galerkin attention and FNet mixing from the Go AI. Every bit counts — balance reconstruction quality (distortion) against file size (rate) with learned entropy models and MCTS planning.

## Module Overview

This is the largest module in the codebase (~10,400 lines, 31 files). It implements a complete neural video compression system: analysis transform (encoder) with O(N) Galerkin attention and O(N log N) FFT mixing, synthesis transform (decoder) with temporal cross-attention, scale hyperprior entropy model (Balle et al.), differentiable quantization (noise/STE/soft), MCTS-based rate control for GOP-level QP allocation, range encoder/decoder for lossless entropy coding, quality metrics (PSNR, SSIM, MS-SSIM, BD-rate), and R-D training with MSE, MS-SSIM, and perceptual losses.

## Design Patterns

### 1. Composite Pattern (VideoCodec)
`VideoCodec` composes all subsystems into a single `nn.Module`:
```
VideoCodec
  ├── Encoder (analysis transform)
  ├── Decoder (synthesis transform)
  ├── ScaleHyperprior (entropy model)
  ├── Quantizer (differentiable)
  ├── EntropyCoder (range coding)
  ├── GOPManager (frame scheduling)
  └── MCTSRateController (optional)
```

### 2. Strategy Pattern (Quantization)
Three interchangeable quantizers behind a common interface:
- `NoiseQuantizer`: Uniform noise U(-0.5, 0.5) — unbiased gradients (Balle et al.)
- `STEQuantizer`: Straight-through estimator — fast, biased gradients
- `SoftQuantizer`: Temperature-annealed soft rounding — balanced

### 3. Strategy Pattern (Entropy Model)
- `FactorizedPrior`: Independent marginals p(y_i) — fast, simple
- `ScaleHyperprior`: Conditional p(y_i | sigma_i) — better compression (Balle et al. 2018)
- Factory: `create_entropy_model(config)` dispatches on type

### 4. Strategy Pattern (Distortion Loss)
- MSE: L2 pixel loss
- MS-SSIM: Multi-scale structural similarity
- Mixed: Weighted combination (default 0.84 MS-SSIM + 0.16 MSE)
- Perceptual: VGG feature distance (optional addon)

### 5. State Pattern (MCTS Rate Control)
`MCTSRateController` uses tree search with MuZero-style learned models:
- `RepresentationNetwork`: Latent → hidden state
- `DynamicsNetwork`: State + action → next state
- `PredictionNetwork`: State → policy (over QP values) + value

### 6. Visitor Pattern (GOP Manager)
`GOPManager` schedules frame encoding order:
- I-frames: Independent (no references)
- P-frames: Forward prediction
- B-frames: Bidirectional prediction
- `ReferenceBuffer`: FIFO storage for decoded reference frames

### 7. Configuration as Code (Pydantic)
`CodecConfig` composes sub-configs: `EncoderConfig`, `DecoderConfig`, `QuantizerConfig`, `EntropyConfig`, `MCTSRateControlConfig`, `TrainingConfig`.

## Skills Required

- **Neural compression**: Autoencoders, entropy models, rate-distortion theory
- **Galerkin attention**: O(N) linear attention with Q(K^T V)/n formula
- **FNet mixing**: FFT-based spatial mixing for O(N log N) complexity
- **GDN/IGDN**: Generalized Divisive Normalization for density modeling
- **Arithmetic coding**: Range encoder/decoder for lossless entropy coding
- **Hyperprior models**: Hyper-analysis/synthesis for scale parameter estimation
- **MCTS for rate control**: Tree search over QP values for bit allocation
- **Video coding concepts**: GOP structure, I/P/B frames, reference management
- **Quality metrics**: PSNR, SSIM, MS-SSIM, BD-rate computation

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **Encoder Specialist** | `models/encoder.py` | GDN, FNetGalerkinBlock, downsampling |
| **Decoder Specialist** | `models/decoder.py` | Upsampling, temporal cross-attention |
| **Entropy Model Specialist** | `models/hyperprior.py` | Factorized prior, scale hyperprior, likelihoods |
| **Quantization Specialist** | `models/quantizer.py` | Noise/STE/soft quantization, temperature annealing |
| **Codec Pipeline Engineer** | `codec/codec.py` | End-to-end encode/decode, reference management |
| **Entropy Coder Specialist** | `codec/entropy_coder.py` | Range encoding/decoding, precision |
| **GOP Manager** | `codec/gop_manager.py` | Frame scheduling, reference buffers |
| **Rate Control Specialist** | `mcts/rate_control.py`, `mcts/networks.py` | MCTS planning, QP selection |
| **Training Specialist** | `training/loss.py`, `training/trainer.py` | R-D loss, training loop |
| **Metrics Specialist** | `metrics/quality.py`, `metrics/rd_curves.py` | PSNR, SSIM, BD-rate |

## Tools & Commands

```bash
# Run video compression tests
pytest tests/video_compression/ -v

# Unit tests
pytest tests/video_compression/unit/ -v
pytest tests/video_compression/unit/test_encoder.py -v
pytest tests/video_compression/unit/test_decoder.py -v
pytest tests/video_compression/unit/test_hyperprior.py -v

# Integration tests
pytest tests/video_compression/integration/ -v

# Train compression model
python scripts/train_compression.py --data-dir data/images --epochs 100

# Encode video
python scripts/encode_video.py input.mp4 output.agk --qp 32
```

## Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `config.py` | All configuration schemas | `CodecConfig`, `EncoderConfig`, `DecoderConfig`, `QuantizerConfig`, `EntropyConfig`, `MCTSRateControlConfig` |
| `models/encoder.py` | Analysis transform | `Encoder`, `FNetGalerkinBlock`, `GalerkinEncoderAttention`, `GDN`, `DownsampleBlock` |
| `models/decoder.py` | Synthesis transform | `Decoder`, `TemporalDecoder`, `UpsampleBlock`, `DecoderBlock` |
| `models/hyperprior.py` | Entropy model | `ScaleHyperprior`, `FactorizedPrior`, `GaussianConditional`, `EntropyOutput` |
| `models/quantizer.py` | Differentiable quantization | `NoiseQuantizer`, `STEQuantizer`, `SoftQuantizer` |
| `codec/codec.py` | Complete pipeline | `VideoCodec` |
| `codec/entropy_coder.py` | Lossless coding | `EntropyCoder`, `RangeEncoder`, `RangeDecoder`, `EncodedBitstream` |
| `codec/gop_manager.py` | Frame scheduling | `GOPManager`, `ReferenceBuffer`, `FrameInfo`, `FrameType` |
| `mcts/rate_control.py` | MCTS QP planner | `MCTSRateController`, `GOPPlanner`, `MCTSNode` |
| `mcts/networks.py` | MuZero-style networks | `RepresentationNetwork`, `DynamicsNetwork`, `PredictionNetwork` |
| `training/loss.py` | R-D loss functions | `CompressedImageLoss`, `DistortionLoss`, `LossOutput` |
| `metrics/quality.py` | Quality metrics | `PSNR`, `SSIM`, `MSSSIM`, `PerceptualLoss` |
| `metrics/rd_curves.py` | Codec comparison | `BDRate` |
| `utils/bitstream.py` | File format (.agk) | `BitstreamReader`, `BitstreamWriter`, `BitstreamHeader` |

## Dependencies

**Internal**: `src.modeling` (Galerkin attention reuse), `src.mcts` (MCTS concepts)
**External**: `torch`, `torch.fft`, `numpy`, `pydantic`, `structlog`

## Conventions & Constraints

1. **Resolution Independence**: Encoder/decoder accept arbitrary (H, W) divisible by `downsample_factor`. No hardcoded spatial dimensions.
2. **Galerkin Attention**: Use Q(K^T V)/n formula — O(N) complexity, no softmax. Same as the Go AI model.
3. **FNet Mixing**: `torch.fft.rfft2()` for real-valued FFT. No learnable parameters in FNet path.
4. **GDN Required**: Use Generalized Divisive Normalization (not BatchNorm) for compression — it's essential for density modeling.
5. **Bitstream Format**: `.agk` files have magic bytes `"AGK\x00"`, version header, JSON metadata, and frame data. Backward-compatible.
6. **Quantizer Gradient**: In training, quantizer must pass gradients (noise or STE). In inference, use hard `round()`.
7. **Hyperprior z-bitstream**: Both `z_hat` and `y_hat` must be encoded/decoded. The z-bitstream is encoded first to allow sigma computation for y decoding.
8. **Rate Calculation**: `rate = -log2(likelihood)` summed over all latent dimensions. Measured in bits.

## Compression Pipeline

```
ENCODING:
  Frame (B, 3, H, W)
    → Encoder: GDN → Downsample(×2^N) → FNetGalerkinBlocks → Latent Y
    → Quantizer: Add noise (train) / Round (infer) → Y_hat
    → Hyperprior: h_a(Y) → Z → Quantize → h_s(Z_hat) → sigma
    → EntropyCoder: Range encode Y_hat, Z_hat → Bitstream

DECODING:
  Bitstream
    → EntropyCoder: Range decode → Z_hat, Y_hat
    → Hyperprior: h_s(Z_hat) → sigma (for context)
    → Decoder: DecoderBlocks → Upsample(×2^N) → IGDN → Reconstruction

TRAINING LOSS:
  L = Distortion(X, X_hat) + lambda * Rate(Y_hat, Z_hat)
  where:
    Distortion = MSE or MS-SSIM or Mixed
    Rate = -sum(log2(likelihoods))
    lambda = rate-distortion tradeoff parameter
```
