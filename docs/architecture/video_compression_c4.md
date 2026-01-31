# AlphaGalerkin Video Compression - C4 Architecture

## Overview

This document describes the architecture of the AlphaGalerkin Neural Video Compression system using the C4 model (Context, Containers, Components, Code).

**Key Design Principles:**
- Resolution-independent: O(N) Galerkin attention + O(N log N) FNet mixing
- Zero-shot transfer: Train on one resolution, infer on another
- MCTS-based rate control for optimal bit allocation
- Differentiable end-to-end training with R-D optimization

---

## Level 1: System Context Diagram

Shows how the video compression system fits into its environment.

```mermaid
C4Context
    title System Context Diagram - AlphaGalerkin Video Compression

    Person(user, "User", "Provides video/images for compression")
    Person(researcher, "ML Researcher", "Trains and evaluates models")

    System(videocodec, "AlphaGalerkin Video Codec", "Neural video compression with Galerkin attention and FNet mixing")

    System_Ext(storage, "Storage System", "Video files, compressed bitstreams")
    System_Ext(gpu, "GPU Cluster", "Training infrastructure")
    System_Ext(wandb, "W&B / TensorBoard", "Experiment tracking")

    Rel(user, videocodec, "Encodes/Decodes video")
    Rel(researcher, videocodec, "Trains models, evaluates R-D curves")
    Rel(videocodec, storage, "Reads/writes video and bitstreams")
    Rel(videocodec, gpu, "Runs inference and training")
    Rel(videocodec, wandb, "Logs metrics and checkpoints")
```

---

## Level 2: Container Diagram

Shows the high-level modules within the video compression system.

```mermaid
C4Container
    title Container Diagram - Video Compression System

    Person(user, "User")

    Container_Boundary(codec_system, "Video Compression System") {
        Container(codec, "Video Codec", "Python/PyTorch", "Main encode/decode pipeline with GOP management")
        Container(models, "Neural Models", "Python/PyTorch", "Encoder, Decoder, Entropy Model, Quantizer")
        Container(mcts, "MCTS Rate Control", "Python/PyTorch", "MuZero-style QP selection")
        Container(training, "Training Pipeline", "Python/PyTorch", "R-D loss, trainer, checkpoints")
        Container(metrics, "Quality Metrics", "Python/PyTorch", "PSNR, SSIM, MS-SSIM, BD-rate")
        Container(data, "Data Pipeline", "Python/PyTorch", "Video/image datasets, transforms")
        Container(utils, "Utilities", "Python", "Bitstream I/O, padding, logging")
    }

    ContainerDb(config, "Configuration", "Pydantic", "Codec, training, and experiment configs")

    Rel(user, codec, "encode_video() / decode_video()")
    Rel(codec, models, "Uses encoder, decoder, entropy model")
    Rel(codec, mcts, "Queries for QP decisions")
    Rel(training, codec, "Trains end-to-end")
    Rel(training, metrics, "Computes R-D metrics")
    Rel(training, data, "Loads training batches")
    Rel(codec, utils, "Bitstream serialization, padding")
    Rel(codec, config, "Reads configuration")
```

---

## Level 3: Component Diagrams

### 3.1 Neural Models Component

```mermaid
C4Component
    title Component Diagram - Neural Models (src/video_compression/models/)

    Container_Boundary(models, "Neural Models") {
        Component(encoder, "Encoder", "nn.Module", "Analysis transform: RGB → Latent<br/>FNet + Galerkin attention blocks<br/>4-stage 16x downsampling")

        Component(decoder, "Decoder", "nn.Module", "Synthesis transform: Latent → RGB<br/>Inverse of encoder<br/>4-stage 16x upsampling")

        Component(temporal, "TemporalDecoder", "nn.Module", "Temporal cross-attention<br/>for P/B frame references")

        Component(quantizer, "Quantizer", "nn.Module", "Differentiable quantization<br/>Noise / STE / Soft modes")

        Component(entropy, "HyperpriorEntropyModel", "nn.Module", "Scale hyperprior (Ballé)<br/>Rate estimation via CDFs")

        Component(gdn, "GDN/IGDN", "nn.Module", "Generalized Divisive<br/>Normalization layers")

        Component(fnet, "FNetGalerkinBlock", "nn.Module", "Hybrid FFT + Galerkin<br/>attention mixing")
    }

    Rel(encoder, fnet, "Uses for spatial mixing")
    Rel(encoder, gdn, "Uses for normalization")
    Rel(decoder, gdn, "Uses inverse GDN")
    Rel(temporal, decoder, "Extends with cross-attention")
    Rel(entropy, quantizer, "Quantizes before entropy coding")
```

### 3.2 Codec Component

```mermaid
C4Component
    title Component Diagram - Video Codec (src/video_compression/codec/)

    Container_Boundary(codec, "Video Codec") {
        Component(videocodec, "VideoCodec", "nn.Module", "Main orchestrator<br/>Combines all components")

        Component(gop, "GOPManager", "Class", "Frame scheduling (I/P/B)<br/>Reference buffer management")

        Component(entropy_coder, "EntropyCoder", "Class", "Range encoder/decoder<br/>Lossless bitstream coding")

        Component(ref_buffer, "ReferenceBuffer", "Dataclass", "Stores decoded frames<br/>and latents for P/B refs")
    }

    Component_Ext(encoder, "Encoder", "Neural model")
    Component_Ext(decoder, "Decoder", "Neural model")
    Component_Ext(entropy_model, "EntropyModel", "Neural model")
    Component_Ext(mcts, "MCTSRateController", "Rate control")

    Rel(videocodec, encoder, "encode_frame()")
    Rel(videocodec, decoder, "decode_frame()")
    Rel(videocodec, entropy_model, "compress() / decompress()")
    Rel(videocodec, entropy_coder, "encode() / decode()")
    Rel(videocodec, gop, "get_frame_info()")
    Rel(videocodec, ref_buffer, "add() / get()")
    Rel(videocodec, mcts, "select_qp()")
```

### 3.3 MCTS Rate Control Component

```mermaid
C4Component
    title Component Diagram - MCTS Rate Control (src/video_compression/mcts/)

    Container_Boundary(mcts, "MCTS Rate Control") {
        Component(controller, "MCTSRateController", "Class", "Tree search for QP selection<br/>UCB-based node selection")

        Component(planner, "GOPPlanner", "Class", "GOP-level bit allocation<br/>Frame type weighting")

        Component(repr_net, "RepresentationNetwork", "nn.Module", "Latent → hidden state<br/>Adaptive pooling + MLP")

        Component(dyn_net, "DynamicsNetwork", "nn.Module", "state + action → next_state<br/>Reward prediction")

        Component(pred_net, "PredictionNetwork", "nn.Module", "state → policy + value<br/>Categorical value distribution")

        Component(node, "MCTSNode", "Dataclass", "Tree node with UCB score<br/>Visit counts, value sum")
    }

    Rel(controller, repr_net, "Initial state encoding")
    Rel(controller, dyn_net, "Tree expansion")
    Rel(controller, pred_net, "Policy/value prediction")
    Rel(controller, node, "Tree structure")
    Rel(controller, planner, "GOP planning")
```

### 3.4 Training Component

```mermaid
C4Component
    title Component Diagram - Training Pipeline (src/video_compression/training/)

    Container_Boundary(training, "Training Pipeline") {
        Component(trainer, "VideoCompressionTrainer", "Class", "Training loop orchestration<br/>Checkpoint management")

        Component(rd_loss, "RDLoss", "nn.Module", "L = D + λR<br/>Rate-distortion tradeoff")

        Component(comp_loss, "CompressionLoss", "nn.Module", "R-D + perceptual loss<br/>VGG feature distance")

        Component(dist_loss, "DistortionLoss", "nn.Module", "MSE / MS-SSIM / Mixed<br/>Distortion metrics")
    }

    Component_Ext(codec, "VideoCodec", "Main codec")
    Component_Ext(dataset, "Dataset", "Data loading")
    Component_Ext(optimizer, "AdamW", "Optimizer")
    Component_Ext(scheduler, "CosineAnnealingLR", "LR schedule")

    Rel(trainer, codec, "compute_rd_loss()")
    Rel(trainer, comp_loss, "Forward pass")
    Rel(trainer, dataset, "Load batches")
    Rel(trainer, optimizer, "Gradient updates")
    Rel(trainer, scheduler, "LR scheduling")
    Rel(comp_loss, rd_loss, "Rate-distortion term")
    Rel(comp_loss, dist_loss, "Distortion term")
```

---

## Level 4: Code Diagrams

### 4.1 Encoding Flow Sequence

```mermaid
sequenceDiagram
    participant User
    participant VideoCodec
    participant GOPManager
    participant Encoder
    participant Quantizer
    participant EntropyModel
    participant EntropyCoder
    participant MCTSController
    participant ReferenceBuffer

    User->>VideoCodec: encode_frame(frame, frame_info)

    VideoCodec->>GOPManager: get_frame_info(idx)
    GOPManager-->>VideoCodec: FrameInfo(type, refs)

    alt MCTS enabled
        VideoCodec->>MCTSController: select_qp(latent)
        MCTSController-->>VideoCodec: RateControlDecision(qp)
    end

    VideoCodec->>Encoder: forward(frame)
    Encoder-->>VideoCodec: latent y

    VideoCodec->>Quantizer: forward(y * qp_scale)
    Quantizer-->>VideoCodec: y_hat

    VideoCodec->>EntropyModel: compress(y_hat)
    EntropyModel-->>VideoCodec: {symbols, scales}

    VideoCodec->>EntropyCoder: encode(symbols, scales)
    EntropyCoder-->>VideoCodec: bitstream

    VideoCodec->>VideoCodec: decode internally
    VideoCodec->>ReferenceBuffer: add(idx, reconstructed)

    VideoCodec-->>User: CodecOutput(bitstream, rate, distortion)
```

### 4.2 Training Flow Sequence

```mermaid
sequenceDiagram
    participant Trainer
    participant DataLoader
    participant VideoCodec
    participant CompressionLoss
    participant Optimizer

    loop Each Epoch
        loop Each Batch
            DataLoader->>Trainer: batch_images

            Trainer->>VideoCodec: forward(batch)
            VideoCodec-->>Trainer: reconstructed, rate

            Trainer->>CompressionLoss: forward(recon, original, rate)
            CompressionLoss-->>Trainer: LossOutput(total, components)

            Trainer->>Trainer: loss.backward()
            Trainer->>Optimizer: step()
            Trainer->>Optimizer: zero_grad()
        end

        Trainer->>Trainer: save_checkpoint()
    end
```

### 4.3 Class Relationships

```mermaid
classDiagram
    class VideoCodec {
        +encoder: Encoder
        +decoder: Decoder
        +temporal_decoder: TemporalDecoder
        +quantizer: Quantizer
        +entropy_model: EntropyModel
        +entropy_coder: EntropyCoder
        +gop_manager: GOPManager
        +rate_controller: MCTSRateController
        +forward(x, reference)
        +encode_frame(frame, info)
        +decode_frame(bitstream, info)
        +encode_video(frames)
        +decode_video(bitstreams)
    }

    class Encoder {
        +config: EncoderConfig
        +initial_conv: Conv2d
        +encoder_blocks: ModuleList
        +output_conv: Conv2d
        +forward(x) Tensor
    }

    class Decoder {
        +config: DecoderConfig
        +initial_conv: Conv2d
        +decoder_blocks: ModuleList
        +output_conv: Conv2d
        +forward(y) Tensor
    }

    class HyperpriorEntropyModel {
        +hyper_analysis: HyperAnalysis
        +hyper_synthesis: HyperSynthesis
        +hyperprior: FactorizedPrior
        +gaussian: GaussianConditional
        +forward(y) EntropyOutput
        +compress(y) dict
        +decompress(symbols) Tensor
    }

    class GOPManager {
        +gop_size: int
        +use_b_frames: bool
        +reference_buffer: ReferenceBuffer
        +get_frame_info(idx) FrameInfo
        +get_encoding_order(gop) list
        +reset()
    }

    class MCTSRateController {
        +config: MCTSRateControlConfig
        +repr_net: RepresentationNetwork
        +dynamics_net: DynamicsNetwork
        +pred_net: PredictionNetwork
        +select_qp(latent) RateControlDecision
    }

    VideoCodec --> Encoder
    VideoCodec --> Decoder
    VideoCodec --> HyperpriorEntropyModel
    VideoCodec --> GOPManager
    VideoCodec --> MCTSRateController
```

---

## Data Flow Diagrams

### Encoding Pipeline

```mermaid
flowchart TB
    subgraph Input
        A[RGB Frame<br/>B×3×H×W]
    end

    subgraph Encoder["Encoder (Analysis Transform)"]
        B[Initial Conv<br/>3→64 channels]
        C[DownsampleBlock ×4<br/>GDN + FNet-Galerkin]
        D[Output Conv<br/>→192 channels]
    end

    subgraph Quantization
        E[QP Scaling<br/>y / 2^((QP-23)/6)]
        F[Quantizer<br/>Noise/STE/Soft]
    end

    subgraph Entropy["Entropy Model"]
        G[HyperAnalysis<br/>y → z]
        H[HyperSynthesis<br/>z → σ]
        I[GaussianConditional<br/>p(y|σ)]
    end

    subgraph Coding
        J[EntropyCoder<br/>Range coding]
        K[Bitstream]
    end

    A --> B --> C --> D
    D --> E --> F
    F --> G --> H
    H --> I --> J --> K

    style Input fill:#e1f5fe
    style Coding fill:#c8e6c9
```

### Decoding Pipeline

```mermaid
flowchart TB
    subgraph Input
        A[Bitstream]
        B[Scales σ]
    end

    subgraph Decoding
        C[EntropyCoder<br/>Range decode]
        D[Reshape to<br/>B×192×H/16×W/16]
    end

    subgraph Inverse
        E[Inverse QP Scale<br/>y × 2^((QP-23)/6)]
    end

    subgraph Decoder["Decoder (Synthesis Transform)"]
        F[Initial Conv]
        G[UpsampleBlock ×4<br/>IGDN + Attention]
        H[Output Conv<br/>→3 channels]
    end

    subgraph Output
        I[RGB Frame<br/>B×3×H×W]
    end

    A --> C
    B --> C
    C --> D --> E --> F --> G --> H --> I

    style Input fill:#fff3e0
    style Output fill:#c8e6c9
```

---

## Configuration Structure

```mermaid
classDiagram
    class CodecConfig {
        +name: str
        +seed: int
        +encoder: EncoderConfig
        +decoder: DecoderConfig
        +quantizer: QuantizerConfig
        +entropy: EntropyConfig
        +mcts: MCTSRateControlConfig
        +training: TrainingConfig
    }

    class EncoderConfig {
        +latent_channels: int = 192
        +n_layers: int = 4
        +d_model: int = 256
        +n_heads: int = 8
        +downsample_factor: int = 16
        +use_fnet_mixing: bool = True
        +fnet_ratio: float = 0.5
    }

    class DecoderConfig {
        +latent_channels: int = 192
        +n_layers: int = 4
        +upsample_factor: int = 16
    }

    class EntropyConfig {
        +model_type: EntropyModelType
        +hyper_channels: int = 64
        +num_filters: int = 192
    }

    class MCTSRateControlConfig {
        +rate_control_mode: RateControlMode
        +gop_size: int = 16
        +use_b_frames: bool = True
        +qp_min: int = 0
        +qp_max: int = 51
        +fps: float = 30.0
    }

    CodecConfig --> EncoderConfig
    CodecConfig --> DecoderConfig
    CodecConfig --> EntropyConfig
    CodecConfig --> MCTSRateControlConfig
```

---

## Key Algorithms

### Galerkin Attention (O(N) Complexity)

```
Input: Q, K, V ∈ R^(N×d)

# Monte Carlo integral approximation
Context = K^T @ V / N          # (d×d) matrix

# Project queries onto basis
Output = Q @ Context           # (N×d) result

# Key insight: O(N) instead of O(N²) softmax attention
```

### FNet Mixing (O(N log N) Complexity)

```
Input: x ∈ R^(B×N×d) reshaped to (B×H×W×d)

# 2D FFT for spatial mixing
x_freq = rfft2(x, dim=(1,2))   # Real FFT
x_mixed = irfft2(x_freq)       # Inverse FFT

# No learnable parameters - pure frequency mixing
```

### MCTS QP Selection

```
1. Encode frame latent → state via RepresentationNetwork
2. Initialize root node with PredictionNetwork(state)
3. For n_simulations:
   a. Select leaf via UCB: argmax(Q + c*prior*sqrt(N_parent)/(1+N))
   b. Expand: DynamicsNetwork(state, action) → next_state
   c. Evaluate: PredictionNetwork(next_state) → policy, value
   d. Backpropagate value up tree
4. Return action with highest visit count
```

---

## Module Dependencies

```mermaid
flowchart LR
    subgraph External
        PT[PyTorch]
        PD[Pydantic]
        NP[NumPy]
    end

    subgraph Core
        CFG[config.py]
        MOD[models/]
        COD[codec/]
        MTS[mcts/]
    end

    subgraph Support
        TRN[training/]
        MET[metrics/]
        DAT[data/]
        UTL[utils/]
    end

    PT --> MOD
    PT --> COD
    PT --> MTS
    PT --> TRN
    PD --> CFG
    NP --> MET

    CFG --> MOD
    CFG --> COD
    CFG --> MTS
    CFG --> TRN

    MOD --> COD
    MTS --> COD
    UTL --> COD

    COD --> TRN
    MET --> TRN
    DAT --> TRN
```

---

## Deployment View

```mermaid
flowchart TB
    subgraph Development
        DEV[Developer Machine<br/>Training & Testing]
    end

    subgraph Training Infrastructure
        GPU[GPU Cluster<br/>Multi-GPU Training]
        CKPT[(Checkpoints<br/>Model Weights)]
        LOGS[(Experiment Logs<br/>W&B / TensorBoard)]
    end

    subgraph Inference
        ONNX[ONNX Export<br/>Optimized Runtime]
        EDGE[Edge Device<br/>Real-time Encoding]
        CLOUD[Cloud Server<br/>Batch Processing]
    end

    DEV --> GPU
    GPU --> CKPT
    GPU --> LOGS
    CKPT --> ONNX
    ONNX --> EDGE
    ONNX --> CLOUD
```

---

## Summary

The AlphaGalerkin Video Compression system is a modular, resolution-independent neural codec featuring:

1. **O(N) Galerkin Attention**: Linear complexity attention for scalability
2. **FNet Mixing**: O(N log N) FFT-based spatial mixing
3. **Scale Hyperprior**: Learned entropy model for rate estimation
4. **MCTS Rate Control**: MuZero-style tree search for optimal QP selection
5. **GOP Management**: Efficient I/P/B frame scheduling with reference buffers

The architecture supports zero-shot resolution transfer and end-to-end differentiable training with rate-distortion optimization.
