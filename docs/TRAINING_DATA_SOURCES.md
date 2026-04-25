# Training Data Sources for AlphaGalerkin

## Context

AlphaGalerkin currently generates **all training data synthetically** (Poisson, Darcy, Heat solvers) with no external dataset downloads. The project has three major domains that could benefit from real-world training data:

1. **Game AI** (Go, Chess) — self-play + supervised pre-training
2. **PDE/Scientific Computing** — neural operator benchmarks
3. **Neural Video Compression** — image/video codec training

Adding external datasets would enable supervised pre-training, benchmarking against published baselines, and training on real-world distributions rather than synthetic-only data.

---

## 1. Go Game Datasets

| Dataset | Size | Format | License | URL |
|---------|------|--------|---------|-----|
| **KataGo Distributed Training** | 92M+ games, 4.5B+ training rows | KataGo native | Open/community | https://katagotraining.org/ |
| **ELF OpenGo** | 20M self-play games + 87K pro games | ELF format | Open source | https://github.com/pytorch/ELF/releases |
| **PAGE (Pro Annotation)** | 98,525 games, 2007 players | SGF | CC BY-NC-SA 4.0 | https://github.com/yifangao112/PAGE |
| **CWI.nl Pro Archive** | 88,888+ games | SGF | Free | https://homepages.cwi.nl/~aeb/go/games/games/ |
| **KGS Server Archives** | All KGS games ever played | SGF (monthly) | Free (rate-limited) | https://www.gokgs.com/archives.jsp |
| **Computer Go Dataset Repo** | Curated collection | Multiple | Open source | https://github.com/yenw/computer-go-dataset |

**Integration**: The codebase already has an SGF parser (`src/games/sgf/parser.py`) supporting FF[4]. SGF datasets can be loaded directly into Go `GameState` objects for supervised pre-training of the policy head before self-play begins.

**Recommendation**: Start with **PAGE** (98K annotated pro games, SGF) for supervised pre-training, then use **KataGo** data for large-scale training if needed.

---

## 2. Chess Game Datasets

| Dataset | Size | Format | License | URL |
|---------|------|--------|---------|-----|
| **Lichess Open Database** | 7.5B+ games | PGN (.zst compressed) | Free | https://database.lichess.org/ |
| **Lichess Elite Database** | Filtered high-rated subset | PGN | Free | https://database.nikonoel.fr/ |
| **Leela Chess Zero Self-Play** | 475K+ games (multiple sets) | PGN | GPLv3 | Kaggle (anthonytherrien) |
| **FICS Games Database** | All FICS games since 1999 | PGN | Free | https://www.ficsgames.org/download.html |

**Integration**: Chess implementation (`src/games/chess.py`) uses FEN notation and 119-plane AlphaZero encoding. A PGN parser would need to be added to convert game records into training experiences. The action space is 4672 (8x8x73 move types).

**Recommendation**: **Lichess Elite** for manageable size of high-quality games. **Leela Chess Zero** self-play data is closest to AlphaZero-style training.

---

## 3. PDE / Neural Operator Datasets

| Dataset | PDEs Covered | Format | License | URL |
|---------|-------------|--------|---------|-----|
| **PDEBench** (NeurIPS 2022) | Advection, Burgers, Diffusion-Reaction, Navier-Stokes (1D/2D/3D), Darcy, Shallow Water | HDF5 | Open | https://github.com/pdebench/PDEBench |
| **DeepONet/FNO Datasets** | Darcy Flow, various PDEs | HDF5/MATLAB | Open source | https://github.com/lu-group/deeponet-fno |
| **Caltech Operator Learning** | Burgers, Darcy | NumPy/MATLAB | Open | https://data.caltech.edu/records/55tdh-hda68 |
| **Burgers FSU** | Burgers (40 solutions) | ASCII | GNU LGPL | https://people.sc.fsu.edu/~jburkardt/datasets/burgers/burgers.html |

**Integration**: The physics data pipeline (`src/data/physics_dataset.py`) expects `PhysicsSample` objects with `input_field`, `output_field`, `coords`, `grid_size`. External HDF5 datasets would need a loader to convert into this format.

**Recommendation**: **PDEBench** is the gold standard — covers Burgers, Navier-Stokes, Darcy (all of which AlphaGalerkin already implements). Enables direct comparison with published FNO/U-Net/PINN baselines. The **DeepONet/FNO** datasets are also directly relevant for Darcy flow benchmarking.

---

## 4. Video/Image Compression Datasets

| Dataset | Type | Size | License | URL |
|---------|------|------|---------|-----|
| **Vimeo-90K** | Video (7-frame clips) | 82 GB, 91K sequences | Academic | https://github.com/anchen1011/toflow |
| **Kodak PhotoCD** | Images (24) | Small (768x512) | Unrestricted | Public mirrors |
| **CLIC 2021** | Images (2K res) | ~41 images | Research | https://www.clic2021.org/ |
| **PE Video Dataset** (Meta) | Video | 1M videos | Open | https://ai.meta.com/datasets/pe-video/ |
| **CompressAI** | Framework + datasets | Integrated | Apache 2.0 | https://interdigitalinc.github.io/CompressAI/ |

**Integration**: Video compression data loader (`src/video_compression/data/dataset.py`) expects images (.jpg/.png/.bmp/.webp) as `(C, H, W)` tensors in [0,1], or `VideoClip` objects with `(T, C, H, W)` frames. Datasets like Vimeo-90K and Kodak can be used directly.

**Recommendation**: **Kodak** (standard eval benchmark, tiny), **CLIC** (modern compression benchmark), and **Vimeo-90K** (large-scale training). These are the standard datasets used in all learned compression papers.

---

## Implementation Plan

### What to build: `src/data/external_datasets.py`

A dataset download and loading module with:

1. **Download manager** — fetch, cache, and verify datasets with checksums
2. **Format converters** per domain:
   - SGF/PGN → `Experience` objects (game data)
   - HDF5 → `PhysicsSample` objects (PDE data)
   - Image/video folders → existing `ImageDataset`/`VideoClip` format
3. **Registry** — declarative dataset catalog with metadata (URL, size, format, license)
4. **CLI command** — `python -m src.data.download --dataset pdebench-burgers`

### Priority Order

1. **PDEBench** — immediate benchmarking value, enables published baseline comparison
2. **SGF Go datasets** (PAGE/CWI) — parser already exists, enables supervised pre-training
3. **Kodak + CLIC** — standard compression eval, tiny download
4. **Vimeo-90K** — large-scale video compression training
5. **Chess PGN** — requires new PGN parser, lower priority
6. **Lichess/KataGo** — massive scale, only needed later

### Verification

- Download smallest dataset (Kodak, 24 images) and verify loading through existing `ImageDataset`
- Download PDEBench Burgers subset and verify loading through `PhysicsDataset`
- Load SGF files through existing `src/games/sgf/parser.py` and verify `GameState` conversion
- Run existing tests to ensure no regressions: `pytest tests/ -v`
