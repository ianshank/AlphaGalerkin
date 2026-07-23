#!/usr/bin/env python
"""CLI script for training AlphaGalerkin video compression model.

Usage:
    python scripts/train_compression.py --data-dir data/images --epochs 100
    python scripts/train_compression.py --config configs/compression.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Generator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.video_compression.codec.codec import create_codec
from src.video_compression.config import CodecConfig, TrainingConfig
from src.video_compression.training.trainer import VideoCompressionTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset[torch.Tensor]):
    """Simple image dataset for training."""

    def __init__(
        self,
        root: Path,
        patch_size: int = 256,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        """Initialize dataset.

        Args:
            root: Root directory containing images.
            patch_size: Size of random crops for training.
            extensions: Valid image extensions.

        """
        self.root = Path(root)
        self.patch_size = patch_size

        self.files: list[Path] = []
        for ext in extensions:
            self.files.extend(self.root.glob(f"**/*{ext}"))

        if not self.files:
            raise ValueError(f"No images found in {root}")

        logger.info(f"Found {len(self.files)} images")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and preprocess image.

        Args:
            idx: Image index.

        Returns:
            Image tensor (3, H, W) in [0, 1].

        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL not installed. Install with: pip install pillow")

        pil_img = Image.open(self.files[idx]).convert("RGB")
        img: torch.Tensor = torch.from_numpy(__import__("numpy").array(pil_img)).float() / 255.0
        img = img.permute(2, 0, 1)  # HWC to CHW

        # Random crop
        _, h, w = img.shape
        if h >= self.patch_size and w >= self.patch_size:
            top = int(torch.randint(0, h - self.patch_size + 1, (1,)).item())
            left = int(torch.randint(0, w - self.patch_size + 1, (1,)).item())
            img = img[:, top : top + self.patch_size, left : left + self.patch_size]
        else:
            # Resize if too small
            import torch.nn.functional as F

            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.patch_size, self.patch_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return img


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AlphaGalerkin video compression model",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing training images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/compression"),
        help="Output directory for checkpoints (default: outputs/compression)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--lambda-rd",
        type=float,
        default=0.01,
        help="Rate-distortion tradeoff (default: 0.01)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Training patch size (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training (default: auto)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="Enable Langfuse experiment tracking",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Check data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    torch.manual_seed(42)

    # Create dataset
    dataset = ImageDataset(args.data_dir, patch_size=args.patch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Create config
    training_config = TrainingConfig(
        name="training",
        learning_rate=args.lr,
        batch_size=args.batch_size,
        lambda_rd=args.lambda_rd,
        patch_size=args.patch_size,
        total_steps=args.epochs * len(train_loader),
    )

    codec_config = CodecConfig(
        name="codec",
        training=training_config,
    )

    # Create codec and trainer
    codec = create_codec(codec_config)
    trainer = VideoCompressionTrainer(
        codec=codec,
        config=training_config,
        output_dir=args.output_dir,
    )

    # Resume if specified
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    logger.info(f"Lambda R-D: {args.lambda_rd}")

    def infinite_loader() -> Generator[torch.Tensor, None, None]:
        while True:
            yield from train_loader

    trainer.train(infinite_loader())

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
