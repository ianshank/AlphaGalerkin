#!/usr/bin/env python3
r"""Launch AlphaGalerkin training on Vertex AI.

This script provides a CLI for launching training jobs on Google Cloud
Vertex AI, including job configuration, container specification, and
optional checkpoint resumption.

Usage:
    python -m scripts.train_vertex --help

    # Launch new training job
    python -m scripts.train_vertex \\
        --project my-gcp-project \\
        --bucket my-training-bucket \\
        --container-uri us-docker.pkg.dev/my-project/alphagalerkin/trainer:latest

    # Launch with custom configuration
    python -m scripts.train_vertex \\
        --project my-gcp-project \\
        --bucket my-training-bucket \\
        --container-uri my-container:latest \\
        --machine-type a2-highgpu-4g \\
        --region us-west1 \\
        --spot

    # Resume from checkpoint
    python -m scripts.train_vertex \\
        --project my-gcp-project \\
        --bucket my-training-bucket \\
        --container-uri my-container:latest \\
        --resume gs://my-bucket/checkpoints/checkpoint_00010000.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch AlphaGalerkin training on Vertex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "--project",
        type=str,
        required=True,
        help="GCP project ID",
    )
    required.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="GCS bucket for checkpoints and data (without gs://)",
    )
    required.add_argument(
        "--container-uri",
        type=str,
        required=True,
        help="Container image URI (gcr.io or Artifact Registry)",
    )

    # Resource configuration
    resources = parser.add_argument_group("Resource configuration")
    resources.add_argument(
        "--machine-type",
        type=str,
        default="a2-highgpu-1g",
        help="VM machine type (default: a2-highgpu-1g)",
    )
    resources.add_argument(
        "--accelerator-type",
        type=str,
        default="NVIDIA_TESLA_A100",
        help="GPU accelerator type (default: NVIDIA_TESLA_A100)",
    )
    resources.add_argument(
        "--accelerator-count",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)",
    )
    resources.add_argument(
        "--replica-count",
        type=int,
        default=1,
        help="Number of training replicas (default: 1)",
    )

    # Location and naming
    location = parser.add_argument_group("Location and naming")
    location.add_argument(
        "--region",
        type=str,
        default="us-central1",
        help="GCP region (default: us-central1)",
    )
    location.add_argument(
        "--display-name",
        type=str,
        default="alphagalerkin-training",
        help="Display name for the job",
    )

    # Cost optimization
    cost = parser.add_argument_group("Cost optimization")
    cost.add_argument(
        "--spot",
        action="store_true",
        help="Use spot/preemptible instances for cost savings",
    )
    cost.add_argument(
        "--timeout-hours",
        type=int,
        default=24,
        help="Maximum training duration in hours (default: 24)",
    )

    # Training configuration
    training = parser.add_argument_group("Training configuration")
    training.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
        help="Training configuration file (default: config/train.yaml)",
    )
    training.add_argument(
        "--resume",
        type=str,
        help="GCS path to checkpoint for resuming training",
    )

    # Additional options
    options = parser.add_argument_group("Additional options")
    options.add_argument(
        "--service-account",
        type=str,
        help="Service account email for the training job",
    )
    options.add_argument(
        "--tensorboard",
        type=str,
        help="Vertex AI TensorBoard instance name",
    )
    options.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Labels in key=value format",
    )
    options.add_argument(
        "--sync",
        action="store_true",
        help="Wait for job to complete before returning",
    )
    options.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without launching",
    )
    options.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # W&B configuration
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument(
        "--wandb-api-key",
        type=str,
        help="W&B API key (or set WANDB_API_KEY env var)",
    )
    wandb_group.add_argument(
        "--wandb-project",
        type=str,
        default="alphagalerkin",
        help="W&B project name (default: alphagalerkin)",
    )
    wandb_group.add_argument(
        "--wandb-entity",
        type=str,
        help="W&B entity (team or username)",
    )
    wandb_group.add_argument(
        "--wandb-run-name",
        type=str,
        help="W&B run name (auto-generated if not set)",
    )
    wandb_group.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (default: online)",
    )

    # Authentication configuration
    auth_group = parser.add_argument_group("Authentication")
    auth_group.add_argument(
        "--auth-method",
        type=str,
        default="adc",
        choices=["adc", "service_account", "gcloud"],
        help="GCP auth method: adc (default), service_account, or gcloud",
    )
    auth_group.add_argument(
        "--service-account-key",
        type=str,
        help="Path to service account JSON key file (for --auth-method service_account)",
    )
    auth_group.add_argument(
        "--validate-auth",
        action="store_true",
        default=True,
        help="Validate credentials before job submission (default: enabled)",
    )
    auth_group.add_argument(
        "--no-validate-auth",
        action="store_true",
        help="Skip credential validation before job submission",
    )

    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """Configure structured logging."""
    import logging

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="%(message)s",
        level=level,
        stream=sys.stderr,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def parse_labels(label_strs: list[str] | None) -> dict[str, str]:
    """Parse label strings into dictionary.

    Args:
        label_strs: List of "key=value" strings.

    Returns:
        Dictionary of labels.

    """
    if not label_strs:
        return {}

    labels = {}
    for label in label_strs:
        if "=" in label:
            key, value = label.split("=", 1)
            labels[key.strip()] = value.strip()
    return labels


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success).

    """
    args = parse_args()
    setup_logging(debug=args.debug)

    try:
        from src.vertex.config import (
            AcceleratorType,
            VertexMachineType,
            VertexRegion,
            VertexResourceConfig,
            VertexStorageConfig,
            VertexTrainingConfig,
        )
        from src.vertex.cost import estimate_job_cost
        from src.vertex.launcher import VertexLauncher
    except ImportError as e:
        logger.error(
            "import_error",
            error=str(e),
            hint="Ensure google-cloud-aiplatform is installed",
        )
        return 1

    # Parse enums
    try:
        machine_type = VertexMachineType(args.machine_type)
    except ValueError:
        logger.error("invalid_machine_type", value=args.machine_type)
        return 1

    try:
        region = VertexRegion(args.region)
    except ValueError:
        logger.error("invalid_region", value=args.region)
        return 1

    accelerator_type = None
    if args.accelerator_type and args.accelerator_count > 0:
        try:
            accelerator_type = AcceleratorType(args.accelerator_type)
        except ValueError:
            logger.error("invalid_accelerator_type", value=args.accelerator_type)
            return 1

    # Parse labels
    labels = parse_labels(args.labels)

    # Build configuration
    resources = VertexResourceConfig(
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=args.accelerator_count,
        replica_count=args.replica_count,
    )

    storage = VertexStorageConfig(
        bucket_name=args.bucket,
    )

    # Determine auth validation setting
    validate_auth = not args.no_validate_auth

    config = VertexTrainingConfig(
        project_id=args.project,
        region=region,
        staging_bucket=f"gs://{args.bucket}",
        resources=resources,
        storage=storage,
        service_account=args.service_account,
        tensorboard_name=args.tensorboard,
        timeout_hours=args.timeout_hours,
        enable_spot=args.spot,
        labels=labels,
        # Auth settings
        auth_method=args.auth_method,
        service_account_key_path=args.service_account_key,
        validate_auth_before_launch=validate_auth,
    )

    # Estimate cost
    estimate = estimate_job_cost(
        machine_type=machine_type,
        duration_hours=float(args.timeout_hours),
        accelerator_type=accelerator_type,
        accelerator_count=args.accelerator_count,
        replica_count=args.replica_count,
        is_spot=args.spot,
    )

    # Build W&B environment variables
    wandb_env: dict[str, str] = {}
    api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb_env["WANDB_API_KEY"] = api_key
        wandb_env["WANDB_PROJECT"] = args.wandb_project
        wandb_env["WANDB_MODE"] = args.wandb_mode
        if args.wandb_entity:
            wandb_env["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_run_name:
            wandb_env["WANDB_RUN_NAME"] = args.wandb_run_name

    # Print configuration summary
    print("\n" + "=" * 60)
    print("Vertex AI Training Job Configuration")
    print("=" * 60)
    print(f"Project:         {args.project}")
    print(f"Region:          {args.region}")
    print(f"Display Name:    {args.display_name}")
    print(f"Container:       {args.container_uri}")
    print()
    print("Resources:")
    print(f"  Machine Type:  {args.machine_type}")
    if accelerator_type:
        print(f"  Accelerator:   {args.accelerator_count}x {args.accelerator_type}")
    print(f"  Replicas:      {args.replica_count}")
    print(f"  Spot:          {args.spot}")
    print()
    if wandb_env.get("WANDB_API_KEY"):
        print(f"W&B:             Enabled (project: {args.wandb_project})")
    else:
        print("W&B:             Disabled (no API key)")
    print()
    print("Authentication:")
    print(f"  Method:        {args.auth_method}")
    print(f"  Validate:      {validate_auth}")
    if args.service_account_key:
        print(f"  Key File:      {args.service_account_key}")
    print()
    print("Cost Estimate:")
    print(f"  Hourly Rate:   ${estimate.total_cost_per_hour:.2f}/hr")
    print(f"  Max Duration:  {args.timeout_hours} hours")
    print(f"  Max Cost:      ${estimate.estimated_total_cost:.2f}")
    print("=" * 60 + "\n")

    if args.dry_run:
        logger.info("dry_run_complete")
        return 0

    # Build training arguments
    training_args = ["--config", f"/app/{args.config}"]
    if args.resume:
        training_args.extend(["--resume", args.resume])

    # Launch job
    try:
        from src.vertex.auth import AuthenticationError

        launcher = VertexLauncher(config)
        result = launcher.launch(
            display_name=args.display_name,
            container_uri=args.container_uri,
            args=training_args,
            environment_variables=wandb_env if wandb_env else None,
            sync=args.sync,
        )

        print("\nJob launched successfully!")
        print(f"  Job Name:   {result.job_name}")
        print(f"  Job ID:     {result.job_id}")
        print(f"  Console:    {result.console_url}")
        print(f"  State:      {result.state.value}")

        if args.sync:
            print(f"\nJob completed with state: {result.state.value}")

        return 0

    except AuthenticationError as e:
        print("\nAuthentication failed!")
        print(f"  Error: {e}")
        if e.validation_result:
            if e.validation_result.suggestions:
                print("\n  Suggestions:")
                for suggestion in e.validation_result.suggestions:
                    print(f"    - {suggestion}")
        print("\n  Use --no-validate-auth to skip this check (not recommended).")
        return 1

    except Exception as e:
        logger.exception("launch_failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
