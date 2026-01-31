#!/usr/bin/env python3
"""Manage Vertex AI training jobs.

This script provides a CLI for managing Vertex AI training jobs,
including listing, monitoring, and cancelling jobs.

Usage:
    python -m scripts.vertex_jobs --help

    # List recent jobs
    python -m scripts.vertex_jobs --project my-project list

    # Show job details
    python -m scripts.vertex_jobs --project my-project show JOB_ID

    # Cancel a running job
    python -m scripts.vertex_jobs --project my-project cancel JOB_ID

    # Stream job logs
    python -m scripts.vertex_jobs --project my-project logs JOB_ID
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage Vertex AI training jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global arguments
    parser.add_argument(
        "--project",
        type=str,
        help="GCP project ID (uses gcloud config if not set)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-central1",
        help="GCP region (default: us-central1)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List training jobs")
    list_parser.add_argument(
        "--filter",
        type=str,
        help="Filter expression (e.g., 'display_name=\"my-job\"')",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of jobs to show (default: 20)",
    )
    list_parser.add_argument(
        "--state",
        type=str,
        choices=["running", "pending", "succeeded", "failed", "all"],
        default="all",
        help="Filter by job state",
    )

    # Show command
    show_parser = subparsers.add_parser("show", help="Show job details")
    show_parser.add_argument(
        "job_id",
        type=str,
        help="Job resource name or ID",
    )

    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a running job")
    cancel_parser.add_argument(
        "job_id",
        type=str,
        help="Job resource name or ID",
    )
    cancel_parser.add_argument(
        "--force",
        action="store_true",
        help="Cancel without confirmation",
    )

    # Wait command
    wait_parser = subparsers.add_parser("wait", help="Wait for job to complete")
    wait_parser.add_argument(
        "job_id",
        type=str,
        help="Job resource name or ID",
    )
    wait_parser.add_argument(
        "--timeout",
        type=int,
        default=86400,  # 24 hours
        help="Maximum wait time in seconds",
    )
    wait_parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between status checks",
    )

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show job logs")
    logs_parser.add_argument(
        "job_id",
        type=str,
        help="Job resource name or ID",
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
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def get_project(args: argparse.Namespace) -> str:
    """Get project ID from args or gcloud config.

    Args:
        args: Parsed arguments.

    Returns:
        Project ID.

    Raises:
        ValueError: If project cannot be determined.
    """
    if args.project:
        return args.project

    # Try to get from gcloud config
    import subprocess
    try:
        result = subprocess.run(
            ["gcloud", "config", "get", "project"],
            capture_output=True,
            text=True,
            check=True,
        )
        project = result.stdout.strip()
        if project:
            return project
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise ValueError(
        "Project ID not specified. Use --project or set via 'gcloud config set project'"
    )


def cmd_list(args: argparse.Namespace) -> int:
    """List training jobs.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from src.vertex.config import VertexRegion, VertexStorageConfig, VertexTrainingConfig
    from src.vertex.launcher import JobState, VertexLauncher

    try:
        project = get_project(args)
    except ValueError as e:
        logger.error("project_required", error=str(e))
        return 1

    # Build minimal config for launcher
    config = VertexTrainingConfig(
        project_id=project,
        region=VertexRegion(args.region),
        staging_bucket=f"gs://{project}-staging",
        storage=VertexStorageConfig(bucket_name=f"{project}-staging"),
    )

    launcher = VertexLauncher(config)

    # Build filter
    filter_str = args.filter
    if args.state != "all":
        state_filter = {
            "running": "state=JOB_STATE_RUNNING",
            "pending": "state=JOB_STATE_PENDING OR state=JOB_STATE_QUEUED",
            "succeeded": "state=JOB_STATE_SUCCEEDED",
            "failed": "state=JOB_STATE_FAILED",
        }.get(args.state)
        if state_filter:
            if filter_str:
                filter_str = f"({filter_str}) AND ({state_filter})"
            else:
                filter_str = state_filter

    try:
        jobs = launcher.list_jobs(filter_str=filter_str, limit=args.limit)
    except Exception as e:
        logger.error("list_failed", error=str(e))
        return 1

    if not jobs:
        print("No jobs found.")
        return 0

    # Print table header
    print(f"\n{'Display Name':<35} {'State':<20} {'Created':<20}")
    print("-" * 80)

    for job in jobs:
        # Truncate display name if too long
        display_name = job.job_name[:32] + "..." if len(job.job_name) > 35 else job.job_name

        # Format state
        state = job.state.name.replace("JOB_STATE_", "")

        # Format created time
        created = job.create_time[:19] if job.create_time else "N/A"

        print(f"{display_name:<35} {state:<20} {created:<20}")

    print(f"\nTotal: {len(jobs)} job(s)\n")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show job details.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from src.vertex.config import VertexRegion, VertexStorageConfig, VertexTrainingConfig
    from src.vertex.launcher import VertexLauncher

    try:
        project = get_project(args)
    except ValueError as e:
        logger.error("project_required", error=str(e))
        return 1

    config = VertexTrainingConfig(
        project_id=project,
        region=VertexRegion(args.region),
        staging_bucket=f"gs://{project}-staging",
        storage=VertexStorageConfig(bucket_name=f"{project}-staging"),
    )

    launcher = VertexLauncher(config)

    try:
        status = launcher.get_job_status(args.job_id)
    except Exception as e:
        logger.error("show_failed", error=str(e))
        return 1

    print(f"\nJob Details")
    print("=" * 60)
    print(f"Job ID:       {status.job_id}")
    print(f"State:        {status.state.name}")
    if status.state_message:
        print(f"Message:      {status.state_message}")
    if status.start_time:
        print(f"Started:      {status.start_time}")
    if status.end_time:
        print(f"Ended:        {status.end_time}")
    if status.error_message:
        print(f"Error:        {status.error_message}")
    print("=" * 60 + "\n")

    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    """Cancel a running job.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from src.vertex.config import VertexRegion, VertexStorageConfig, VertexTrainingConfig
    from src.vertex.launcher import VertexLauncher

    try:
        project = get_project(args)
    except ValueError as e:
        logger.error("project_required", error=str(e))
        return 1

    if not args.force:
        confirm = input(f"Cancel job {args.job_id}? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return 0

    config = VertexTrainingConfig(
        project_id=project,
        region=VertexRegion(args.region),
        staging_bucket=f"gs://{project}-staging",
        storage=VertexStorageConfig(bucket_name=f"{project}-staging"),
    )

    launcher = VertexLauncher(config)

    if launcher.cancel_job(args.job_id):
        print(f"Cancellation requested for job {args.job_id}")
        return 0
    else:
        logger.error("cancel_failed")
        return 1


def cmd_wait(args: argparse.Namespace) -> int:
    """Wait for job to complete.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from src.vertex.config import VertexRegion, VertexStorageConfig, VertexTrainingConfig
    from src.vertex.launcher import JobState, VertexLauncher

    try:
        project = get_project(args)
    except ValueError as e:
        logger.error("project_required", error=str(e))
        return 1

    config = VertexTrainingConfig(
        project_id=project,
        region=VertexRegion(args.region),
        staging_bucket=f"gs://{project}-staging",
        storage=VertexStorageConfig(bucket_name=f"{project}-staging"),
    )

    launcher = VertexLauncher(config)

    print(f"Waiting for job {args.job_id} to complete...")

    try:
        status = launcher.wait_for_completion(
            args.job_id,
            poll_interval=float(args.poll_interval),
            timeout=float(args.timeout),
        )
    except TimeoutError:
        logger.error("wait_timeout", timeout=args.timeout)
        return 1
    except Exception as e:
        logger.error("wait_failed", error=str(e))
        return 1

    print(f"\nJob completed with state: {status.state.name}")

    if status.state == JobState.SUCCEEDED:
        return 0
    else:
        if status.error_message:
            print(f"Error: {status.error_message}")
        return 1


def cmd_logs(args: argparse.Namespace) -> int:
    """Show job logs.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    try:
        project = get_project(args)
    except ValueError as e:
        logger.error("project_required", error=str(e))
        return 1

    # For logs, we use gcloud CLI as it provides better streaming
    print(f"To stream logs, run:")
    print(f"  gcloud ai custom-jobs stream-logs {args.job_id} \\")
    print(f"    --region={args.region} \\")
    print(f"    --project={project}")
    print()
    print("Or view in Cloud Console:")
    print(f"  https://console.cloud.google.com/logs/query;query=resource.labels.job_id%3D%22{args.job_id}%22?project={project}")

    return 0


def main() -> int:
    """Main entry point.

    Returns:
        Exit code.
    """
    args = parse_args()
    setup_logging(debug=args.debug)

    if not args.command:
        print("No command specified. Use --help for usage.")
        return 1

    commands = {
        "list": cmd_list,
        "show": cmd_show,
        "cancel": cmd_cancel,
        "wait": cmd_wait,
        "logs": cmd_logs,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        return cmd_func(args)
    else:
        logger.error("unknown_command", command=args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
