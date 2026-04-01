#!/usr/bin/env python3
"""Diagnostic script for Vertex AI connectivity and credentials."""

import socket
import ssl
import sys
from datetime import datetime

import structlog

try:
    import google.auth

    HAS_SDK = True
except ImportError:
    HAS_SDK = False

logger = structlog.get_logger(__name__)


def check_dns(host: str) -> bool:
    """Check if host resolves in DNS."""
    try:
        ip = socket.gethostbyname(host)
        logger.info("dns_resolved", host=host, ip=ip)
        return True
    except socket.gaierror as e:
        logger.error("dns_resolution_failed", host=host, error=str(e))
        return False


def check_tcp(host: str, port: int = 443) -> bool:
    """Check TCP connectivity to host:port."""
    try:
        with socket.create_connection((host, port), timeout=5):
            logger.info("tcp_connection_ok", host=host, port=port)
            return True
    except Exception as e:
        logger.error("tcp_connection_failed", host=host, port=port, error=str(e))
        return False


def check_ssl(host: str, port: int = 443) -> bool:
    """Check SSL/TLS handshake."""
    try:
        context = ssl.create_default_context()
        with (
            socket.create_connection((host, port), timeout=5) as sock,
            context.wrap_socket(sock, server_hostname=host) as ssock,
        ):
            logger.info("ssl_handshake_ok", host=host, version=ssock.version())
            return True
    except Exception as e:
        logger.error("ssl_handshake_failed", host=host, error=str(e))
        return False


def check_auth() -> bool:
    """Check Google Cloud credentials."""
    if not HAS_SDK:
        logger.warning("sdk_not_installed", hint="pip install google-cloud-aiplatform")
        return False

    try:
        credentials, project = google.auth.default()
        logger.info("credentials_found", project=project, type=type(credentials).__name__)

        if not project:
            logger.warning(
                "no_default_project",
                hint="Run 'gcloud config set project [PROJECT_ID]'",
            )

        return True
    except Exception as e:
        logger.error(
            "auth_check_failed",
            error=str(e),
            hint="Run 'gcloud auth application-default login'",
        )
        return False


def main() -> int:
    """Run all diagnostics."""
    print(f"\nVertex AI Diagnostic Tool - {datetime.now().isoformat()}")
    print("=" * 60)

    hosts = [
        "us-central1-aiplatform.googleapis.com",
        "oauth2.googleapis.com",
        "storage.googleapis.com",
    ]

    results = {}

    for host in hosts:
        print(f"\nChecking: {host}")
        dns = check_dns(host)
        tcp = check_tcp(host) if dns else False
        ssl_ok = check_ssl(host) if tcp else False
        results[host] = all([dns, tcp, ssl_ok])

    print("\nAuthentication:")
    results["auth"] = check_auth()

    print("\n" + "=" * 60)
    success = all(results.values())

    if success:
        print("DIAGNOSTIC PASSED: Connectivity and credentials look good.")
        return 0
    else:
        print("DIAGNOSTIC FAILED: See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
