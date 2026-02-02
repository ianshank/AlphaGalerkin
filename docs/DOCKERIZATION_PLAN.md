# AlphaGalerkin Dockerization Plan: Safety-First Approach

## Executive Summary

This plan outlines a comprehensive strategy for dockerizing AlphaGalerkin with a focus on **security around trained neural networks**. The primary safety concerns include:

1. **Checkpoint deserialization attacks** - PyTorch's `pickle`-based checkpoints can execute arbitrary code
2. **Model inference isolation** - Preventing compromised models from attacking the host
3. **Supply chain security** - Verifying base images and pinning dependencies
4. **Runtime containment** - Resource limits, network isolation, read-only filesystems

---

## Table of Contents

1. [Threat Model](#1-threat-model)
2. [Container Architecture](#2-container-architecture)
3. [Dockerfile Specifications](#3-dockerfile-specifications)
4. [Safety Mechanisms](#4-safety-mechanisms)
5. [Checkpoint Validation System](#5-checkpoint-validation-system)
6. [Docker Compose Configuration](#6-docker-compose-configuration)
7. [Secrets Management](#7-secrets-management)
8. [CI/CD Integration](#8-cicd-integration)
9. [Implementation Phases](#9-implementation-phases)
10. [Verification Checklist](#10-verification-checklist)

---

## 1. Threat Model

### 1.1 Adversary Capabilities

| Threat Actor | Capability | Motivation |
|-------------|-----------|------------|
| Malicious Checkpoint | Arbitrary code execution via pickle | Data exfiltration, cryptomining |
| Compromised Dependency | Supply chain attack | Backdoor installation |
| External Attacker | Network-based attacks | Service disruption, data theft |
| Insider Threat | Access to training infrastructure | Model poisoning, IP theft |

### 1.2 Attack Vectors

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ATTACK SURFACE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Checkpoint  │    │ Dependencies │    │   Network    │          │
│  │    Files     │    │   (PyPI)     │    │   Egress     │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────────────────────────────────────────────┐          │
│  │              CONTAINER RUNTIME                        │          │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │          │
│  │  │   Model     │  │   MCTS      │  │   Codec     │  │          │
│  │  │  Loading    │  │  Inference  │  │  Pipeline   │  │          │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │          │
│  └──────────────────────────────────────────────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Host        │    │   GPU       │    │  External    │          │
│  │  Filesystem  │    │  Memory     │    │  Services    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Risk Matrix

| Risk | Likelihood | Impact | Mitigation Priority |
|------|-----------|--------|---------------------|
| Malicious checkpoint execution | High | Critical | P0 |
| Dependency vulnerability | Medium | High | P0 |
| Container escape | Low | Critical | P1 |
| Resource exhaustion DoS | Medium | Medium | P1 |
| Data exfiltration | Low | High | P2 |

---

## 2. Container Architecture

### 2.1 Multi-Container Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ALPHAGALERKIN CONTAINER STACK                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    TIER 1: BUILD-ONLY                        │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                   │   │
│  │  │  builder-base   │  │  test-runner    │                   │   │
│  │  │  (compile deps) │  │  (CI/CD only)   │                   │   │
│  │  └─────────────────┘  └─────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    TIER 2: TRAINING                          │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                   │   │
│  │  │   trainer       │  │  trainer-dist   │                   │   │
│  │  │   (single GPU)  │  │  (multi-node)   │                   │   │
│  │  └─────────────────┘  └─────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    TIER 3: INFERENCE (HARDENED)              │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │   │
│  │  │  inference      │  │  codec          │  │  api-server │  │   │
│  │  │  (Go AI eval)   │  │  (video comp)   │  │  (Gradio)   │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    TIER 4: VALIDATION                        │   │
│  │  ┌─────────────────────────────────────────────────────────┐│   │
│  │  │  checkpoint-validator (sandbox for checkpoint loading)  ││   │
│  │  └─────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Container Purposes

| Container | Purpose | Network | GPU | Filesystem |
|-----------|---------|---------|-----|------------|
| `builder-base` | Compile dependencies | Yes | No | RW |
| `test-runner` | CI/CD testing | Yes | Optional | RW |
| `trainer` | Single-node training | Egress only | Yes | RW volumes |
| `trainer-dist` | Multi-node training | Internal + Egress | Yes | RW volumes |
| `inference` | Go game evaluation | None | Yes | RO + tmpfs |
| `codec` | Video compression | None | Yes | RO + tmpfs |
| `api-server` | Web interface | Ingress only | Yes | RO + tmpfs |
| `checkpoint-validator` | Validate checkpoints | None | No | Isolated |

---

## 3. Dockerfile Specifications

### 3.1 Base Image Selection

**Recommendation**: Use NVIDIA's official PyTorch images with pinned digests.

```dockerfile
# DO NOT use floating tags
# BAD:  FROM nvcr.io/nvidia/pytorch:24.01-py3
# GOOD: FROM nvcr.io/nvidia/pytorch:24.01-py3@sha256:<digest>
```

**Base Image Matrix**:

| Use Case | Base Image | Size | Security |
|----------|-----------|------|----------|
| Training | `nvcr.io/nvidia/pytorch:24.01-py3` | ~15GB | Medium |
| Inference | `python:3.11-slim` + ONNX Runtime | ~1GB | High |
| Validation | `python:3.11-alpine` | ~150MB | Highest |

### 3.2 Dockerfile.base (Builder Stage)

```dockerfile
# docker/Dockerfile.base
# syntax=docker/dockerfile:1.6
ARG PYTORCH_VERSION=24.01
ARG BASE_DIGEST=sha256:abc123...

FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}-py3@${BASE_DIGEST} AS builder

# Security: Run as non-root
RUN useradd -m -u 1000 -s /bin/bash alphagalerkin

# Install build dependencies with pinned versions
COPY requirements-lock.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-lock.txt

# Copy source code (after deps to leverage caching)
WORKDIR /app
COPY --chown=alphagalerkin:alphagalerkin . .

# Install package in editable mode
RUN pip install --no-cache-dir -e ".[vertex]"

# Security: Remove unnecessary tools
RUN apt-get purge -y wget curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
```

### 3.3 Dockerfile.inference (Hardened Inference)

```dockerfile
# docker/Dockerfile.inference
# syntax=docker/dockerfile:1.6
ARG PYTHON_VERSION=3.11
ARG BASE_DIGEST=sha256:def456...

# Stage 1: Build dependencies
FROM python:${PYTHON_VERSION}-slim@${BASE_DIGEST} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt /tmp/
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r /tmp/requirements-inference.txt

# Stage 2: Runtime (minimal)
FROM python:${PYTHON_VERSION}-slim@${BASE_DIGEST} AS runtime

# Security hardening
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 -s /bin/false alphagalerkin

# Install pre-built wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index /wheels/*.whl && rm -rf /wheels

# Copy only inference code (not training)
WORKDIR /app
COPY --chown=alphagalerkin:alphagalerkin src/modeling/ src/modeling/
COPY --chown=alphagalerkin:alphagalerkin src/mcts/ src/mcts/
COPY --chown=alphagalerkin:alphagalerkin src/games/ src/games/
COPY --chown=alphagalerkin:alphagalerkin src/deployment/ src/deployment/

# Security: Read-only filesystem marker
LABEL org.opencontainers.image.readonly="true"

# Drop all capabilities, run as non-root
USER alphagalerkin
ENTRYPOINT ["python", "-m", "src.deployment.serve"]
```

### 3.4 Dockerfile.validator (Checkpoint Sandbox)

```dockerfile
# docker/Dockerfile.validator
# syntax=docker/dockerfile:1.6
# CRITICAL: This container validates untrusted checkpoints

FROM python:3.11-alpine AS validator

# Absolute minimal attack surface
RUN apk add --no-cache libstdc++ && \
    adduser -D -u 1000 validator

# Only install torch (no network libraries)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    safetensors==0.4.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy validation script only
WORKDIR /app
COPY src/safety/checkpoint_validator.py .

# Security: Heavily restricted
USER validator

# No network access is enforced at runtime via docker-compose
ENTRYPOINT ["python", "checkpoint_validator.py"]
```

---

## 4. Safety Mechanisms

### 4.1 Runtime Security Controls

```yaml
# Security controls applied to inference containers
security_controls:
  capabilities:
    drop: [ALL]
    add: []  # No capabilities needed for inference

  read_only_rootfs: true

  no_new_privileges: true

  seccomp_profile: "inference-seccomp.json"

  apparmor_profile: "alphagalerkin-inference"

  resource_limits:
    memory: "8g"
    cpu: "4"
    pids: 100
    nofile: 1024
```

### 4.2 Seccomp Profile for Inference

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": [
        "read", "write", "close", "fstat", "lseek",
        "mmap", "mprotect", "munmap", "brk",
        "rt_sigaction", "rt_sigprocmask",
        "ioctl", "access", "pipe", "select",
        "sched_yield", "mremap", "msync",
        "mincore", "madvise", "shmget", "shmat",
        "clone", "fork", "vfork", "execve",
        "exit", "wait4", "kill", "uname",
        "fcntl", "flock", "fsync", "fdatasync",
        "truncate", "ftruncate", "getdents",
        "getcwd", "chdir", "rename", "mkdir",
        "rmdir", "creat", "unlink", "readlink",
        "chmod", "fchmod", "chown", "fchown",
        "umask", "gettimeofday", "getrlimit",
        "getrusage", "sysinfo", "times",
        "getuid", "getgid", "geteuid", "getegid",
        "setuid", "setgid", "getgroups",
        "setgroups", "setresuid", "setresgid",
        "getpgid", "setpgid", "getpgrp",
        "setsid", "getpriority", "setpriority",
        "sched_setparam", "sched_getparam",
        "sched_setscheduler", "sched_getscheduler",
        "sched_get_priority_max", "sched_get_priority_min",
        "sched_rr_get_interval", "mlock", "munlock",
        "mlockall", "munlockall", "vhangup",
        "prctl", "arch_prctl", "adjtimex",
        "setrlimit", "sync", "acct",
        "mount", "umount2", "swapon", "swapoff",
        "reboot", "sethostname", "setdomainname",
        "ioperm", "iopl", "create_module",
        "init_module", "delete_module",
        "get_kernel_syms", "query_module",
        "quotactl", "nfsservctl", "getpmsg",
        "putpmsg", "afs_syscall", "tuxcall",
        "security", "gettid", "readahead",
        "setxattr", "lsetxattr", "fsetxattr",
        "getxattr", "lgetxattr", "fgetxattr",
        "listxattr", "llistxattr", "flistxattr",
        "removexattr", "lremovexattr", "fremovexattr",
        "tkill", "time", "futex", "sched_setaffinity",
        "sched_getaffinity", "set_thread_area",
        "io_setup", "io_destroy", "io_getevents",
        "io_submit", "io_cancel", "get_thread_area",
        "epoll_create", "epoll_ctl_old",
        "epoll_wait_old", "remap_file_pages",
        "getdents64", "set_tid_address",
        "restart_syscall", "semtimedop",
        "fadvise64", "timer_create",
        "timer_settime", "timer_gettime",
        "timer_getoverrun", "timer_delete",
        "clock_settime", "clock_gettime",
        "clock_getres", "clock_nanosleep",
        "exit_group", "epoll_wait", "epoll_ctl",
        "tgkill", "utimes", "mbind",
        "set_mempolicy", "get_mempolicy",
        "mq_open", "mq_unlink", "mq_timedsend",
        "mq_timedreceive", "mq_notify",
        "mq_getsetattr", "kexec_load", "waitid",
        "add_key", "request_key", "keyctl",
        "ioprio_set", "ioprio_get", "inotify_init",
        "inotify_add_watch", "inotify_rm_watch",
        "migrate_pages", "openat", "mkdirat",
        "mknodat", "fchownat", "futimesat",
        "newfstatat", "unlinkat", "renameat",
        "linkat", "symlinkat", "readlinkat",
        "fchmodat", "faccessat", "pselect6",
        "ppoll", "unshare", "set_robust_list",
        "get_robust_list", "splice", "tee",
        "sync_file_range", "vmsplice",
        "move_pages", "utimensat", "epoll_pwait",
        "signalfd", "timerfd_create", "eventfd",
        "fallocate", "timerfd_settime",
        "timerfd_gettime", "accept4", "signalfd4",
        "eventfd2", "epoll_create1", "dup3",
        "pipe2", "inotify_init1", "preadv",
        "pwritev", "rt_tgsigqueueinfo",
        "perf_event_open", "recvmmsg",
        "fanotify_init", "fanotify_mark",
        "prlimit64", "name_to_handle_at",
        "open_by_handle_at", "clock_adjtime",
        "syncfs", "sendmmsg", "setns",
        "getcpu", "process_vm_readv",
        "process_vm_writev", "kcmp",
        "finit_module", "sched_setattr",
        "sched_getattr", "renameat2",
        "seccomp", "getrandom", "memfd_create",
        "kexec_file_load", "bpf",
        "execveat", "userfaultfd",
        "membarrier", "mlock2", "copy_file_range",
        "preadv2", "pwritev2", "pkey_mprotect",
        "pkey_alloc", "pkey_free", "statx"
      ],
      "action": "SCMP_ACT_ALLOW"
    },
    {
      "names": ["socket", "connect", "bind", "listen", "accept"],
      "action": "SCMP_ACT_ERRNO",
      "comment": "Block all network syscalls for inference"
    }
  ]
}
```

### 4.3 Network Isolation Policies

```yaml
# docker/network-policies.yaml
networks:
  # Internal network for container communication
  alphagalerkin-internal:
    driver: bridge
    internal: true  # No external access
    ipam:
      config:
        - subnet: 172.28.0.0/16

  # Egress-only network for training (GCS, W&B)
  alphagalerkin-egress:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.enable_ip_masquerade: "true"
    # Use iptables rules to restrict to specific domains

  # Ingress network for API server
  alphagalerkin-ingress:
    driver: bridge
    # Exposed only to reverse proxy
```

### 4.4 GPU Isolation

```yaml
# docker-compose.gpu.yaml
services:
  inference:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          # Limit GPU memory to prevent DoS
          nvidia.com/gpu.memory: "4Gi"
    environment:
      # Restrict to specific GPU
      - CUDA_VISIBLE_DEVICES=0
      # Disable CUDA caching (security)
      - CUDA_CACHE_DISABLE=1
```

---

## 5. Checkpoint Validation System

### 5.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  CHECKPOINT VALIDATION PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────┐                                                  │
│  │   Untrusted   │                                                  │
│  │  Checkpoint   │                                                  │
│  └───────┬───────┘                                                  │
│          │                                                          │
│          ▼                                                          │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  STAGE 1: Static Analysis (No Execution)                      │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │ │
│  │  │ File Format │  │   Size      │  │  Pickle     │           │ │
│  │  │   Check     │  │  Limits     │  │  Opcodes    │           │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘           │ │
│  └───────────────────────────────────────────────────────────────┘ │
│          │                                                          │
│          ▼ (Pass)                                                   │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  STAGE 2: Sandboxed Deserialization                           │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │         checkpoint-validator container                   │ │ │
│  │  │  • No network access                                     │ │ │
│  │  │  • No filesystem (except input)                          │ │ │
│  │  │  • CPU-only (no GPU attack surface)                      │ │ │
│  │  │  • 30s timeout, 1GB memory limit                         │ │ │
│  │  │  • Custom RestrictedUnpickler                            │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│          │                                                          │
│          ▼ (Pass)                                                   │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  STAGE 3: Schema Validation                                   │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │ │
│  │  │   State     │  │   Shape     │  │   Dtype     │           │ │
│  │  │  Dict Keys  │  │  Validation │  │   Check     │           │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘           │ │
│  └───────────────────────────────────────────────────────────────┘ │
│          │                                                          │
│          ▼ (Pass)                                                   │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  STAGE 4: Cryptographic Verification (Optional)               │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │  Verify SHA256 signature against trusted manifest       │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│          │                                                          │
│          ▼                                                          │
│  ┌───────────────┐                                                  │
│  │   Validated   │ ──────► Ready for Inference                     │
│  │  Checkpoint   │                                                  │
│  └───────────────┘                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Validation Script

```python
# src/safety/checkpoint_validator.py
"""
Checkpoint validation with defense-in-depth.

This module provides secure checkpoint loading by:
1. Static analysis of pickle opcodes
2. Sandboxed deserialization with RestrictedUnpickler
3. Schema validation of state dict structure
4. Optional cryptographic verification
"""

import hashlib
import io
import pickle
import pickletools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ValidationResult:
    """Result of checkpoint validation."""

    valid: bool
    checkpoint_hash: str
    errors: list[str]
    warnings: list[str]
    metadata: dict[str, Any]


# Allowlist of safe classes for unpickling
SAFE_CLASSES = {
    # PyTorch core
    ("torch", "FloatTensor"),
    ("torch", "LongTensor"),
    ("torch", "IntTensor"),
    ("torch", "DoubleTensor"),
    ("torch", "HalfTensor"),
    ("torch", "BFloat16Tensor"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_parameter_with_state"),
    ("torch.storage", "_load_from_bytes"),
    ("torch.storage", "TypedStorage"),
    ("torch.storage", "UntypedStorage"),
    # Collections
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    # Numpy (read-only operations)
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
}

# Dangerous pickle opcodes
DANGEROUS_OPCODES = {
    "GLOBAL",      # Can import arbitrary modules
    "INST",        # Can instantiate arbitrary classes
    "OBJ",         # Can call arbitrary constructors
    "REDUCE",      # Can call arbitrary callables
    "BUILD",       # Can call __setstate__ with arbitrary data
    "EXT1", "EXT2", "EXT4",  # Extension registry (untrusted)
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows safe classes."""

    def find_class(self, module: str, name: str) -> type:
        if (module, name) in SAFE_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Forbidden class: {module}.{name}"
        )


def analyze_pickle_opcodes(data: bytes) -> tuple[bool, list[str]]:
    """
    Static analysis of pickle opcodes without executing.

    Returns (is_safe, list of warnings/errors).
    """
    warnings = []

    try:
        ops = list(pickletools.genops(data))
    except Exception as e:
        return False, [f"Failed to parse pickle: {e}"]

    for op, arg, pos in ops:
        opname = op.name

        if opname in DANGEROUS_OPCODES:
            if opname == "REDUCE":
                # REDUCE is used legitimately by PyTorch
                # Check if it's calling a safe function
                warnings.append(
                    f"REDUCE opcode at position {pos} "
                    f"(common in PyTorch, will be validated during load)"
                )
            else:
                return False, [f"Dangerous opcode {opname} at position {pos}"]

        if opname == "GLOBAL":
            # Check module.name against allowlist
            if arg:
                module, name = arg.rsplit(".", 1) if "." in arg else ("", arg)
                # Will be validated by RestrictedUnpickler
                pass

    return True, warnings


def validate_state_dict_schema(
    state_dict: dict,
    expected_keys: set[str] | None = None
) -> tuple[bool, list[str]]:
    """Validate state dict structure and tensor properties."""
    errors = []

    # Check it's a dict
    if not isinstance(state_dict, dict):
        return False, ["State dict is not a dictionary"]

    # Check for expected keys
    if expected_keys:
        missing = expected_keys - set(state_dict.keys())
        extra = set(state_dict.keys()) - expected_keys
        if missing:
            errors.append(f"Missing keys: {missing}")
        if extra:
            errors.append(f"Unexpected keys: {extra}")

    # Validate tensor properties
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Check for NaN/Inf (potential attack or corruption)
            if torch.isnan(value).any():
                errors.append(f"Tensor {key} contains NaN")
            if torch.isinf(value).any():
                errors.append(f"Tensor {key} contains Inf")

            # Check reasonable size (< 10GB per tensor)
            size_gb = value.numel() * value.element_size() / (1024**3)
            if size_gb > 10:
                errors.append(f"Tensor {key} is too large: {size_gb:.2f}GB")

    return len(errors) == 0, errors


def compute_checkpoint_hash(data: bytes) -> str:
    """Compute SHA256 hash of checkpoint data."""
    return hashlib.sha256(data).hexdigest()


def validate_checkpoint(
    checkpoint_path: Path,
    expected_hash: str | None = None,
    expected_keys: set[str] | None = None,
    max_size_gb: float = 50.0
) -> ValidationResult:
    """
    Comprehensive checkpoint validation.

    Args:
        checkpoint_path: Path to checkpoint file
        expected_hash: Expected SHA256 hash (optional)
        expected_keys: Expected state dict keys (optional)
        max_size_gb: Maximum allowed file size in GB

    Returns:
        ValidationResult with validation status and details
    """
    errors = []
    warnings = []
    metadata = {}

    # Stage 0: File existence and size
    if not checkpoint_path.exists():
        return ValidationResult(
            valid=False,
            checkpoint_hash="",
            errors=["Checkpoint file does not exist"],
            warnings=[],
            metadata={}
        )

    file_size = checkpoint_path.stat().st_size
    file_size_gb = file_size / (1024**3)
    metadata["file_size_gb"] = file_size_gb

    if file_size_gb > max_size_gb:
        return ValidationResult(
            valid=False,
            checkpoint_hash="",
            errors=[f"File too large: {file_size_gb:.2f}GB > {max_size_gb}GB"],
            warnings=[],
            metadata=metadata
        )

    # Stage 1: Read and hash
    try:
        with open(checkpoint_path, "rb") as f:
            data = f.read()
    except Exception as e:
        return ValidationResult(
            valid=False,
            checkpoint_hash="",
            errors=[f"Failed to read file: {e}"],
            warnings=[],
            metadata=metadata
        )

    checkpoint_hash = compute_checkpoint_hash(data)
    metadata["sha256"] = checkpoint_hash

    # Stage 1b: Hash verification
    if expected_hash and checkpoint_hash != expected_hash:
        errors.append(
            f"Hash mismatch: expected {expected_hash}, got {checkpoint_hash}"
        )

    # Stage 2: Static pickle analysis
    is_safe, pickle_warnings = analyze_pickle_opcodes(data)
    warnings.extend(pickle_warnings)

    if not is_safe:
        errors.extend(pickle_warnings)
        return ValidationResult(
            valid=False,
            checkpoint_hash=checkpoint_hash,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

    # Stage 3: Sandboxed deserialization
    try:
        buffer = io.BytesIO(data)
        # Use weights_only=True as first line of defense
        checkpoint = torch.load(
            buffer,
            map_location="cpu",
            weights_only=True
        )
    except Exception as e:
        # If weights_only fails, try RestrictedUnpickler
        try:
            buffer = io.BytesIO(data)
            # Skip the magic number and protocol
            unpickler = RestrictedUnpickler(buffer)
            checkpoint = unpickler.load()
        except Exception as e2:
            errors.append(f"Failed to deserialize: {e2}")
            return ValidationResult(
                valid=False,
                checkpoint_hash=checkpoint_hash,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )

    # Stage 4: Schema validation
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    schema_valid, schema_errors = validate_state_dict_schema(
        state_dict, expected_keys
    )
    errors.extend(schema_errors)

    # Extract metadata
    metadata["keys"] = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    if isinstance(checkpoint, dict):
        metadata["version"] = checkpoint.get("version", "unknown")
        metadata["step"] = checkpoint.get("step", "unknown")

    return ValidationResult(
        valid=len(errors) == 0,
        checkpoint_hash=checkpoint_hash,
        errors=errors,
        warnings=warnings,
        metadata=metadata
    )


def main():
    """CLI entry point for checkpoint validation."""
    if len(sys.argv) < 2:
        print("Usage: python checkpoint_validator.py <checkpoint_path> [expected_hash]")
        sys.exit(1)

    checkpoint_path = Path(sys.argv[1])
    expected_hash = sys.argv[2] if len(sys.argv) > 2 else None

    result = validate_checkpoint(checkpoint_path, expected_hash)

    print(f"Valid: {result.valid}")
    print(f"Hash: {result.checkpoint_hash}")

    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    print(f"Metadata: {result.metadata}")

    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
```

### 5.3 SafeTensors Migration Path

```python
# src/safety/safetensors_converter.py
"""
Convert PyTorch checkpoints to SafeTensors format.

SafeTensors provides memory-safe serialization without arbitrary code execution.
"""

from pathlib import Path

import torch
from safetensors.torch import save_file, load_file


def convert_to_safetensors(
    pytorch_path: Path,
    safetensors_path: Path,
    include_metadata: bool = True
) -> dict:
    """
    Convert PyTorch checkpoint to SafeTensors format.

    Note: SafeTensors only stores tensors, not arbitrary Python objects.
    Metadata is stored as string key-value pairs.
    """
    # Load with weights_only for safety
    checkpoint = torch.load(pytorch_path, map_location="cpu", weights_only=True)

    # Extract state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Prepare metadata
    metadata = {}
    if include_metadata and isinstance(checkpoint, dict):
        if "step" in checkpoint:
            metadata["step"] = str(checkpoint["step"])
        if "version" in checkpoint:
            metadata["version"] = str(checkpoint["version"])

    # Save as SafeTensors
    save_file(state_dict, safetensors_path, metadata=metadata)

    return {
        "original_path": str(pytorch_path),
        "safetensors_path": str(safetensors_path),
        "num_tensors": len(state_dict),
        "metadata": metadata
    }


def load_safetensors(path: Path, device: str = "cpu") -> dict:
    """Load SafeTensors checkpoint safely."""
    return load_file(path, device=device)
```

---

## 6. Docker Compose Configuration

### 6.1 Production Compose File

```yaml
# docker-compose.prod.yaml
version: "3.9"

x-security-defaults: &security-defaults
  security_opt:
    - no-new-privileges:true
    - seccomp:./docker/seccomp/default.json
  read_only: true
  cap_drop:
    - ALL

x-resource-limits: &resource-limits
  deploy:
    resources:
      limits:
        memory: 8G
        cpus: "4"
        pids: 100
      reservations:
        memory: 4G
        cpus: "2"

services:
  # Checkpoint validation sandbox
  checkpoint-validator:
    build:
      context: .
      dockerfile: docker/Dockerfile.validator
    <<: *security-defaults
    network_mode: "none"  # Complete network isolation
    volumes:
      - type: bind
        source: ./checkpoints/incoming
        target: /checkpoints
        read_only: true
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: "1"
        reservations:
          memory: 512M
    tmpfs:
      - /tmp:size=100M,mode=1777

  # Inference service (Go AI)
  inference:
    build:
      context: .
      dockerfile: docker/Dockerfile.inference
    <<: [*security-defaults, *resource-limits]
    networks:
      - internal
    volumes:
      - type: bind
        source: ./checkpoints/validated
        target: /models
        read_only: true
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - CUDA_CACHE_DISABLE=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "python", "-c", "import torch; assert torch.cuda.is_available()"]
      interval: 30s
      timeout: 10s
      retries: 3
    tmpfs:
      - /tmp:size=1G,mode=1777

  # API Gateway (Gradio)
  api-server:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    <<: *security-defaults
    networks:
      - internal
      - ingress
    ports:
      - "127.0.0.1:7860:7860"  # Only localhost
    depends_on:
      inference:
        condition: service_healthy
    environment:
      - INFERENCE_HOST=inference
      - INFERENCE_PORT=50051
    tmpfs:
      - /tmp:size=100M,mode=1777

  # Video codec service
  codec:
    build:
      context: .
      dockerfile: docker/Dockerfile.codec
    <<: [*security-defaults, *resource-limits]
    networks:
      - internal
    volumes:
      - type: bind
        source: ./checkpoints/validated/codec
        target: /models
        read_only: true
      - type: bind
        source: ./data/input
        target: /input
        read_only: true
      - type: bind
        source: ./data/output
        target: /output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    tmpfs:
      - /tmp:size=2G,mode=1777

networks:
  internal:
    driver: bridge
    internal: true
  ingress:
    driver: bridge

volumes:
  checkpoints-validated:
    driver: local
```

### 6.2 Training Compose File

```yaml
# docker-compose.train.yaml
version: "3.9"

services:
  trainer:
    build:
      context: .
      dockerfile: docker/Dockerfile.trainer
    runtime: nvidia
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
      - WANDB_PROJECT=alphagalerkin
      - WANDB_MODE=${WANDB_MODE:-online}
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./outputs:/app/outputs
      - ./config:/app/config:ro
    networks:
      - egress
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      python -m scripts.train
      --config-name=train
      experiment_name=docker_train

  # Multi-node distributed training
  trainer-dist:
    build:
      context: .
      dockerfile: docker/Dockerfile.trainer
    runtime: nvidia
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
      - MASTER_ADDR=trainer-dist
      - MASTER_PORT=29500
      - WORLD_SIZE=${WORLD_SIZE:-1}
      - RANK=${RANK:-0}
      - LOCAL_RANK=${LOCAL_RANK:-0}
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./outputs:/app/outputs
    networks:
      - internal
      - egress
    deploy:
      replicas: ${WORLD_SIZE:-1}
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

networks:
  internal:
    driver: bridge
    internal: true
  egress:
    driver: bridge
```

### 6.3 Development Compose File

```yaml
# docker-compose.dev.yaml
version: "3.9"

services:
  dev:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    volumes:
      - .:/app
      - ~/.cache/torch:/root/.cache/torch
    environment:
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
    ports:
      - "7860:7860"   # Gradio
      - "6006:6006"   # TensorBoard
    runtime: nvidia
    command: bash
    stdin_open: true
    tty: true
```

---

## 7. Secrets Management

### 7.1 Docker Secrets (Swarm Mode)

```yaml
# docker-compose.secrets.yaml
version: "3.9"

secrets:
  wandb_api_key:
    external: true
  gcs_credentials:
    file: ./secrets/gcs-credentials.json
  hf_token:
    external: true

services:
  trainer:
    secrets:
      - wandb_api_key
      - gcs_credentials
    environment:
      - WANDB_API_KEY_FILE=/run/secrets/wandb_api_key
      - GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcs_credentials
```

### 7.2 Environment Variable Handling

```bash
#!/bin/bash
# scripts/docker-entrypoint.sh

# Read secrets from files if present
if [ -f "/run/secrets/wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat /run/secrets/wandb_api_key)
fi

if [ -f "/run/secrets/hf_token" ]; then
    export HF_TOKEN=$(cat /run/secrets/hf_token)
fi

# Validate required environment
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"

exec "$@"
```

### 7.3 .env Template

```bash
# .env.template (copy to .env and fill in)

# Weights & Biases
WANDB_API_KEY=
WANDB_PROJECT=alphagalerkin
WANDB_ENTITY=
WANDB_MODE=online  # online, offline, disabled

# HuggingFace
HF_TOKEN=

# Google Cloud (for Vertex AI)
GOOGLE_APPLICATION_CREDENTIALS=
VERTEX_PROJECT=
VERTEX_REGION=us-central1
VERTEX_BUCKET=

# Training
CUDA_VISIBLE_DEVICES=0
```

---

## 8. CI/CD Integration

### 8.1 GitHub Actions Workflow

```yaml
# .github/workflows/docker-build.yaml
name: Docker Build and Security Scan

on:
  push:
    branches: [main]
    paths:
      - 'docker/**'
      - 'requirements*.txt'
      - 'pyproject.toml'
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      security-events: write

    strategy:
      matrix:
        target: [inference, trainer, validator]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/${{ matrix.target }}
          tags: |
            type=sha,prefix=
            type=ref,event=branch
            type=semver,pattern={{version}}

      - name: Build image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.${{ matrix.target }}
          push: false
          load: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/${{ matrix.target }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
          ignore-unfixed: true

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Run Dockle linter
        uses: erzz/dockle-action@v1
        with:
          image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/${{ matrix.target }}:${{ github.sha }}
          failure-threshold: high
          exit-code: 1

      - name: Push image (on main only)
        if: github.ref == 'refs/heads/main'
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.${{ matrix.target }}
          push: true
          tags: ${{ steps.meta.outputs.tags }}

  integration-test:
    needs: build-and-scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run integration tests
        run: |
          docker compose -f docker-compose.test.yaml up --build --abort-on-container-exit
          docker compose -f docker-compose.test.yaml down -v
```

### 8.2 Pre-commit Hooks for Docker

```yaml
# .pre-commit-config.yaml (additions)
repos:
  - repo: https://github.com/hadolint/hadolint
    rev: v2.12.0
    hooks:
      - id: hadolint
        args: ['--ignore', 'DL3008', '--ignore', 'DL3013']

  - repo: https://github.com/IamTheFij/docker-pre-commit
    rev: v3.0.1
    hooks:
      - id: docker-compose-check
        files: docker-compose.*\.ya?ml$
```

---

## 9. Implementation Phases

### Phase 1: Foundation (Week 1-2)

- [ ] Create `docker/Dockerfile.inference` (hardened inference container)
- [ ] Create `docker/Dockerfile.trainer` (training container)
- [ ] Create `docker/Dockerfile.validator` (checkpoint sandbox)
- [ ] Create `.dockerignore` file
- [ ] Create `requirements-lock.txt` with pinned versions
- [ ] Implement `src/safety/checkpoint_validator.py`
- [ ] Add basic `docker-compose.yaml`

### Phase 2: Security Hardening (Week 2-3)

- [ ] Create seccomp profiles for each container type
- [ ] Implement network isolation policies
- [ ] Add resource limits and health checks
- [ ] Implement secrets management
- [ ] Create SafeTensors converter utility
- [ ] Add checkpoint signature verification

### Phase 3: CI/CD Integration (Week 3-4)

- [ ] Add GitHub Actions workflow for Docker builds
- [ ] Integrate Trivy vulnerability scanning
- [ ] Add Dockle linting
- [ ] Create automated integration tests
- [ ] Set up container registry (GHCR/Artifact Registry)

### Phase 4: Production Deployment (Week 4-5)

- [ ] Create production docker-compose configurations
- [ ] Document deployment procedures
- [ ] Create runbooks for common operations
- [ ] Implement monitoring and alerting
- [ ] Performance benchmarking

### Phase 5: Advanced Features (Week 5+)

- [ ] Kubernetes manifests (optional)
- [ ] Helm charts (optional)
- [ ] GPU sharing with MPS/MIG
- [ ] Distributed training with Horovod/DeepSpeed

---

## 10. Verification Checklist

### Container Security

- [ ] All containers run as non-root user
- [ ] Read-only root filesystem where possible
- [ ] No capabilities beyond minimum required
- [ ] Seccomp profiles applied
- [ ] Resource limits set
- [ ] Health checks configured

### Network Security

- [ ] Inference containers have no network access
- [ ] Training containers have egress-only access
- [ ] API containers have ingress-only access
- [ ] Internal networks are isolated

### Checkpoint Security

- [ ] Static pickle analysis passes
- [ ] Sandbox deserialization succeeds
- [ ] Schema validation passes
- [ ] Hash verification (if provided)
- [ ] SafeTensors migration path documented

### Supply Chain Security

- [ ] Base images pinned by digest
- [ ] Dependencies pinned by version
- [ ] Vulnerability scanning in CI
- [ ] SBOM generation enabled

### Secrets Management

- [ ] No secrets in Dockerfiles
- [ ] No secrets in docker-compose files
- [ ] Docker secrets or env files used
- [ ] .env.template provided

---

## Appendix A: Quick Start Commands

```bash
# Build all containers
docker compose build

# Run checkpoint validator
docker compose run --rm checkpoint-validator /checkpoints/model.pt

# Start inference service
docker compose up -d inference api-server

# Train with GPU
docker compose -f docker-compose.train.yaml up trainer

# Run security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image alphagalerkin/inference:latest

# View container logs
docker compose logs -f inference

# Stop all services
docker compose down
```

---

## Appendix B: File Structure

```
docker/
├── Dockerfile.base         # Shared build stage
├── Dockerfile.inference    # Hardened inference container
├── Dockerfile.trainer      # Training container
├── Dockerfile.validator    # Checkpoint validation sandbox
├── Dockerfile.api          # API server (Gradio)
├── Dockerfile.codec        # Video codec container
├── Dockerfile.dev          # Development container
├── seccomp/
│   ├── inference.json      # Inference seccomp profile
│   └── default.json        # Default seccomp profile
├── apparmor/
│   └── alphagalerkin       # AppArmor profile
└── scripts/
    └── docker-entrypoint.sh

docker-compose.yaml         # Default compose file
docker-compose.prod.yaml    # Production configuration
docker-compose.train.yaml   # Training configuration
docker-compose.dev.yaml     # Development configuration
docker-compose.test.yaml    # CI/CD testing

src/safety/
├── __init__.py
├── checkpoint_validator.py # Checkpoint validation
└── safetensors_converter.py # SafeTensors migration

.dockerignore
.env.template
```

---

## Appendix C: References

- [PyTorch Security Best Practices](https://pytorch.org/docs/stable/notes/security.html)
- [SafeTensors Documentation](https://huggingface.co/docs/safetensors)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
