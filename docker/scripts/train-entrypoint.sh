#!/bin/bash
# train-entrypoint.sh - Entrypoint script for training container
#
# Features:
# - Secret loading from files
# - Environment validation
# - Graceful shutdown handling
# - Distributed training setup

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Secret file locations (Docker secrets mount point)
SECRETS_DIR="${SECRETS_DIR:-/run/secrets}"

# =============================================================================
# Functions
# =============================================================================

log() {
    echo "[entrypoint] $(date -Iseconds) $*"
}

error() {
    echo "[entrypoint] ERROR: $*" >&2
}

# Load secret from file if exists
load_secret() {
    local var_name="$1"
    local file_path="$2"

    if [[ -f "$file_path" ]]; then
        export "$var_name"="$(cat "$file_path")"
        log "Loaded $var_name from $file_path"
    fi
}

# Setup signal handlers for graceful shutdown
setup_signals() {
    # Trap SIGTERM and SIGINT for graceful shutdown
    trap 'log "Received SIGTERM, shutting down..."; kill -TERM "$child_pid" 2>/dev/null; wait "$child_pid"' SIGTERM
    trap 'log "Received SIGINT, shutting down..."; kill -INT "$child_pid" 2>/dev/null; wait "$child_pid"' SIGINT
}

# Validate required environment variables
validate_env() {
    local missing=()

    # Check if WANDB_API_KEY is set (required for online mode)
    if [[ "${WANDB_MODE:-online}" == "online" ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
        missing+=("WANDB_API_KEY (required for WANDB_MODE=online)")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing required environment variables:"
        for var in "${missing[@]}"; do
            error "  - $var"
        done
        error ""
        error "Either set these variables or mount secrets to $SECRETS_DIR/"
        return 1
    fi
}

# Setup distributed training environment
setup_distributed() {
    # Auto-detect Vertex AI environment
    if [[ -n "${CLUSTER_SPEC:-}" ]]; then
        log "Detected Vertex AI distributed training environment"
        # Vertex AI sets these automatically
        return 0
    fi

    # Auto-detect Kubernetes environment
    if [[ -n "${KUBERNETES_SERVICE_HOST:-}" ]]; then
        log "Detected Kubernetes environment"
        # Use pod IP as node address if not set
        if [[ -z "${MASTER_ADDR:-}" ]]; then
            export MASTER_ADDR="${POD_IP:-localhost}"
            log "Set MASTER_ADDR=$MASTER_ADDR"
        fi
    fi

    # Set defaults for single-node training
    export MASTER_ADDR="${MASTER_ADDR:-localhost}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    export WORLD_SIZE="${WORLD_SIZE:-1}"
    export RANK="${RANK:-0}"
    export LOCAL_RANK="${LOCAL_RANK:-0}"

    log "Distributed config: MASTER_ADDR=$MASTER_ADDR, WORLD_SIZE=$WORLD_SIZE, RANK=$RANK"
}

# Setup CUDA environment
setup_cuda() {
    # Set CUDA device order for consistent GPU mapping
    export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

    # Check CUDA availability
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local gpu_count
        gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
        log "CUDA available: $gpu_count GPU(s) detected"

        # Log GPU info
        for ((i=0; i<gpu_count; i++)); do
            local gpu_name
            gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name($i))")
            log "  GPU $i: $gpu_name"
        done
    else
        log "WARNING: CUDA not available, running on CPU"
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    log "Starting AlphaGalerkin training container"
    log "Python: $(python --version)"
    log "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

    # Load secrets from files
    load_secret "WANDB_API_KEY" "$SECRETS_DIR/wandb_api_key"
    load_secret "HF_TOKEN" "$SECRETS_DIR/hf_token"
    load_secret "GOOGLE_APPLICATION_CREDENTIALS" "$SECRETS_DIR/gcs_credentials"

    # Setup environment
    setup_cuda
    setup_distributed

    # Validate environment
    if ! validate_env; then
        exit 1
    fi

    # Setup signal handlers
    setup_signals

    # Create output directories
    mkdir -p "${CHECKPOINT_DIR:-/app/checkpoints}" "${OUTPUT_DIR:-/app/outputs}"

    log "Executing command: $*"

    # Execute the command
    "$@" &
    child_pid=$!

    # Wait for the process
    wait "$child_pid"
    exit_code=$?

    log "Command exited with code $exit_code"
    exit $exit_code
}

# Run main with all arguments
main "$@"
