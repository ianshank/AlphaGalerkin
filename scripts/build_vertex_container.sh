#!/bin/bash
# Build and push AlphaGalerkin training container to Google Artifact Registry
#
# Usage:
#   ./scripts/build_vertex_container.sh                    # Use defaults from gcloud config
#   ./scripts/build_vertex_container.sh my-project         # Specify project
#   ./scripts/build_vertex_container.sh my-project us-west1 v1.0.0  # Full options
#
# Environment variables:
#   VERTEX_PROJECT: GCP project ID (overrides arg)
#   VERTEX_REGION: GCP region (default: us-central1)
#   VERTEX_REPO: Artifact Registry repository name (default: alphagalerkin)
#   VERTEX_IMAGE_TAG: Container image tag (default: latest)
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Docker installed and running
#   - Artifact Registry API enabled

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
DEFAULT_REGION="us-central1"
DEFAULT_REPO="alphagalerkin"
DEFAULT_TAG="latest"

# Parse arguments
PROJECT_ID="${1:-${VERTEX_PROJECT:-}}"
REGION="${2:-${VERTEX_REGION:-$DEFAULT_REGION}}"
IMAGE_TAG="${3:-${VERTEX_IMAGE_TAG:-$DEFAULT_TAG}}"
REPO_NAME="${VERTEX_REPO:-$DEFAULT_REPO}"

# Get project from gcloud if not specified
if [[ -z "$PROJECT_ID" ]]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
    if [[ -z "$PROJECT_ID" ]]; then
        echo "Error: Project ID not specified and not found in gcloud config"
        echo "Usage: $0 PROJECT_ID [REGION] [TAG]"
        exit 1
    fi
    echo "Using project from gcloud config: $PROJECT_ID"
fi

# Build image URI
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/trainer:${IMAGE_TAG}"

echo "=============================================="
echo "AlphaGalerkin Vertex AI Container Build"
echo "=============================================="
echo "Project:     $PROJECT_ID"
echo "Region:      $REGION"
echo "Repository:  $REPO_NAME"
echo "Tag:         $IMAGE_TAG"
echo "Image URI:   $IMAGE_URI"
echo "=============================================="
echo ""

# Check if running from project root
if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    echo "Error: Must run from project root or scripts directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Check Docker is running
if ! docker info &>/dev/null; then
    echo "Error: Docker is not running or not accessible"
    exit 1
fi

# Check gcloud authentication
if ! gcloud auth print-access-token &>/dev/null; then
    echo "Error: Not authenticated with gcloud"
    echo "Run: gcloud auth login"
    exit 1
fi

# Configure Docker for Artifact Registry
echo "Configuring Docker for Artifact Registry..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Ensure Artifact Registry repository exists
echo "Checking Artifact Registry repository..."
if ! gcloud artifacts repositories describe "$REPO_NAME" \
    --project="$PROJECT_ID" \
    --location="$REGION" &>/dev/null; then
    echo "Creating Artifact Registry repository: $REPO_NAME"
    gcloud artifacts repositories create "$REPO_NAME" \
        --project="$PROJECT_ID" \
        --location="$REGION" \
        --repository-format=docker \
        --description="AlphaGalerkin training containers"
else
    echo "Repository already exists: $REPO_NAME"
fi

# Check if Dockerfile exists
DOCKERFILE="docker/Dockerfile.vertex"
if [[ ! -f "$DOCKERFILE" ]]; then
    echo "Error: Dockerfile not found: $DOCKERFILE"
    exit 1
fi

# Build the container
echo ""
echo "Building container..."
echo ""

docker build \
    --file "$DOCKERFILE" \
    --tag "$IMAGE_URI" \
    --tag "${REPO_NAME}:${IMAGE_TAG}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --progress=plain \
    .

echo ""
echo "Build complete. Image: $IMAGE_URI"
echo ""

# Push to Artifact Registry
echo "Pushing to Artifact Registry..."
docker push "$IMAGE_URI"

echo ""
echo "=============================================="
echo "Container pushed successfully!"
echo "=============================================="
echo ""
echo "Image URI: $IMAGE_URI"
echo ""
echo "To use this container for training:"
echo ""
echo "  python -m scripts.train_vertex \\"
echo "    --project $PROJECT_ID \\"
echo "    --bucket YOUR_BUCKET \\"
echo "    --container-uri $IMAGE_URI"
echo ""
echo "Or pull the image:"
echo "  docker pull $IMAGE_URI"
echo ""
