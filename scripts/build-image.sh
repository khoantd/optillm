#!/usr/bin/env bash
# Build the OptiLLM full Docker image.
#
# Usage examples:
#   # Local build for current machine
#   ./scripts/build-image.sh
#
#   # Build amd64 on Apple Silicon, load locally
#   ./scripts/build-image.sh -p linux/amd64 -r optillm
#
#   # Push to GHCR (after docker login ghcr.io)
#   ./scripts/build-image.sh -r ghcr.io/algorithmicsuperintelligence/optillm --push
#
#   # Multi-arch manifest push
#   ./scripts/build-image.sh -p linux/amd64,linux/arm64 -r ghcr.io/algorithmicsuperintelligence/optillm --push

set -euo pipefail

BUILDER_NAME="optillm-builder"
DOCKERFILE="Dockerfile"
TAG="latest"
PORT=8000

usage() {
    cat <<'EOF'
Build the OptiLLM full Docker image.

Usage:
  ./scripts/build-image.sh [options]

Options:
  -p, --platform PLATFORM   Target platform(s), comma-separated
                            (default: native linux/amd64 or linux/arm64)
                            Accepts shorthand: amd64, arm64, linux/amd64, linux/arm64
  -r, --registry REGISTRY   Image name without tag (default: optillm)
  --push                    Push to registry instead of loading locally
  -h, --help                Show this help message

Notes:
  - Without --push, the image is loaded locally as REGISTRY:latest via buildx --load.
  - Multi-platform builds require --push; they cannot be loaded into local Docker.
  - For --push, log in first: docker login <registry-host>
  - Cross-arch builds may require QEMU; buildx handles this when configured.
EOF
}

native_platform() {
    case "$(uname -m)" in
        x86_64|amd64) echo "linux/amd64" ;;
        aarch64|arm64) echo "linux/arm64" ;;
        *)
            echo "Unsupported architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac
}

normalize_platform() {
    local input="$1"
    local normalized=""
    local part trimmed

    IFS=',' read -ra parts <<< "$input"
    for part in "${parts[@]}"; do
        trimmed="${part#"${part%%[![:space:]]*}"}"
        trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"

        case "$trimmed" in
            amd64|x86_64) trimmed="linux/amd64" ;;
            arm64|aarch64) trimmed="linux/arm64" ;;
            linux/amd64|linux/arm64) ;;
            *)
                echo "Unsupported platform: $part" >&2
                echo "Use amd64, arm64, linux/amd64, or linux/arm64." >&2
                exit 1
                ;;
        esac

        if [[ -n "$normalized" ]]; then
            normalized+=",$trimmed"
        else
            normalized="$trimmed"
        fi
    done

    echo "$normalized"
}

is_multi_platform() {
    [[ "$1" == *","* ]]
}

ensure_buildx() {
    if ! docker buildx version >/dev/null 2>&1; then
        echo "docker buildx is required but not available." >&2
        exit 1
    fi

    if docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
        docker buildx use "$BUILDER_NAME" >/dev/null
        return
    fi

    echo "Creating buildx builder: $BUILDER_NAME"
    docker buildx create --name "$BUILDER_NAME" --use >/dev/null
    docker buildx inspect --bootstrap >/dev/null
}

validate_options() {
    if is_multi_platform "$PLATFORM" && [[ "$PUSH" != "true" ]]; then
        echo "Multi-platform builds require --push; buildx cannot load multi-arch images locally." >&2
        exit 1
    fi
}

PLATFORM="$(native_platform)"
REGISTRY="optillm"
PUSH="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--platform)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            PLATFORM="$2"
            shift 2
            ;;
        -r|--registry)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            REGISTRY="$2"
            shift 2
            ;;
        --push)
            PUSH="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PLATFORM="$(normalize_platform "$PLATFORM")"
validate_options
ensure_buildx

IMAGE_REF="${REGISTRY}:${TAG}"

echo "Building OptiLLM Docker image"
echo "  Context:    $REPO_ROOT"
echo "  Dockerfile: $DOCKERFILE"
echo "  Platform:   $PLATFORM"
echo "  Image:      $IMAGE_REF"
echo "  Push:       $PUSH"
echo

build_args=(
    buildx build
    --file "$DOCKERFILE"
    --platform "$PLATFORM"
    --build-arg "PORT=$PORT"
    --tag "$IMAGE_REF"
    --provenance=false
)

if [[ "$PUSH" == "true" ]]; then
    build_args+=(--push)
else
    build_args+=(--load)
fi

build_args+=(.)

if ! docker "${build_args[@]}"; then
    if [[ "$PUSH" == "true" ]]; then
        echo >&2
        echo "Build/push failed. If authentication failed, log in first:" >&2
        echo "  docker login <registry-host>" >&2
    fi
    exit 1
fi

echo
echo "Build complete: $IMAGE_REF"
if [[ "$PUSH" == "true" ]]; then
    echo "Image pushed to registry."
else
    echo "Image loaded locally. Next steps:"
    echo "  docker run -p 8000:8000 $IMAGE_REF"
    echo "  docker compose up -d"
fi
