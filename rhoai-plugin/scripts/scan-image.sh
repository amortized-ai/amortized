#!/usr/bin/env bash
set -euo pipefail

# Configuration
IMAGE_TAG="${IMAGE_TAG:-latest}"
BUILDER="${BUILDER:-podman}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [TARGET] [SEVERITY]

Build and scan container images for vulnerabilities using Trivy.

Arguments:
  TARGET    Which image to scan: frontend, bff, or all (default: all)
  SEVERITY  Trivy severity filter (default: HIGH,CRITICAL)

Environment variables:
  IMAGE_TAG   Tag for built images (default: latest)
  BUILDER     Container build tool (default: podman)

Examples:
  $(basename "$0")                  # Scan both frontend and BFF
  $(basename "$0") frontend         # Scan frontend only
  $(basename "$0") bff              # Scan BFF only
  $(basename "$0") all MEDIUM       # Scan both with MEDIUM+ severity
  BUILDER=docker $(basename "$0")   # Use Docker instead of Podman
EOF
}

# Target configurations
frontend_image_name="amortized-studio"
frontend_containerfile="Containerfile"
frontend_context="."

bff_image_name="amortized-studio-bff"
bff_containerfile="bff/Containerfile"
bff_context="bff/"

# Check prerequisites
check_prerequisites() {
    local targets=("$@")
    local missing=0

    if ! command -v "${BUILDER}" &> /dev/null; then
        log_error "${BUILDER} is not installed or not in PATH"
        missing=1
    fi

    if ! command -v trivy &> /dev/null; then
        log_error "trivy is not installed or not in PATH"
        log_info "Install with: brew install aquasecurity/trivy/trivy"
        missing=1
    fi

    for target in "${targets[@]}"; do
        local containerfile
        if [[ "${target}" == "frontend" ]]; then
            containerfile="${frontend_containerfile}"
        else
            containerfile="${bff_containerfile}"
        fi

        if [[ ! -f "${containerfile}" ]]; then
            log_error "Containerfile not found: ${containerfile}"
            missing=1
        fi
    done

    if [[ ${missing} -eq 1 ]]; then
        exit 1
    fi
}

# Build a container image
build_image() {
    local image_name="$1"
    local containerfile="$2"
    local context="$3"
    local full_image="${image_name}:${IMAGE_TAG}"

    log_info "Building image: ${full_image}"
    ${BUILDER} build -t "${full_image}" -f "${containerfile}" "${context}"
    log_success "Image built successfully: ${full_image}"
}

# Scan an image for vulnerabilities
scan_image() {
    local image_name="$1"
    local severity="$2"
    local full_image="${image_name}:${IMAGE_TAG}"

    log_info "Scanning image: ${full_image}"

    trivy image --severity "${severity}" --format table "${full_image}"

    local exit_code=$?
    if [[ ${exit_code} -eq 0 ]]; then
        log_success "No ${severity} vulnerabilities found in ${full_image}."
    else
        log_error "Vulnerabilities detected in ${full_image} (exit code: ${exit_code})"
    fi

    return ${exit_code}
}

# Build and scan a single target
process_target() {
    local target="$1"
    local severity="$2"
    local image_name containerfile context

    if [[ "${target}" == "frontend" ]]; then
        image_name="${frontend_image_name}"
        containerfile="${frontend_containerfile}"
        context="${frontend_context}"
    else
        image_name="${bff_image_name}"
        containerfile="${bff_containerfile}"
        context="${bff_context}"
    fi

    echo ""
    log_info "--- ${target} ---"
    build_image "${image_name}" "${containerfile}" "${context}"
    echo ""
    scan_image "${image_name}" "${severity}"
}

# Main
main() {
    local target="${1:-all}"
    local severity="${2:-HIGH,CRITICAL}"

    if [[ "${target}" == "-h" || "${target}" == "--help" ]]; then
        usage
        exit 0
    fi

    case "${target}" in
        frontend|bff|all) ;;
        *)
            log_error "Unknown target: ${target}"
            usage
            exit 1
            ;;
    esac

    local targets=()
    if [[ "${target}" == "all" ]]; then
        targets=("frontend" "bff")
    else
        targets=("${target}")
    fi

    echo "============================================"
    echo "  Container Image Build & Vulnerability Scan"
    echo "============================================"
    echo ""
    log_info "Target: ${target}"
    log_info "Builder: ${BUILDER}"
    log_info "Image tag: ${IMAGE_TAG}"
    log_info "Severity filter: ${severity}"

    check_prerequisites "${targets[@]}"

    local failed=0
    for t in "${targets[@]}"; do
        if ! process_target "${t}" "${severity}"; then
            failed=1
        fi
    done

    echo ""
    if [[ ${failed} -eq 0 ]]; then
        log_success "All scans passed."
    else
        log_error "One or more scans reported vulnerabilities."
        exit 1
    fi
}

main "$@"
