#!/bin/bash
# Install amortized on a K8s cluster
#
# Usage:
#   ./install.sh dev      # local dev (includes MinIO + MLflow)
#   ./install.sh rosa     # OpenShift/ROSA (bring your own MLflow + S3)
#
set -euo pipefail

OVERLAY="${1:-dev}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -d "$SCRIPT_DIR/overlays/$OVERLAY" ]]; then
  echo "Unknown overlay: $OVERLAY"
  echo "Available: $(ls "$SCRIPT_DIR/overlays/")"
  exit 1
fi

echo "Installing amortized ($OVERLAY overlay)..."
kubectl apply -k "$SCRIPT_DIR/overlays/$OVERLAY"

echo ""
echo "Waiting for deployments..."
kubectl -n amortized wait --for=condition=Available deployment --all --timeout=120s

echo ""
echo "Amortized is running:"
kubectl -n amortized get pods
echo ""
echo "API:    kubectl -n amortized port-forward svc/amortized-server 8000:8000"
echo "Studio: kubectl -n amortized port-forward svc/amortized-studio 8080:8080"
