CLUSTER_NAME  ?= amortized
IMAGE_TAG     ?= kind-$(shell git rev-parse --short HEAD)
KUBECTL       := kubectl --context kind-$(CLUSTER_NAME)
STUDIO_DIR    ?= $(shell cd .. && pwd)/studio
STUDIO_REPO   ?= https://github.com/amortized-ai/studio
REPO_ROOT     := $(shell pwd)

# Third-party images to pre-load into kind
MINIO_IMAGE   ?= quay.io/minio/minio:latest
MLFLOW_IMAGE  ?= ghcr.io/mlflow/mlflow:latest
AWSCLI_IMAGE  ?= docker.io/amazon/aws-cli:latest
NVIDIA_DP_IMAGE ?= nvcr.io/nvidia/k8s-device-plugin:v0.19.3
TRAINING_IMAGE ?= ghcr.io/amortized-ai/training:latest
ASYNTH_IMAGE  ?= ghcr.io/amortized-ai/asynth:latest
OPENCODE_IMAGE ?= ghcr.io/anomalyco/opencode:latest

# Source cluster for OpenCode credentials (existing deployment)
CREDS_CLUSTER ?= kind-amortized-dev

.PHONY: help up build build-server build-studio pull-images \
        load load-server load-studio load-deps \
        deploy deploy-dev \
        test-server test-studio \
        cluster gpu \
        down destroy status

# ──────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Full setup
# ──────────────────────────────────────────────

up: cluster gpu build load deploy status ## Create cluster, build images, deploy prod stack

# ──────────────────────────────────────────────
# Cluster lifecycle
# ──────────────────────────────────────────────

cluster: ## Create kind cluster with GPU support
	@if kind get clusters 2>/dev/null | grep -q "^$(CLUSTER_NAME)$$"; then \
		echo "Cluster '$(CLUSTER_NAME)' already exists, skipping."; \
	else \
		echo "Creating kind cluster '$(CLUSTER_NAME)'..."; \
		kind create cluster --name $(CLUSTER_NAME) --config k8s/kind/kind-config.yaml; \
	fi

gpu: ## Install NVIDIA runtime in worker + deploy device plugin
	@echo "Installing NVIDIA Container Toolkit in worker node..."
	@docker exec $(CLUSTER_NAME)-worker bash -c '\
		apt-get update -qq && \
		apt-get install -y -qq curl gpg >/dev/null 2>&1 && \
		curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null && \
		curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
			sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" > /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
		apt-get update -qq && \
		apt-get install -y -qq nvidia-container-toolkit >/dev/null 2>&1 && \
		nvidia-ctk runtime configure --runtime=containerd --set-as-default && \
		systemctl restart containerd && \
		sleep 3 && \
		systemctl restart kubelet'
	@echo "Loading NVIDIA device plugin image..."
	@docker pull $(NVIDIA_DP_IMAGE) 2>/dev/null || true
	@kind load docker-image $(NVIDIA_DP_IMAGE) --name $(CLUSTER_NAME) 2>/dev/null
	$(KUBECTL) apply -f k8s/kind/nvidia-device-plugin.yaml
	@echo "Waiting for NVIDIA device plugin..."
	@$(KUBECTL) -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=120s
	@echo "Waiting for GPU detection..."
	@sleep 15
	@echo "GPU allocatable:"
	@$(KUBECTL) get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'

# ──────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────

build: build-server build-studio pull-images ## Build all images

build-server: ## Build amortized server image
	@echo "Building amortized-server:$(IMAGE_TAG)..."
	docker build -t amortized-server:$(IMAGE_TAG) -f Dockerfile .

build-studio: ## Build studio image (expects ../studio/)
	@if [ ! -d "$(STUDIO_DIR)" ]; then \
		echo "Cloning studio repository to $(STUDIO_DIR)..."; \
		git clone $(STUDIO_REPO) $(STUDIO_DIR); \
	fi
	@cp studio/Dockerfile $(STUDIO_DIR)/Dockerfile.kind
	@cp studio/nginx.conf.template $(STUDIO_DIR)/nginx.conf.template
	@echo "Building amortized-studio:$(IMAGE_TAG)..."
	docker build -t amortized-studio:$(IMAGE_TAG) -f $(STUDIO_DIR)/Dockerfile.kind $(STUDIO_DIR)

pull-images: ## Pull third-party images (MinIO, MLflow, training, etc.)
	@echo "Pulling third-party images..."
	@for img in $(MINIO_IMAGE) $(MLFLOW_IMAGE) $(AWSCLI_IMAGE) $(OPENCODE_IMAGE); do \
		docker pull $$img 2>/dev/null || true; \
	done
	@echo "Pulling ML images (training image is ~12GB, this may take a while)..."
	@for img in $(TRAINING_IMAGE) $(ASYNTH_IMAGE); do \
		if ! docker image inspect $$img >/dev/null 2>&1; then \
			echo "  Pulling $$img..."; \
			docker pull $$img; \
		else \
			echo "  $$img already present."; \
		fi; \
	done

# ──────────────────────────────────────────────
# Load into kind
# ──────────────────────────────────────────────

load: load-server load-studio load-deps ## Load all images into kind

load-server: ## Load server image into kind
	kind load docker-image amortized-server:$(IMAGE_TAG) --name $(CLUSTER_NAME)

load-studio: ## Load studio image into kind
	kind load docker-image amortized-studio:$(IMAGE_TAG) --name $(CLUSTER_NAME)

load-deps: ## Load third-party images into kind
	@echo "Loading third-party images into kind..."
	@for img in $(MINIO_IMAGE) $(MLFLOW_IMAGE) $(AWSCLI_IMAGE) $(OPENCODE_IMAGE); do \
		kind load docker-image $$img --name $(CLUSTER_NAME) 2>/dev/null || true; \
	done
	@echo "Loading ML images into kind (training is ~12GB, be patient)..."
	@for img in $(TRAINING_IMAGE) $(ASYNTH_IMAGE); do \
		kind load docker-image $$img --name $(CLUSTER_NAME) 2>/dev/null || true; \
	done

# ──────────────────────────────────────────────
# Deploy prod
# ──────────────────────────────────────────────

deploy: ## Deploy prod stack (amortized namespace)
	@echo "Deploying prod stack..."
	@# Namespaces
	$(KUBECTL) apply -f k8s/base/namespace.yaml
	@# Base manifests (skip routes and opencode secret — creds come from existing cluster)
	@for f in k8s/base/*.yaml; do \
		case "$$(basename $$f)" in \
			*route*|opencode-secret.yaml|namespace.yaml) continue ;; \
		esac; \
		sed \
			-e 's|image: ghcr.io/amortized-ai/amortized:latest|image: amortized-server:$(IMAGE_TAG)|g' \
			-e 's|image: ghcr.io/amortized-ai/studio:latest|image: amortized-studio:$(IMAGE_TAG)|g' \
			-e 's|imagePullPolicy: Always|imagePullPolicy: IfNotPresent|g' \
			"$$f" | $(KUBECTL) apply -f -; \
	done
	@# Copy OpenCode credentials from existing cluster
	@echo "Copying OpenCode credentials from $(CREDS_CLUSTER)..."
	@for secret in opencode-gcp opencode-llm; do \
		kubectl --context $(CREDS_CLUSTER) -n amortized get secret $$secret -o json 2>/dev/null | \
			jq 'del(.metadata.resourceVersion,.metadata.uid,.metadata.creationTimestamp,.metadata.annotations)' | \
			$(KUBECTL) apply -f - 2>/dev/null || \
			echo "  Warning: could not copy $$secret from $(CREDS_CLUSTER). OpenCode may not start."; \
	done
	@# Dev infra: MinIO + MLflow (skip routes)
	@for f in k8s/dev/*.yaml; do \
		case "$$(basename $$f)" in \
			*route*) continue ;; \
		esac; \
		$(KUBECTL) apply -f "$$f"; \
	done
	@# Kind-specific: NodePorts + GPU quotas
	$(KUBECTL) apply -f k8s/kind/nodeport-services.yaml
	$(KUBECTL) apply -f k8s/kind/gpu-quota.yaml
	@# Patch prod configmap for kind
	$(KUBECTL) patch configmap amortized-config -n amortized --type merge \
		-p '{"data":{"AMORTIZED_MLFLOW_TRACKING_URI":"http://mlflow.amortized.svc.cluster.local:5000","AMORTIZED_S3_BUCKET":"amortized","AMORTIZED_IMAGE_PULL_POLICY":"IfNotPresent"}}'
	@# Kind-specific patches: remove runAsNonRoot (MinIO/MLflow run as root), fix opencode host
	@for dep in minio mlflow amortized-server amortized-studio opencode; do \
		$(KUBECTL) -n amortized patch deployment $$dep --type json \
			-p '[{"op":"remove","path":"/spec/template/spec/securityContext/runAsNonRoot"}]' 2>/dev/null || true; \
	done
	@# Create MinIO bucket
	@echo "Waiting for MinIO to be ready..."
	@$(KUBECTL) -n amortized rollout status deployment/minio --timeout=120s
	@$(KUBECTL) run minio-init --rm -i --restart=Never \
		--image=$(AWSCLI_IMAGE) \
		--namespace=amortized \
		--overrides='{"spec":{"containers":[{"name":"minio-init","image":"$(AWSCLI_IMAGE)","command":["sh","-c","aws --endpoint-url http://minio.amortized.svc.cluster.local:9000 s3 mb s3://amortized 2>/dev/null || true"],"env":[{"name":"AWS_ACCESS_KEY_ID","value":"minioadmin"},{"name":"AWS_SECRET_ACCESS_KEY","value":"minioadmin"}]}]}}' \
		|| echo "  Warning: could not create MinIO bucket. Create manually."
	@# Wait for rollouts
	@echo "Waiting for deployments..."
	@$(KUBECTL) -n amortized rollout status deployment/mlflow --timeout=120s
	@$(KUBECTL) -n amortized rollout status deployment/amortized-server --timeout=120s
	@$(KUBECTL) -n amortized rollout status deployment/amortized-studio --timeout=120s
	@$(KUBECTL) -n amortized rollout status deployment/opencode --timeout=120s
	@echo "Prod stack deployed."

# ──────────────────────────────────────────────
# Deploy dev
# ──────────────────────────────────────────────

deploy-dev: build-server build-studio load-server load-studio ## Build + deploy dev stack from current code
	@echo "Deploying dev stack..."
	@for f in k8s/kind/dev/*.yaml; do \
		sed \
			-e 's|image: ghcr.io/amortized-ai/amortized:latest|image: amortized-server:$(IMAGE_TAG)|g' \
			-e 's|image: ghcr.io/amortized-ai/studio:latest|image: amortized-studio:$(IMAGE_TAG)|g' \
			"$$f" | $(KUBECTL) apply -f -; \
	done
	$(KUBECTL) apply -f k8s/kind/dev/gpu-quota.yaml
	@echo "Waiting for dev deployments..."
	@$(KUBECTL) -n amortized-dev rollout status deployment/amortized-server --timeout=120s
	@$(KUBECTL) -n amortized-dev rollout status deployment/amortized-studio --timeout=120s
	@echo "Dev stack deployed."

# ──────────────────────────────────────────────
# PR testing shortcuts
# ──────────────────────────────────────────────

test-server: build-server load-server deploy-dev ## Build server from current branch + deploy to dev
	@$(KUBECTL) -n amortized-dev rollout restart deployment/amortized-server
	@$(KUBECTL) -n amortized-dev rollout status deployment/amortized-server --timeout=120s
	@echo "Dev server updated. API at http://localhost:31091"

test-studio: build-studio load-studio deploy-dev ## Build studio from current branch + deploy to dev
	@$(KUBECTL) -n amortized-dev rollout restart deployment/amortized-studio
	@$(KUBECTL) -n amortized-dev rollout status deployment/amortized-studio --timeout=120s
	@echo "Dev studio updated. UI at http://localhost:31090"

# ──────────────────────────────────────────────
# Teardown
# ──────────────────────────────────────────────

down: ## Delete dev namespaces (keep prod and cluster)
	@echo "Tearing down dev namespace..."
	$(KUBECTL) delete namespace amortized-dev amortized-dev-jobs --ignore-not-found
	@echo "Dev namespace removed. Prod untouched."

destroy: ## Delete the entire kind cluster
	@echo "Destroying kind cluster '$(CLUSTER_NAME)'..."
	kind delete cluster --name $(CLUSTER_NAME)

# ──────────────────────────────────────────────
# Status
# ──────────────────────────────────────────────

status: ## Show cluster status, pods, and access URLs
	@echo ""
	@echo "=== Cluster ==="
	@kind get clusters 2>/dev/null || echo "No kind clusters"
	@echo ""
	@echo "=== GPU Nodes ==="
	@$(KUBECTL) get nodes -o custom-columns="NAME:.metadata.name,GPUs:.status.allocatable.nvidia\.com/gpu" 2>/dev/null || true
	@echo ""
	@echo "=== Pods (amortized) ==="
	@$(KUBECTL) get pods -n amortized 2>/dev/null || true
	@echo ""
	@echo "=== Pods (amortized-jobs) ==="
	@$(KUBECTL) get pods -n amortized-jobs 2>/dev/null || true
	@echo ""
	@echo "=== Pods (amortized-dev) ==="
	@$(KUBECTL) get pods -n amortized-dev 2>/dev/null || true
	@echo ""
	@echo "=== Pods (amortized-dev-jobs) ==="
	@$(KUBECTL) get pods -n amortized-dev-jobs 2>/dev/null || true
	@echo ""
	@echo "=== Access ==="
	@echo "  Prod Studio:  http://localhost:31080"
	@echo "  Prod API:     http://localhost:31081"
	@echo "  MLflow:       http://localhost:31082"
	@echo "  Dev Studio:   http://localhost:31090"
	@echo "  Dev API:      http://localhost:31091"
	@echo ""
	@echo "  SSH tunnel:"
	@echo "    ssh -L 31080:localhost:31080 -L 31081:localhost:31081 -L 31082:localhost:31082 \\"
	@echo "        -L 31090:localhost:31090 -L 31091:localhost:31091 user@REDACTED_IP"
