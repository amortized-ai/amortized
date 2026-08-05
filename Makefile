CLUSTER_NAME  ?= amortized
IMAGE_TAG     ?= kind-$(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
KUBECTL       := kubectl --context kind-$(CLUSTER_NAME)
STUDIO_DIR    ?= $(REPO_ROOT)/studio
REPO_ROOT     := $(shell pwd)

# Developer environments
USERS := meyceoz ssudalai mathale nmalepat esivaram asaluja

# GHCR images (used as base refs in kustomize, overridden via sed for local builds)
SERVER_IMAGE  ?= ghcr.io/amortized-ai/amortized:latest
STUDIO_IMAGE  ?= ghcr.io/amortized-ai/studio:latest

# Third-party images to pre-load into kind
MINIO_IMAGE   ?= quay.io/minio/minio:latest
MLFLOW_IMAGE  ?= ghcr.io/mlflow/mlflow:latest
AWSCLI_IMAGE  ?= docker.io/amazon/aws-cli:latest
NVIDIA_DP_IMAGE ?= nvcr.io/nvidia/k8s-device-plugin:v0.19.3
TRAINING_IMAGE ?= ghcr.io/amortized-ai/training:latest
DATA_DESIGNER_IMAGE ?= ghcr.io/amortized-ai/data-designer:latest
OPENCODE_IMAGE ?= ghcr.io/anomalyco/opencode:latest
DOCLING_IMAGE  ?= ghcr.io/docling-project/docling-serve:latest
POSTGRES_IMAGE ?= docker.io/library/postgres:16-alpine

# GHCR credentials (set GHCR_USER and GHCR_TOKEN to enable private image pulls)
GHCR_USER  ?=
GHCR_TOKEN ?=

# OpenCode credentials
GOOGLE_ADC_PATH ?= $(HOME)/.config/gcloud/application_default_credentials.json
VERTEX_PROJECT  ?= itpc-gcp-ai-eng-claude
VERTEX_LOCATION ?= global

.PHONY: help up build build-server build-studio pull-images \
        load load-server load-studio load-deps \
        prompt deploy-shared deploy-all deploy-user \
        down-all down-user clean-images \
        cluster gpu ghcr-pull-secret \
        socat-setup destroy status

# ──────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Full setup
# ──────────────────────────────────────────────

up: cluster gpu pull-images load-deps build load deploy-all status ## Full setup: cluster + GPU + images + deploy all

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
	@echo "Labelling worker node for GPU scheduling..."
	@$(KUBECTL) label node $(CLUSTER_NAME)-worker nvidia.com/gpu.present=true --overwrite
	@echo "GPU allocatable:"
	@$(KUBECTL) get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'

# ──────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────

build: build-server build-studio pull-images ## Build all images

build-server: ## Build amortized server image
	@echo "Removing old amortized-server images from Docker..."
	@docker images --format '{{.Repository}}:{{.Tag}}' | grep '^amortized-server:kind-' | xargs -r docker rmi 2>/dev/null || true
	@echo "Building amortized-server:$(IMAGE_TAG)..."
	docker build -t amortized-server:$(IMAGE_TAG) -f Dockerfile .

build-studio: ## Build studio image
	@echo "Removing old amortized-studio images from Docker..."
	@docker images --format '{{.Repository}}:{{.Tag}}' | grep '^amortized-studio:kind-' | xargs -r docker rmi 2>/dev/null || true
	@docker images --format '{{.Repository}}:{{.Tag}}' | grep '^studio:kind-' | xargs -r docker rmi 2>/dev/null || true
	@echo "Building amortized-studio:$(IMAGE_TAG)..."
	docker build -t amortized-studio:$(IMAGE_TAG) -f $(STUDIO_DIR)/Dockerfile.kind $(STUDIO_DIR)

pull-images: ## Pull third-party images (MinIO, MLflow, docling-serve, training, etc.)
	@echo "Pulling third-party images..."
	@for img in $(MINIO_IMAGE) $(MLFLOW_IMAGE) $(DOCLING_IMAGE) $(AWSCLI_IMAGE) $(OPENCODE_IMAGE) $(POSTGRES_IMAGE); do \
		docker pull $$img 2>/dev/null || true; \
	done
	@echo "Pulling ML images (training image is ~12GB, this may take a while)..."
	@for img in $(TRAINING_IMAGE) $(DATA_DESIGNER_IMAGE); do \
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
	@for img in $(MINIO_IMAGE) $(MLFLOW_IMAGE) $(DOCLING_IMAGE) $(AWSCLI_IMAGE) $(OPENCODE_IMAGE) $(POSTGRES_IMAGE); do \
		kind load docker-image $$img --name $(CLUSTER_NAME) 2>/dev/null || true; \
	done
	@echo "Loading ML images into kind (training is ~12GB, be patient)..."
	@for img in $(TRAINING_IMAGE) $(DATA_DESIGNER_IMAGE); do \
		kind load docker-image $$img --name $(CLUSTER_NAME) 2>/dev/null || true; \
	done

# ──────────────────────────────────────────────
# GHCR pull secret
# ──────────────────────────────────────────────

ghcr-pull-secret: ## Create ghcr.io pull secret in all user namespaces (requires GHCR_USER and GHCR_TOKEN)
	@if [ -z "$(GHCR_USER)" ] || [ -z "$(GHCR_TOKEN)" ]; then \
		echo "Skipping GHCR pull secret: set GHCR_USER and GHCR_TOKEN to enable."; \
	else \
		for user in $(USERS); do \
			for ns in amortized-$$user amortized-$$user-jobs; do \
				if $(KUBECTL) -n $$ns get secret ghcr-pull >/dev/null 2>&1; then \
					echo "  ghcr-pull already exists in $$ns, skipping."; \
				else \
					echo "  Creating ghcr-pull in $$ns..."; \
					$(KUBECTL) create secret docker-registry ghcr-pull \
						--docker-server=ghcr.io \
						--docker-username=$(GHCR_USER) \
						--docker-password=$(GHCR_TOKEN) \
						-n $$ns; \
				fi; \
				$(KUBECTL) -n $$ns patch serviceaccount default \
					-p '{"imagePullSecrets": [{"name": "ghcr-pull"}]}' 2>/dev/null || true; \
			done; \
		done; \
		echo "GHCR pull secret configured."; \
	fi

# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────

PROMPT_DIR   := agent/prompts
COMBINED_DIR := $(PROMPT_DIR)/_combined

SKILLS_DIR   := agent/skills
K8S_SKILLS   := k8s/base/morty-skills

prompt: ## Build combined Morty prompt and sync skills to k8s
	@mkdir -p $(COMBINED_DIR)
	@cat $(PROMPT_DIR)/identity.md $(PROMPT_DIR)/capabilities.md $(PROMPT_DIR)/workflow.md > $(COMBINED_DIR)/morty.md
	@cp $(COMBINED_DIR)/morty.md k8s/base/morty-prompt.md
	@rm -rf $(K8S_SKILLS)
	@cp -r $(SKILLS_DIR) $(K8S_SKILLS)
	@echo "Generated $(COMBINED_DIR)/morty.md and synced skills to $(K8S_SKILLS)"

# ──────────────────────────────────────────────
# Deploy shared services (MLflow, MinIO)
# ──────────────────────────────────────────────

deploy-shared: ## Deploy shared services (MLflow, MinIO, PostgreSQL) into amortized namespace
	@echo "Deploying shared services..."
	$(KUBECTL) apply -k k8s/overlays/shared
	@echo "Waiting for MinIO to be ready..."
	@$(KUBECTL) -n amortized rollout status deployment/minio --timeout=120s
	@$(KUBECTL) run minio-init --rm -i --restart=Never \
		--image=$(AWSCLI_IMAGE) \
		--namespace=amortized \
		--overrides='{"spec":{"containers":[{"name":"minio-init","image":"$(AWSCLI_IMAGE)","command":["sh","-c","aws --endpoint-url http://minio.amortized.svc.cluster.local:9000 s3 mb s3://amortized 2>/dev/null || true"],"env":[{"name":"AWS_ACCESS_KEY_ID","value":"minioadmin"},{"name":"AWS_SECRET_ACCESS_KEY","value":"minioadmin"}]}]}}' \
		|| echo "  Warning: could not create MinIO bucket (may already exist)."
	@echo "Waiting for MLflow..."
	@$(KUBECTL) -n amortized rollout status deployment/mlflow --timeout=120s
	@echo "Waiting for PostgreSQL..."
	@$(KUBECTL) -n amortized rollout status deployment/postgres --timeout=120s
	@echo "Shared services deployed."

# ──────────────────────────────────────────────
# Deploy per-user environment
# ──────────────────────────────────────────────

deploy-user: prompt ## Deploy a user's environment (USER=<name>)
	@if [ -z "$(USER)" ]; then echo "Usage: make deploy-user USER=<username>"; exit 1; fi
	@echo "Deploying environment for $(USER)..."
	@# Create namespaces
	$(KUBECTL) apply -f k8s/overlays/users/$(USER)/namespace.yaml
	@# GHCR pull secret
	@if [ -n "$(GHCR_USER)" ] && [ -n "$(GHCR_TOKEN)" ]; then \
		for ns in amortized-$(USER) amortized-$(USER)-jobs; do \
			if $(KUBECTL) -n $$ns get secret ghcr-pull >/dev/null 2>&1; then \
				echo "  ghcr-pull already exists in $$ns, skipping."; \
			else \
				echo "  Creating ghcr-pull in $$ns..."; \
				$(KUBECTL) create secret docker-registry ghcr-pull \
					--docker-server=ghcr.io \
					--docker-username=$(GHCR_USER) \
					--docker-password=$(GHCR_TOKEN) \
					-n $$ns; \
			fi; \
			$(KUBECTL) -n $$ns patch serviceaccount default \
				-p '{"imagePullSecrets": [{"name": "ghcr-pull"}]}' 2>/dev/null || true; \
		done; \
	fi
	@# Copy OpenCode credentials from shared namespace
	@for secret in opencode-gcp opencode-llm; do \
		if $(KUBECTL) -n amortized-$(USER) get secret $$secret >/dev/null 2>&1; then \
			echo "  Secret $$secret already exists in amortized-$(USER), skipping."; \
		elif $(KUBECTL) -n amortized get secret $$secret >/dev/null 2>&1; then \
			echo "  Copying $$secret from amortized to amortized-$(USER)..."; \
			$(KUBECTL) -n amortized get secret $$secret -o json | \
				jq '.metadata.namespace = "amortized-$(USER)" | del(.metadata.resourceVersion,.metadata.uid,.metadata.creationTimestamp,.metadata.annotations)' | \
				$(KUBECTL) apply -f - || \
				echo "  Warning: could not copy $$secret to amortized-$(USER)."; \
		else \
			echo "  Warning: $$secret not found in amortized namespace."; \
		fi; \
	done
	@# Apply kustomize overlay (sed replaces GHCR refs with local image tags)
	@$(KUBECTL) kustomize k8s/overlays/users/$(USER) | \
		sed \
			-e 's|image: ghcr.io/amortized-ai/amortized:latest|image: amortized-server:$(IMAGE_TAG)|g' \
			-e 's|image: ghcr.io/amortized-ai/studio:latest|image: amortized-studio:$(IMAGE_TAG)|g' \
		| $(KUBECTL) apply -f -
	@# Apply resources that live outside kustomize (pre-namespaced)
	$(KUBECTL) apply -f k8s/overlays/users/$(USER)/s3-secrets.yaml
	$(KUBECTL) apply -f k8s/overlays/users/$(USER)/nodeport-services.yaml
	$(KUBECTL) apply -f k8s/overlays/users/$(USER)/gpu-quota.yaml
	$(KUBECTL) apply -f k8s/overlays/users/$(USER)/rbac-jobs.yaml
	@# Restart all deployments so pods pick up new configs / images
	@$(KUBECTL) -n amortized-$(USER) rollout restart deployment/amortized-server deployment/amortized-studio deployment/opencode deployment/claude-code
	@# Wait for rollouts
	@echo "Waiting for $(USER) deployments..."
	@$(KUBECTL) -n amortized-$(USER) rollout status deployment/amortized-server --timeout=120s
	@$(KUBECTL) -n amortized-$(USER) rollout status deployment/amortized-studio --timeout=120s
	@$(KUBECTL) -n amortized-$(USER) rollout status deployment/opencode --timeout=120s
	@$(KUBECTL) -n amortized-$(USER) rollout status deployment/claude-code --timeout=120s
	@echo "$(USER) environment deployed."

# ──────────────────────────────────────────────
# Deploy all / tear down
# ──────────────────────────────────────────────

deploy-all: deploy-shared ## Deploy shared services + all user environments
	@for user in $(USERS); do \
		echo ""; \
		echo ">>> Deploying $$user"; \
		$(MAKE) deploy-user USER=$$user; \
	done
	@echo ""
	@echo "All environments deployed."

down-user: ## Tear down a user's environment (USER=<name>)
	@if [ -z "$(USER)" ]; then echo "Usage: make down-user USER=<username>"; exit 1; fi
	@echo "Tearing down $(USER) environment..."
	$(KUBECTL) delete namespace amortized-$(USER) amortized-$(USER)-jobs --ignore-not-found
	@echo "$(USER) environment removed."

down-all: ## Tear down all user environments (keeps shared services)
	@for user in $(USERS); do \
		$(MAKE) down-user USER=$$user; \
	done
	@echo "All user environments removed. Shared services untouched."

# ──────────────────────────────────────────────
# Image cleanup
# ──────────────────────────────────────────────

clean-images: ## Remove old amortized images from kind nodes and reclaim disk
	@echo "Cleaning old images from kind nodes..."
	@for node in $(CLUSTER_NAME)-control-plane $(CLUSTER_NAME)-worker; do \
		echo "  Cleaning $$node..."; \
		IN_USE=$$(docker exec $$node crictl ps 2>/dev/null | awk 'NR>1 {print $$2}' | sort -u); \
		ALL=$$(docker exec $$node crictl images 2>/dev/null | \
			grep -E 'amortized-server|amortized-studio|library/studio' | \
			awk '{print $$3, $$1":"$$2}'); \
		echo "$$ALL" | while read id ref; do \
			KEEP=0; \
			for used in $$IN_USE; do \
				case "$$used" in "$$id"*) KEEP=1; break;; esac; \
			done; \
			if [ "$$KEEP" = "0" ] && [ -n "$$ref" ]; then \
				docker exec $$node ctr -n k8s.io images rm "$$ref" 2>/dev/null || true; \
			fi; \
		done; \
		docker exec $$node sh -c "ctr -n k8s.io content prune references" 2>/dev/null || true; \
	done
	@echo "Pruning Docker build cache..."
	@docker builder prune -f --filter 'until=24h' 2>/dev/null || true
	@echo "Image cleanup complete."

# ──────────────────────────────────────────────
# Refresh (rebuild + redeploy)
# ──────────────────────────────────────────────

refresh-user: build-server build-studio clean-images load-server load-studio ## Rebuild images and redeploy a user (USER=<name>)
	@if [ -z "$(USER)" ]; then echo "Usage: make refresh-user USER=<username>"; exit 1; fi
	$(MAKE) deploy-user USER=$(USER)
	@echo "$(USER) refreshed."

# ──────────────────────────────────────────────
# Per-user convenience aliases
# ──────────────────────────────────────────────

define user_targets
.PHONY: deploy-$(1) down-$(1) refresh-$(1)
deploy-$(1): ## Deploy $(1)'s environment
	$$(MAKE) deploy-user USER=$(1)
down-$(1): ## Tear down $(1)'s environment
	$$(MAKE) down-user USER=$(1)
refresh-$(1): ## Rebuild + redeploy $(1)
	$$(MAKE) refresh-user USER=$(1)
endef

$(foreach u,$(USERS),$(eval $(call user_targets,$(u))))

# ──────────────────────────────────────────────
# Socat port bridging
# ──────────────────────────────────────────────

socat-setup: ## Set up socat bridges for ports not in kind config (run on the kind host)
	@CONTROL_PLANE_IP=$$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(CLUSTER_NAME)-control-plane 2>/dev/null); \
	if [ -z "$$CONTROL_PLANE_IP" ]; then \
		echo "Could not find control-plane container. Is kind running?"; \
		exit 1; \
	fi; \
	echo "Control plane IP: $$CONTROL_PLANE_IP"; \
	echo "Starting socat bridges..."; \
	for PORT in 31100 31101 31110 31111 31120 31121 31130 31131 31140 31141; do \
		if ss -tlnp | grep -q ":$$PORT "; then \
			echo "  Port $$PORT already listening, skipping."; \
		else \
			echo "  Bridging host:$$PORT -> $$CONTROL_PLANE_IP:$$PORT"; \
			nohup socat TCP-LISTEN:$$PORT,fork,reuseaddr TCP:$$CONTROL_PLANE_IP:$$PORT > /dev/null 2>&1 & \
		fi; \
	done; \
	echo "Done. Verify with: ss -tlnp | grep 311"

# ──────────────────────────────────────────────
# Teardown
# ──────────────────────────────────────────────

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
	@echo "=== Shared Services (amortized) ==="
	@$(KUBECTL) get pods -n amortized 2>/dev/null || true
	@for user in $(USERS); do \
		echo ""; \
		echo "=== $$user (amortized-$$user) ==="; \
		$(KUBECTL) get pods -n amortized-$$user 2>/dev/null || echo "  Namespace not found"; \
		echo "  Jobs:"; \
		$(KUBECTL) get pods -n amortized-$$user-jobs 2>/dev/null || echo "  (none)"; \
	done
	@echo ""
	@echo "=== Access ==="
	@echo "  MLflow:           http://localhost:31082"
	@echo ""
	@echo "  meyceoz Studio:   http://localhost:31100"
	@echo "  meyceoz API:      http://localhost:31101"
	@echo "  ssudalai Studio:  http://localhost:31110"
	@echo "  ssudalai API:     http://localhost:31111"
	@echo "  mathale Studio:   http://localhost:31120"
	@echo "  mathale API:      http://localhost:31121"
	@echo "  nmalepat Studio:  http://localhost:31130"
	@echo "  nmalepat API:     http://localhost:31131"
	@echo "  esivaram Studio:  http://localhost:31140"
	@echo "  esivaram API:     http://localhost:31141"
	@echo "  asaluja Studio:   http://localhost:31150"
	@echo "  asaluja API:      http://localhost:31151"
	@echo ""
	@echo "  SSH tunnel:"
	@echo "    ssh -L 31082:localhost:31082 \\"
	@echo "        -L 31100:localhost:31100 -L 31101:localhost:31101 \\"
	@echo "        -L 31110:localhost:31110 -L 31111:localhost:31111 \\"
	@echo "        -L 31120:localhost:31120 -L 31121:localhost:31121 \\"
	@echo "        -L 31130:localhost:31130 -L 31131:localhost:31131 \\"
	@echo "        -L 31140:localhost:31140 -L 31141:localhost:31141 \\"
	@echo "        -L 31150:localhost:31150 -L 31151:localhost:31151 \\"
	@echo "        user@<gpu-host>"
