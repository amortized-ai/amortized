IMAGE_TAG     ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
REPO_ROOT     := $(shell pwd)
STUDIO_DIR    ?= $(REPO_ROOT)/studio

# GHCR images
SERVER_IMAGE  ?= ghcr.io/amortized-ai/amortized:latest
STUDIO_IMAGE  ?= ghcr.io/amortized-ai/studio:latest

.PHONY: help build build-server build-studio prompt deploy-dev lint typecheck test

# ──────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────

build: build-server build-studio ## Build all images

build-server: ## Build amortized server image
	@echo "Removing old amortized-server images from Docker..."
	@docker images --format '{{.Repository}}:{{.Tag}}' | grep '^amortized-server:' | xargs -r docker rmi 2>/dev/null || true
	@echo "Building amortized-server:$(IMAGE_TAG)..."
	docker build -t amortized-server:$(IMAGE_TAG) -f Dockerfile .

build-studio: ## Build studio image
	@echo "Removing old amortized-studio images from Docker..."
	@docker images --format '{{.Repository}}:{{.Tag}}' | grep '^amortized-studio:' | xargs -r docker rmi 2>/dev/null || true
	@echo "Building amortized-studio:$(IMAGE_TAG)..."
	docker build -t amortized-studio:$(IMAGE_TAG) -f $(STUDIO_DIR)/Dockerfile.kind $(STUDIO_DIR)

# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────

AGENTS_DIR   := agents
K8S_SKILLS   := k8s/base/morty-skills

prompt: ## Generate k8s configs from agents directory
	@cat $(AGENTS_DIR)/orchestrator/identity.md $(AGENTS_DIR)/orchestrator/workflow.md > k8s/base/morty-prompt.md
	@cp $(AGENTS_DIR)/orchestrator/identity.md k8s/base/morty-identity.md
	@cp $(AGENTS_DIR)/orchestrator/workflow.md k8s/base/morty-workflow.md
	@cp $(AGENTS_DIR)/sdg/workflow.md k8s/base/morty-sdg-workflow.md
	@cp $(AGENTS_DIR)/training/workflow.md k8s/base/morty-training-workflow.md
	@rm -rf $(K8S_SKILLS)
	@for agent in sdg training; do \
		if [ -d $(AGENTS_DIR)/$$agent/skills ]; then \
			mkdir -p $(K8S_SKILLS)/$$agent; \
			cp -r $(AGENTS_DIR)/$$agent/skills/* $(K8S_SKILLS)/$$agent/; \
		fi; \
	done
	@echo "Generated k8s configs from $(AGENTS_DIR)/"

# ──────────────────────────────────────────────
# Deploy (single-user dev)
# ──────────────────────────────────────────────

deploy-dev: prompt ## Deploy single-user dev environment (requires kubectl)
	kubectl apply -k k8s/overlays/dev

# ──────────────────────────────────────────────
# Quality
# ──────────────────────────────────────────────

lint: ## Run linter (ruff)
	ruff check src/ tests/

typecheck: ## Run type checker (mypy)
	mypy src/

test: ## Run test suite (pytest)
	pytest tests/
