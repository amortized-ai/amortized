IMAGE_TAG     ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
REPO_ROOT     := $(shell pwd)
STUDIO_DIR    ?= $(REPO_ROOT)/studio

# GHCR images
SERVER_IMAGE  ?= ghcr.io/amortized-ai/amortized:latest
STUDIO_IMAGE  ?= ghcr.io/amortized-ai/studio:latest

.PHONY: help build build-server build-studio prompt deploy-dev

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

PROMPT_DIR   := agent/prompts
COMBINED_DIR := $(PROMPT_DIR)/_combined

SKILLS_DIR   := agent/skills
K8S_SKILLS   := k8s/base/morty-skills

prompt: ## Build combined Morty prompt and sync skills to k8s
	@mkdir -p $(COMBINED_DIR)
	@cat $(PROMPT_DIR)/identity.md $(PROMPT_DIR)/workflow.md > $(COMBINED_DIR)/morty.md
	@cp $(COMBINED_DIR)/morty.md k8s/base/morty-prompt.md
	@rm -rf $(K8S_SKILLS)
	@cp -r $(SKILLS_DIR) $(K8S_SKILLS)
	@echo "Generated $(COMBINED_DIR)/morty.md and synced skills to $(K8S_SKILLS)"

# ──────────────────────────────────────────────
# Deploy (single-user dev)
# ──────────────────────────────────────────────

deploy-dev: ## Deploy single-user dev environment (requires kubectl)
	kubectl apply -k k8s/overlays/dev
