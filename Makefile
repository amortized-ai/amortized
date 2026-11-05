IMAGE_TAG     ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
REPO_ROOT     := $(shell pwd)
STUDIO_DIR    ?= $(REPO_ROOT)/studio

# GHCR images
SERVER_IMAGE  ?= ghcr.io/amortized-ai/amortized:latest
STUDIO_IMAGE  ?= ghcr.io/amortized-ai/studio:latest

# Eval Hub benchmark catalog (vendored) and the adapter image it is checked against
EVALHUB_CATALOG_COMMIT ?= 9a18b91657dae635d9ed56a2075ec1b808eb6345
EVALHUB_DIR            := deploy/helm/amortized/files/evalhub
LIGHTEVAL_IMAGE        ?= $(shell sed -n 's/^ *lightEvalImage: *//p' deploy/helm/amortized/values.yaml)
LM_EVAL_IMAGE          ?= $(shell sed -n 's/^ *lmEvalImage: *//p' deploy/helm/amortized/values.yaml)

.PHONY: help build build-server build-studio prompt deploy-dev lint typecheck test evalhub-catalog evalhub-compat evalhub-compat-lighteval evalhub-compat-lm-eval

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

AGENTS_DIR     := agents
K8S_SKILLS     := k8s/base/morty-skills
HELM_FILES     := deploy/helm/amortized/files

# Build the native opencode skill tree at $(1). Every agents/*/skills/**/SKILL.md
# becomes a skill dir named after its frontmatter `name:`, carrying its sibling
# files (config template, model list, ...) EXCEPT reference-payload.json, which the
# server recipe loader (core/recipes.py) reads from agents/ directly. Adding a
# skill is just dropping a SKILL.md with a `name:` — no edit here.
define gen_native_skills
rm -rf $(1); \
find $(AGENTS_DIR) -path '*/skills/*/SKILL.md' | while read skill; do \
	d=$$(dirname "$$skill"); \
	n=$$(sed -n 's/^name:[[:space:]]*//p' "$$skill" | head -1); \
	if [ -z "$$n" ]; then echo "ERROR: no frontmatter name in $$skill" >&2; exit 1; fi; \
	mkdir -p "$(1)/$$n"; \
	find "$$d" -maxdepth 1 -type f ! -name reference-payload.json -exec cp {} "$(1)/$$n/" \; ; \
done
endef

prompt: ## Generate k8s configs from agents directory
	@cat $(AGENTS_DIR)/orchestrator/identity.md $(AGENTS_DIR)/orchestrator/workflow.md > k8s/base/morty-prompt.md
	@cp $(AGENTS_DIR)/orchestrator/identity.md k8s/base/morty-identity.md
	@cp $(AGENTS_DIR)/orchestrator/workflow.md k8s/base/morty-workflow.md
	@cp $(AGENTS_DIR)/sdg/workflow.md k8s/base/morty-sdg-workflow.md
	@cp $(AGENTS_DIR)/training/workflow.md k8s/base/morty-training-workflow.md
	@cp $(AGENTS_DIR)/eval/workflow.md k8s/base/morty-eval-workflow.md
	@# Native opencode skills: expanded in-pod to .opencode/skills/<skill>/SKILL.md.
	@$(call gen_native_skills,$(K8S_SKILLS))
	@# Helm chart carries the same persona + skills. A Helm package is self-contained
	@# (it can't read files outside the chart dir), so the chart needs its own copy of
	@# agents/ (the single source of truth). These are generated artifacts (gitignored),
	@# never hand-edited; run `make prompt` before `helm package`/install-from-checkout.
	@rm -rf $(HELM_FILES)/morty-config $(HELM_FILES)/morty-skills
	@mkdir -p $(HELM_FILES)/morty-config
	@cat $(AGENTS_DIR)/orchestrator/identity.md $(AGENTS_DIR)/orchestrator/workflow.md > $(HELM_FILES)/morty-config/morty.md
	@cp $(AGENTS_DIR)/orchestrator/identity.md $(HELM_FILES)/morty-config/morty-identity.md
	@cp $(AGENTS_DIR)/orchestrator/workflow.md $(HELM_FILES)/morty-config/morty-workflow.md
	@cp $(AGENTS_DIR)/sdg/workflow.md $(HELM_FILES)/morty-config/morty-sdg-workflow.md
	@cp $(AGENTS_DIR)/training/workflow.md $(HELM_FILES)/morty-config/morty-training-workflow.md
	@cp $(AGENTS_DIR)/eval/workflow.md $(HELM_FILES)/morty-config/morty-eval-workflow.md
	@$(call gen_native_skills,$(HELM_FILES)/morty-skills)
	@echo "Generated k8s + Helm configs from $(AGENTS_DIR)/"

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

evalhub-catalog: ## Vendor Eval Hub's LightEval and lm-eval catalogs at EVALHUB_CATALOG_COMMIT (re-run evalhub-compat after)
	for name in lighteval lm_evaluation_harness; do \
	  { printf '# Verbatim copy of https://github.com/eval-hub/eval-hub/blob/%s/config/providers/%s.yaml\n' $(EVALHUB_CATALOG_COMMIT) $$name; \
	    printf '# Refresh with `make evalhub-catalog`; the Helm template overrides only id, runtime and resources.\n'; \
	    curl -fsSL https://raw.githubusercontent.com/eval-hub/eval-hub/$(EVALHUB_CATALOG_COMMIT)/config/providers/$$name.yaml; \
	  } > $(EVALHUB_DIR)/$$name.yaml.tmp && mv $(EVALHUB_DIR)/$$name.yaml.tmp $(EVALHUB_DIR)/$$name.yaml || exit 1; \
	done

# Needs network: LightEval's task registry downloads files when it loads, and both
# helpers resolve Hugging Face dataset names to their canonical repo.
evalhub-compat: evalhub-compat-lighteval evalhub-compat-lm-eval ## Record what each benchmark needs, read from inside each pinned adapter image

evalhub-compat-lighteval:
	docker run --rm --entrypoint python -e HF_HOME=/tmp/hf \
	  -v $(REPO_ROOT)/scripts/evalhub:/helper:ro -v $(REPO_ROOT)/$(EVALHUB_DIR):/catalog:ro \
	  $(LIGHTEVAL_IMAGE) /helper/lighteval_compat.py --catalog /catalog/lighteval.yaml \
	  --catalog-commit $(EVALHUB_CATALOG_COMMIT) --adapter-image $(LIGHTEVAL_IMAGE) \
	  > $(EVALHUB_DIR)/compat/amortized-lighteval.json.tmp
	mv $(EVALHUB_DIR)/compat/amortized-lighteval.json.tmp $(EVALHUB_DIR)/compat/amortized-lighteval.json

evalhub-compat-lm-eval:
	docker run --rm --entrypoint /opt/app-root/bin/python -e HF_HOME=/tmp/hf \
	  -v $(REPO_ROOT)/scripts/evalhub:/helper:ro -v $(REPO_ROOT)/$(EVALHUB_DIR):/catalog:ro \
	  $(LM_EVAL_IMAGE) /helper/lm_eval_compat.py --catalog /catalog/lm_evaluation_harness.yaml \
	  --catalog-commit $(EVALHUB_CATALOG_COMMIT) --adapter-image $(LM_EVAL_IMAGE) \
	  > $(EVALHUB_DIR)/compat/amortized-lm-eval.json.tmp
	mv $(EVALHUB_DIR)/compat/amortized-lm-eval.json.tmp $(EVALHUB_DIR)/compat/amortized-lm-eval.json
