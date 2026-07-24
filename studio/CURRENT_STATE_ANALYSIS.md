# Amortized Studio - Current State Analysis

**Date**: 2026-07-07  
**Branch**: `main`  
**Purpose**: Understand what's been completed and what's next

---

## 🎯 Project Goal

**From SPEC.md**:
> Studio is the web UI that makes the amortized runtime accessible to teams who want to build task models without living in the terminal.

**Key Principle**: Chat-first, dashboard-second. The agent chat is the primary entry point.

---

## ✅ What's Been Completed

### 1. **Overview Page** (`src/features/overview/`)
- ✅ Dashboard with section cards (Jobs, Datasets, Models count)
- ✅ Recent jobs list
- ✅ Empty state with call-to-action
- **Purpose**: Landing page showing system at-a-glance

### 2. **Chat Page** (`src/features/chat/`)
- ✅ OpenCode agent integration (Morty)
- ✅ Conversation list with create/delete
- ✅ Message list with streaming
- ✅ Chat input
- ✅ Plan progress indicator (placeholder)
- ✅ Action confirmation (confirm/reject buttons)
- **Purpose**: PRIMARY WORKFLOW - Natural language task model building

**Key Implementation Details**:
- Uses `/agent/session` endpoint (OpenCode protocol)
- SSE streaming for responses
- Supports action proposals with confirm/reject
- Chat state management via Zustand store

### 3. **Jobs Page** (`src/features/jobs/`)
- ✅ Job list with filtering (type, status)
- ✅ Job detail panel (slide-out)
- ✅ Real-time updates (WebSocket)
- ✅ Job types: training, sdg, eval
- ✅ Job status badges
- ✅ Empty state
- **Purpose**: Monitor all pipeline jobs

**Job Types Supported**:
- `training` - Model training jobs
- `sdg` - Synthetic data generation
- `eval` - Evaluation jobs

### 4. **Datasets Page** (`src/features/datasets/`)
- ✅ Dataset list from MLflow
- ✅ Dataset table with metadata
- ✅ Dataset detail page (separate route)
- ✅ Empty state
- **Purpose**: Browse training datasets

**Data Source**: MLflow experiments (filtered by tags)

### 5. **Models Page** (`src/features/models/`)
- ✅ Model list from MLflow Model Registry
- ✅ Model table with base model, version info
- ✅ Model detail page (separate route)
- ✅ Empty state
- **Purpose**: Browse trained models

**Data Source**: MLflow Model Registry

### 6. **Recipes Page** (`src/features/recipes/`)
- ✅ Recipe list (training configurations)
- ✅ Recipe builder form with sections:
  - Training Method (LoRA SFT, SFT, DPO, GRPO, etc.)
  - Model Selection
  - Data Selection
  - Training Settings
- ✅ JSON editor (split-pane)
- ✅ Execute dialog (submits job)
- ✅ Save/Save As functionality
- **Purpose**: Configure and submit training jobs

**This is the second most important feature** (after Chat) - alternative workflow for power users.

### 7. **Settings Page** (`src/features/settings/`)
- ✅ MLflow Gateway routes management
- ✅ Prerequisites card (health check)
- ✅ Create/delete gateway routes
- **Purpose**: System configuration

---

## 📊 Implementation Status vs. SPEC

### Module 1: Chat Agent Interface ✅ MOSTLY COMPLETE
- ✅ Streaming responses (SSE)
- ✅ Conversation history
- ✅ Action proposals (confirm/reject)
- ⚠️ **Missing**: Structured option cards (bullet points as clickable cards)
- ⚠️ **Missing**: Tool result badges (collapsible indicators)
- ⚠️ **Missing**: Plan progress indicator (has placeholder but not functional)
- ⚠️ **Missing**: Contextual chat panel (side panel on other pages)

### Module 2: Job Dashboard ✅ COMPLETE
- ✅ Job list with filters
- ✅ Real-time updates (WebSocket)
- ✅ Job detail panel
- ✅ Job actions (cancel supported via API)
- ⚠️ **Missing**: Training metrics charts (loss curves, gradient norm)

### Module 3: Dataset Manager ✅ BASIC COMPLETE
- ✅ Dataset list
- ✅ Dataset detail page
- ⚠️ **Missing**: Sample row preview
- ⚠️ **Missing**: Quality tests (token limits, turn validation)
- ⚠️ **Missing**: File upload (drag-and-drop)
- ⚠️ **Missing**: Dataset versioning
- ⚠️ **Missing**: Dataset lineage (which jobs used it)

### Module 4: Evaluator Registry ❌ NOT IMPLEMENTED
- ❌ No evaluator management UI
- ❌ No evaluator CRUD
- **Note**: Backend may have evaluators, but no UI for them

### Module 5: Evaluation Runner & Results ❌ NOT IMPLEMENTED
- ❌ No evaluation runner UI
- ❌ No eval results comparison
- ❌ No model comparison view

### Module 6: Model Registry ✅ BASIC COMPLETE
- ✅ Model list
- ✅ Model detail page
- ⚠️ **Missing**: Training metrics charts
- ⚠️ **Missing**: Model lineage (base model → dataset → recipe)
- ⚠️ **Missing**: Model actions (deploy, export, evaluate)

### Module 7: Recipe Builder ✅ COMPLETE
- ✅ Recipe list
- ✅ Builder form
- ✅ JSON editor (split-pane)
- ✅ Execute
- ✅ Save/Save As
- **This is fully functional!**

### Module 8: Compute & Settings ⚠️ PARTIAL
- ✅ Gateway routes management
- ✅ Health check
- ❌ No compute backends UI
- ❌ No API key management
- ❌ No GPU info display
- ❌ No VRAM estimator

---

## 🏗️ Architecture Decisions (Confirmed)

### Tech Stack
- **Frontend**: React 19 + TypeScript
- **Router**: React Router v7
- **Data Fetching**: TanStack Query (React Query)
- **UI Components**: Radix UI (shadcn/ui)
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **State Management**: Zustand (for chat, UI)
- **Forms**: React Hook Form
- **Build Tool**: Vite

### API Integration
- **REST API**: `/api/v1/*` endpoints
- **WebSocket**: `/api/v1/ws` for real-time job updates
- **SSE**: `/agent/session` for chat streaming
- **MLflow**: Direct queries to MLflow API for datasets/models

### Design Patterns
**Confirmed from codebase**:
1. **Feature-based structure**: Each feature is self-contained in `src/features/<name>/`
2. **API hooks pattern**: `api/use-*.ts` files with React Query
3. **Component composition**: Reusable UI components in `src/components/ui/`
4. **Type safety**: Strict TypeScript with `@/types/api.ts`

---

## 🔍 Key Findings

### 1. **MLflow Integration is Central**
- Datasets come from MLflow experiments (tag filtering)
- Models come from MLflow Model Registry
- Training metrics available via MLflow API
- **This means**: The backend stores everything in MLflow

### 2. **Chat is Implemented but Underutilized**
- Full chat infrastructure exists
- But missing the "structured interactions" from SPEC:
  - Option cards
  - Tool result indicators
  - Progress tracking
- **Opportunity**: Enhance chat to match SPEC vision

### 3. **Recipe Builder is Feature-Complete**
- This is the most polished feature
- Full form builder with all training methods
- JSON editor for power users
- Direct job submission
- **This works well!**

### 4. **Missing: Evaluation Module**
- No UI for managing evaluators
- No UI for running evaluations
- No model comparison
- **This is a gap** - SPEC has 5 user stories about evaluation

### 5. **Dataset/Model Pages are Basic**
- List views work
- Detail pages exist
- But missing:
  - Quality tests
  - File upload
  - Lineage
  - Actions (deploy, export)

---

## 🎯 Recommended Next Steps

### Priority 1: Enhance Chat (High Impact, Aligns with SPEC)
**Why**: Chat is supposed to be the PRIMARY workflow, but it's underutilized

**Tasks**:
1. ✅ **Add structured option cards**
   - When agent presents choices, render as clickable cards
   - Reduces cognitive load (SPEC design principle)

2. ✅ **Add tool result indicators**
   - Show "Tool used: MLflow" badges when agent calls backend
   - Collapsible to see details
   - Builds trust through transparency (SPEC principle)

3. ✅ **Fix plan progress indicator**
   - Show "Step 2/6: Generate data" during agent workflow
   - Currently has placeholder but not functional

4. ✅ **Add contextual chat panel**
   - Side panel version of chat available on Dataset/Model pages
   - "Ask about this dataset"

**Impact**: Makes chat the primary workflow as intended

### Priority 2: Build Evaluation Module (Fill Major Gap)
**Why**: This is completely missing but has 5 user stories in SPEC

**Tasks**:
1. ⚠️ **Create Evaluator Registry page**
   - List evaluators (LLM judges)
   - CRUD operations
   - Show prompt templates

2. ⚠️ **Create Evaluation Runner**
   - Select evaluator + dataset + model
   - Submit eval job
   - View results

3. ⚠️ **Create Model Comparison view**
   - Side-by-side eval results
   - Highlight where fine-tuned beats baseline
   - This is KEY for showing value

**Impact**: Completes the full pipeline (data → train → eval → deploy)

### Priority 3: Polish Dataset/Model Pages (Incremental Improvements)
**Why**: These work but are missing quality-of-life features

**Tasks**:
1. ⚠️ **Add dataset file upload**
   - Drag-and-drop for JSONL/CSV
   - Preview before upload

2. ⚠️ **Add dataset quality tests**
   - Client-side validation
   - Show badge on dataset card

3. ⚠️ **Add training metrics charts**
   - Loss curve on model detail page
   - Fetch from MLflow metrics API

4. ⚠️ **Add model actions**
   - Deploy button (submit vLLM serve job)
   - Export button (download adapter)
   - Evaluate button (link to eval runner)

**Impact**: Better UX, more useful pages

### Priority 4: Settings & Compute (Nice-to-Have)
**Why**: Less critical than core pipeline features

**Tasks**:
1. ⚠️ Add compute backends UI
2. ⚠️ Add API key management
3. ⚠️ Add GPU info display
4. ⚠️ Add VRAM estimator

---

## 💡 Strategic Recommendations

### 1. **Focus on Chat-First Philosophy**
The SPEC says "chat-first, dashboard-second" but currently:
- Recipe builder is more polished than chat
- Chat feels like an add-on, not the primary workflow

**Recommendation**: Invest in chat enhancements (option cards, tool indicators, progress) to make it the preferred workflow.

### 2. **Complete the Pipeline**
Current state: Data → Train ✅, but Eval ❌

**Recommendation**: Build evaluation module to close the loop. This shows ROI of fine-tuning.

### 3. **Don't Build Agent Builder**
Our stashed "Agent Builder" work duplicates what Chat should do.

**Recommendation**: 
- **Delete** agent builder files
- **Enhance** chat page to guide users through agent/task model creation
- Chat already connects to Morty agent - let it do the work!

### 4. **Leverage MLflow Fully**
Everything is in MLflow - use it!

**Recommendation**: 
- Add metrics charts (data is there)
- Show experiment lineage (MLflow tracks it)
- Link jobs ↔ datasets ↔ models via MLflow metadata

---

## 🚫 What NOT to Do

### ❌ Don't Build Separate "Agents" Feature
- Chat page already talks to Morty agent
- Agent creation should happen via chat conversation
- Our stashed "agent builder" duplicates this

### ❌ Don't Rebuild Recipe Builder
- It's already excellent
- Focus elsewhere

### ❌ Don't Add New Tech Stack
- Radix UI is working well
- Don't introduce PatternFly or other component libraries
- Stick with existing patterns

---

## 📝 Summary

### Completed Features
1. ✅ Overview
2. ✅ Chat (basic)
3. ✅ Jobs (complete)
4. ✅ Datasets (basic)
5. ✅ Models (basic)
6. ✅ Recipes (complete)
7. ✅ Settings (partial)

### Missing from SPEC
1. ❌ Evaluator Registry
2. ❌ Evaluation Runner
3. ❌ Model Comparison
4. ⚠️ Chat enhancements (option cards, tool indicators, progress)
5. ⚠️ Dataset quality tests
6. ⚠️ File upload
7. ⚠️ Training metrics charts
8. ⚠️ Model actions (deploy, export)

### Recommended Priority
**High Impact**:
1. Chat enhancements (option cards, tool indicators)
2. Evaluation module (evaluators + runner + comparison)

**Medium Impact**:
3. Dataset polish (upload, quality tests)
4. Model polish (charts, actions)

**Low Impact**:
5. Settings/compute UI

### What to Do with Agent Builder Work
**Delete it.** Chat page should handle agent creation via conversation with Morty.

---

**Next Action**: Should I start with chat enhancements or evaluation module?
