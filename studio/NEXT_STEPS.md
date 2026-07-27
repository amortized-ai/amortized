# What to Do Next

**Date**: 2026-07-07  
**Current State**: Studio frontend working, backend not connected

---

## Immediate Options

### Option 1: Find/Connect Backend (Recommended for Full Testing)

**Goal**: Get Chat and all features working

**Steps**:
1. **Locate the amortized backend repository**
   ```bash
   # Check if it exists nearby
   ls ~/studio/
   ls ~/amortized/
   # Or search GitHub
   ```

2. **Start the backend services** (typical setup):
   ```bash
   # Terminal 1: Amortized API
   cd /path/to/amortized
   uvicorn amortized.api.server:app --reload --port 8000
   
   # Terminal 2: MLflow  
   mlflow server --port 5000
   
   # Terminal 3: OpenCode Agent
   # (command depends on backend setup)
   # Usually something like: python -m amortized.agent --port 4096
   ```

3. **Verify it's working**:
   ```bash
   curl http://localhost:8000/api/v1/health
   curl http://localhost:5000  # MLflow UI
   curl -X POST http://localhost:4096/session -d '{}'
   ```

4. **Refresh Studio** → Chat should work!

**Ask someone on your team**: "Where's the amortized backend code?"

---

### Option 2: Continue Frontend Development (Works Now)

**Goal**: Improve the Studio UI without needing backend

**What Works WITHOUT Backend**:
- ✅ UI layout and navigation
- ✅ Overview page (now with instructions!)
- ✅ Component design and styling
- ✅ Page layouts
- ⚠️ Jobs/Datasets/Models pages (show empty state)
- ⚠️ Recipe builder (UI works, execution won't)

**What You CAN Work On**:

1. **Polish Existing Pages** ✨
   - Improve empty states
   - Better loading skeletons
   - Responsive design
   - Dark mode tweaks

2. **Add Missing UI Features** 🎨
   - Evaluation module (from CURRENT_STATE_ANALYSIS.md)
   - Dataset file upload UI
   - Model detail charts (mock data)
   - Training metrics visualization

3. **Enhance Chat UI** 💬
   - Add option cards design (no backend needed)
   - Tool result badges UI
   - Plan progress indicator
   - Message styling improvements

4. **Documentation** 📝
   - User guide
   - Feature documentation
   - Component storybook

**You can build/design everything, just can't test the API integration.**

---

### Option 3: Create Mock Backend (Quick Testing)

**Goal**: Test frontend features without full backend

**Create**: `/Users/nmalepat/studio/mock-server.js`

```javascript
import express from 'express'
import cors from 'cors'

const app = express()
app.use(cors())
app.use(express.json())

// Mock chat
let sessions = {}
app.post('/session', (req, res) => {
  const id = `session-${Date.now()}`
  sessions[id] = []
  res.json({ id })
})

app.post('/session/:id/message', (req, res) => {
  const text = req.body.parts?.[0]?.text || ''
  res.json({
    parts: [{
      type: 'text',
      text: `**Mock Response**\n\nYou said: "${text}"\n\nThis is a mock backend. For real AI responses, connect to the amortized backend.\n\nI can help you:\n- Generate synthetic data\n- Train task models\n- Evaluate models\n\nJust ask!`
    }],
    info: { providerID: 'mock', modelID: 'mock-gpt' }
  })
})

// Mock jobs
const mockJobs = [
  { id: 'job-1', type: 'sdg', status: 'succeeded', created_at: new Date().toISOString() },
  { id: 'job-2', type: 'training', status: 'running', created_at: new Date().toISOString() },
]
app.get('/api/v1/jobs', (req, res) => res.json(mockJobs))

// Mock health
app.get('/api/v1/health', (req, res) => {
  res.json({ status: 'ok', version: '3.0.0-mock' })
})

app.listen(4096, () => console.log('Mock backend on :4096'))
```

**Run**:
```bash
cd /Users/nmalepat/studio
node mock-server.js
# Keep running, refresh Studio
```

**Then Chat will work** (with mock responses)!

---

## Recommended Path

### For Now (This Week):

**✅ Option 2: Continue Frontend Development**

**Why**:
- You can make progress on UI/UX
- Lots of polish work to do (see CURRENT_STATE_ANALYSIS.md)
- Don't need backend for design work

**Focus Areas** (from our analysis):
1. **Evaluation Module** - Build the UI (big gap)
2. **Chat Enhancements** - Design option cards, tool badges
3. **Dataset Polish** - File upload UI, quality tests UI
4. **Model Detail** - Metrics charts (use mock data)

### Later (When Backend Available):

**✅ Option 1: Connect Real Backend**

**Why**:
- Test full integration
- Verify API contracts
- Real data flows
- End-to-end testing

---

## Current Working Status

**What's Committed** (on `main` branch):
- ✅ Overview with "How to Use" section
- ✅ Backend requirement notice (amber box)
- ✅ CURRENT_STATE_ANALYSIS.md (what's done, what's next)
- ✅ BACKEND_SETUP.md (backend troubleshooting)

**What's Stashed** (on `feature/agent-builder-ui`):
- 📦 Agent builder work (recommend deleting per analysis)

**Branch Status**:
- `main` - Clean, working, ready for frontend work
- `feature/agent-builder-ui` - Has stashed agent work (not needed)

---

## My Recommendation

**Start Here** (2-3 hours):

1. **Add better error handling to Chat** ⚠️
   - Show helpful message when backend isn't available
   - "Backend not connected. See BACKEND_SETUP.md"
   - Make it user-friendly

2. **Build Evaluation Module UI** 🎯
   - Create empty pages (no API calls yet)
   - Design the evaluator list
   - Design model comparison view
   - **This is the biggest missing piece!**

3. **Polish Dataset Page** 📊
   - Add file upload UI (button + drag-drop zone)
   - Design quality test badges
   - **Just UI, no backend needed**

**All of this can be done WITHOUT the backend!**

---

## Questions?

**Q: Can I test anything without backend?**  
A: Yes! All UI/UX work. Just can't test API integration.

**Q: Should I find the backend now?**  
A: Not urgent. Frontend has lots of work. But would be good to locate it for later.

**Q: What about our agent builder work?**  
A: Per CURRENT_STATE_ANALYSIS.md, recommend deleting it. Chat should handle that workflow.

**Q: What's the priority?**  
A: Evaluation module (biggest gap) → Chat polish → Dataset/Model polish

---

## Next Action

**Pick one**:

A. **"Let me build the Evaluation Module UI"** ← Recommended  
   - I'll create the pages/components
   - No backend needed
   - Fills major gap

B. **"Help me find the backend"**  
   - I'll search for amortized repo
   - Get everything connected
   - Full testing enabled

C. **"Create the mock backend"**  
   - Quick testing
   - Chat works with fake data
   - Good for demos

**Which sounds best to you?**
