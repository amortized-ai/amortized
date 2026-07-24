# Backend Setup Guide

## Issue: Chat Not Working

**Error**: "An error occurred. Please try again." when using Chat

**Root Cause**: The Amortized Studio is a **frontend-only application**. It requires the **amortized FastAPI backend** to be running to handle chat requests.

## Current State

### What's Running ✅
- **Frontend**: Vite dev server on `http://localhost:5173`
- **Mock API**: Some endpoints (like `/api/v1/health`) are being mocked/proxied

### What's Missing ❌
- **amortized Backend**: FastAPI server with:
  - `/agent/session` - OpenCode chat endpoints
  - `/api/v1/jobs` - Job management
  - `/api/v1/datasets` - Dataset CRUD
  - `/api/v1/evaluators` - Evaluator registry
  - WebSocket at `/api/v1/ws` - Real-time updates
  - MLflow integration

## Solution Options

### Option 1: Find/Start the amortized Backend

The [amortized](https://github.com/amortized-ai/amortized) library should provide the backend.

**If you have it installed**:
```bash
# Look for amortized installation
which amortized

# Or check if it's in a nearby directory
find ~ -name "amortized" -type d 2>/dev/null | grep -v node_modules
```

**To start it** (typical FastAPI setup):
```bash
cd /path/to/amortized
uvicorn amortized.api.server:app --reload --port 8000
```

Then configure Studio to point to it:
```bash
# In /Users/nmalepat/studio
echo "VITE_API_URL=http://localhost:8000" > .env
```

### Option 2: Use Mock Backend (Quick Test)

For testing the UI without the real backend, create a minimal mock:

```bash
# Create a simple mock server
cd /Users/nmalepat/studio
```

```javascript
// mock-backend.js
import express from 'express'
import cors from 'cors'

const app = express()
app.use(cors())
app.use(express.json())

// Health endpoint
app.get('/api/v1/health', (req, res) => {
  res.json({ status: 'ok', version: '3.0.0' })
})

// Mock agent session
let sessions = {}
app.post('/agent/session', (req, res) => {
  const sessionId = `session-${Date.now()}`
  sessions[sessionId] = []
  res.json({ id: sessionId })
})

// Mock agent message
app.post('/agent/session/:id/message', (req, res) => {
  const { text } = req.body.parts[0]
  res.json({
    parts: [{
      type: 'text',
      text: `I understand you want to: "${text}"\n\nHowever, this is a mock response. To get real AI assistance, you need to connect to the amortized backend server.`
    }],
    info: {
      providerID: 'mock',
      modelID: 'mock-model'
    }
  })
})

// Mock jobs
app.get('/api/v1/jobs', (req, res) => {
  res.json([])
})

app.listen(8000, () => {
  console.log('Mock backend running on http://localhost:8000')
})
```

Then:
```bash
node mock-backend.js
```

And configure Studio:
```bash
echo "VITE_API_URL=http://localhost:8000" > .env
npm run dev  # Restart Vite
```

### Option 3: Check Vite Proxy Configuration

The Studio might have proxy rules configured in `vite.config.ts`:

```bash
cat vite.config.ts
```

If there's a proxy, it might be pointing to a backend that's not running.

## Verification

After starting the backend:

```bash
# Test health endpoint
curl http://localhost:8000/api/v1/health

# Test session creation
curl -X POST http://localhost:8000/agent/session \
  -H "Content-Type: application/json" \
  -d '{}'

# Should return: {"id": "session-..."}
```

## Current VITE_API_URL

By default, it's empty string (`""`), which means requests go to the same origin (Vite dev server).

Vite likely has proxy rules that forward `/api/*` and `/agent/*` to a backend.

**To check**:
```bash
# Look for proxy config
cat /Users/nmalepat/studio/vite.config.ts
```

## Recommended Next Step

1. **Find the amortized backend repository**
2. **Start the backend server**
3. **Configure VITE_API_URL** (if needed)
4. **Refresh Studio** - Chat should work

**Or** use the mock backend above for quick testing.

---

**TL;DR**: Chat needs a backend server running. The Studio is just a UI.
