#!/usr/bin/env node
/**
 * Morty Agent — Interactive Mock Backend for Amortized Studio
 *
 * Provides guided, multi-turn workflows like Oumi.ai:
 *   identify → clarify → plan → execute → results
 *
 * Each session tracks conversation state so Morty asks questions,
 * proposes plans, simulates execution, and shows results.
 */

import express from 'express'
import cors from 'cors'

const app = express()
app.use(cors())
app.use(express.json())

// ==================== Session Storage ====================

const sessions = new Map()
let sessionCounter = 1

// ==================== Flow Definitions ====================

const FLOWS = {

  // ─────────────── Support Ticket Classifier ───────────────
  classifier: {
    name: 'Support Ticket Classifier',
    stages: {
      clarify_1(userText, ctx) {
        ctx.originalRequest = userText
        return {
          text: `Great choice — ticket classification is one of the highest-ROI tasks for small language models.

To design the right solution, I need to understand your setup. **What categories should tickets be classified into?**

For example:
- Billing, Technical, Account, General
- Urgent, High, Medium, Low (priority)
- Bug, Feature Request, Question, Complaint

Describe your categories or tell me about your support workflow and I'll suggest them.`,
          nextStage: 'clarify_2'
        }
      },

      clarify_2(userText, ctx) {
        ctx.categories = userText
        return {
          text: `Got it — I'll design around those categories.

**How many tickets do you handle per day?** This helps me size the training data and estimate your cost savings.

- Under 100/day
- 100–1,000/day
- 1,000–10,000/day
- 10,000+/day`,
          nextStage: 'clarify_3'
        }
      },

      clarify_3(userText, ctx) {
        ctx.volume = userText
        const volumeNum = userText.includes('10,000') || userText.includes('10000') ? 10000
          : userText.includes('1,000') || userText.includes('1000') ? 1000
          : userText.includes('100') ? 100 : 500

        const monthlyCost = (volumeNum * 30 * 0.50).toLocaleString()
        const slmCost = (volumeNum * 30 * 0.008).toLocaleString()
        const savings = (volumeNum * 30 * 0.492).toLocaleString()
        ctx.volumeNum = volumeNum
        ctx.monthlyCost = monthlyCost
        ctx.slmCost = slmCost
        ctx.savings = savings

        return {
          text: `Perfect. Here's my plan for your classifier:

**The Plan**

| Step | What | Time |
|------|------|------|
| 1 | Generate 500 synthetic support tickets using SDG | ~5 min |
| 2 | Fine-tune Llama 3.2 3B with LoRA | ~20 min |
| 3 | Evaluate accuracy vs GPT-4 baseline | ~3 min |
| 4 | Deploy locally on your infrastructure | ~2 min |

**Projected Cost Savings**

| | GPT-4 (current) | Your SLM |
|---|---|---|
| Per ticket | $0.50 | $0.008 |
| Monthly (${volumeNum * 30} tickets) | $${monthlyCost} | $${slmCost} |
| **Monthly savings** | — | **$${savings}** |

**Expected accuracy**: 93–95% (GPT-4 baseline: 96%)

Ready to go?

1) Start building
2) Adjust the plan
3) I have example tickets to upload first`,
          nextStage: 'execute'
        }
      },

      execute(userText, ctx) {
        const lower = userText.toLowerCase()
        if (lower.includes('adjust') || lower.includes('modify') || lower.includes('change')) {
          return {
            text: `Sure — what would you like to change?

1) Use a different base model
2) Change training data size
3) Add more classification categories`,
            nextStage: 'execute'
          }
        }
        if (lower.includes('example') || lower.includes('upload')) {
          return {
            text: `I'd love to use your real tickets as seed data! For now in this demo, I'll generate synthetic examples based on your categories.

In the full version, you'll be able to upload a JSONL file with your tickets and I'll use them as seeds for synthetic data generation.

1) Start building with synthetic data
2) Go back to the plan`,
            nextStage: 'execute'
          }
        }
        if (lower.includes('different') || lower.includes('model')) {
          return {
            text: `Here are the available base models:

| Model | Params | Speed | Accuracy | Cost to Train |
|-------|--------|-------|----------|---------------|
| Llama 3.2 1B | 1B | Fastest | ~90% | $0.50 |
| **Llama 3.2 3B** | 3B | Fast | ~94% | $2.00 |
| Llama 3.2 8B | 8B | Medium | ~96% | $8.00 |
| Mistral 7B | 7B | Medium | ~95% | $6.00 |

1) Use Llama 3.2 3B (recommended)
2) Use Llama 3.2 1B (fastest, cheapest)
3) Use Llama 3.2 8B (highest accuracy)`,
            nextStage: 'execute'
          }
        }

        return {
          text: `**Step 1/4 — Synthetic Data Generation**

Generating 500 support tickets across your categories using Amortized SDG...

✅ **Done!** Created 500 labeled tickets — 125 per category (balanced), including edge cases and ambiguous examples. Dataset saved to MLflow.

| Metric | Value |
|--------|-------|
| Total samples | 500 |
| Format | JSONL |
| Categories | Balanced (125 each) |
| Edge cases | Included |

Ready to fine-tune the model on this data?

1) Continue to fine-tuning
2) Regenerate with more samples
3) View sample data`,
          nextStage: 'execute_2'
        }
      },

      execute_2(userText, ctx) {
        const lower = userText.toLowerCase()
        if (lower.includes('regenerate') || lower.includes('more samples')) {
          return {
            text: `I'll generate a larger dataset. How many samples?

1) 1,000 samples (recommended)
2) 2,000 samples
3) Keep the original 500`,
            nextStage: 'execute_2'
          }
        }
        if (lower.includes('view') || lower.includes('sample data')) {
          return {
            text: `**Sample Generated Ticket**

> Subject: Can't access billing portal
> Body: I've been trying to log into the billing section for the past hour. It keeps showing a 403 error. I need to download my invoice for Q2 before end of day. This is urgent — our finance team is waiting.
> **Label: Billing / Urgent**

> Subject: Feature request - dark mode
> Body: Would love to see a dark mode option in the dashboard. My team works late shifts and the bright interface is hard on the eyes.
> **Label: Feature Request / Low**

These are 2 of your 500 generated tickets. Each has realistic content and accurate labels.

1) Continue to fine-tuning
2) Regenerate with different categories`,
            nextStage: 'execute_2'
          }
        }

        return {
          text: `**Step 2/4 — Fine-Tuning**

Training Llama 3.2 3B with LoRA on your 500-ticket dataset...

✅ **Done!** Model fine-tuned successfully.

| Parameter | Value |
|-----------|-------|
| Base model | Llama 3.2 3B Instruct |
| Method | LoRA (rank=16, alpha=32) |
| Epochs | 3 |
| Training loss | 0.42 → 0.08 |
| Duration | 18 minutes |

The model has learned your classification categories. Ready to evaluate it?

1) Continue to evaluation
2) Train for more epochs
3) View training details`,
          nextStage: 'execute_3'
        }
      },

      execute_3(userText, ctx) {
        const lower = userText.toLowerCase()
        if (lower.includes('more epochs') || lower.includes('train')) {
          return {
            text: `Running 2 additional epochs... Done! Loss improved from 0.08 → 0.05. Marginal gains — the model was already well-converged.

1) Continue to evaluation
2) View updated metrics`,
            nextStage: 'execute_3'
          }
        }
        if (lower.includes('detail') || lower.includes('view')) {
          return {
            text: `**Training Configuration**

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-4 with cosine schedule |
| Batch size | 8 |
| LoRA dropout | 0.05 |
| GPU | A100-40GB |
| Loss curve | Epoch 1: 0.42→0.21, Epoch 2: 0.21→0.12, Epoch 3: 0.12→0.08 |

1) Continue to evaluation
2) Retrain with different settings`,
            nextStage: 'execute_3'
          }
        }

        return {
          text: `**Step 3/4 — Evaluation**

Running your classifier against a held-out test set and GPT-4 baseline...

✅ **Done!** Evaluation complete.

**Accuracy Comparison**

| Model | Accuracy | Latency (p50) | Cost/ticket |
|-------|----------|---------------|-------------|
| GPT-4 | 96.2% | 820ms | $0.500 |
| **Your SLM** | **94.1%** | **48ms** | **$0.008** |

**Per-Category Breakdown**

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Billing | 95.2% | 93.8% | 94.5% |
| Technical | 93.1% | 94.6% | 93.8% |
| Account | 94.8% | 93.2% | 94.0% |
| General | 93.4% | 94.8% | 94.1% |

Your model reaches 94.1% accuracy — 98% of GPT-4's performance at 1.6% of the cost. Ready to see the final summary?

1) Show final results
2) Run evaluation on different data
3) Try to improve accuracy`,
          nextStage: 'execute_4'
        }
      },

      execute_4(userText, ctx) {
        const lower = userText.toLowerCase()
        if (lower.includes('improve') || lower.includes('accuracy')) {
          return {
            text: `To improve accuracy, I can:

1) Generate 500 more training samples and retrain
2) Use a larger base model (Llama 3.2 8B)
3) Show final results as-is`,
            nextStage: 'execute_4'
          }
        }

        return {
          text: `**Step 4/4 — Your Classifier is Ready!**

Here's the complete summary:

**ROI Summary**

| Metric | Value |
|--------|-------|
| Training cost | $2.00 (one-time) |
| Monthly savings | $${ctx.savings || '14,760'} |
| Annual savings | $${ctx.savings ? (parseFloat(ctx.savings.replace(/,/g, '')) * 12).toLocaleString() : '177,120'} |
| Accuracy | 94.1% (vs GPT-4's 96.2%) |
| Latency | 48ms (17x faster than GPT-4) |

**Artifacts saved:**
Model weights, dataset, and eval report are all tracked in MLflow.

What would you like to do next?

1) Deploy this model
2) Build another agent
3) Run more evaluations
4) View training details`,
          nextStage: 'done'
        }
      },

      done(userText, ctx) {
        const lower = userText.toLowerCase()
        if (lower.includes('deploy')) {
          return {
            text: `**Deployment Options**

Your trained model can be served via:

| Method | Latency | Throughput | Setup |
|--------|---------|------------|-------|
| vLLM (recommended) | 48ms | 200 req/s | Simple |
| TGI | 52ms | 180 req/s | Simple |
| ONNX Runtime | 35ms | 300 req/s | Moderate |

For a ${ctx.volumeNum || 500}-ticket/day workload, a single A10G GPU handles everything comfortably.

In the full version, I'll deploy to your OpenShift AI cluster automatically. For now, your trained model weights are saved and ready.

1) Build another agent
2) Back to overview`,
            nextStage: 'done'
          }
        }
        if (lower.includes('another') || lower.includes('build') || lower.includes('new')) {
          return { text: null, nextStage: 'reset' }
        }
        if (lower.includes('detail') || lower.includes('training')) {
          return {
            text: `**Training Details**

**Configuration**

| Parameter | Value |
|-----------|-------|
| Base model | Llama 3.2 3B Instruct |
| Method | LoRA (rank=16, alpha=32, dropout=0.05) |
| Dataset | 500 synthetic tickets (400 train / 100 test) |
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-4 with cosine schedule |
| Training time | 18 minutes on A100-40GB |

**Loss Curve**

| Epoch | Loss |
|-------|------|
| 1 | 0.42 → 0.21 |
| 2 | 0.21 → 0.12 |
| 3 | 0.12 → 0.08 |

**Artifacts**

| File | Path |
|------|------|
| Model weights | \`models/ticket-classifier-v1/\` |
| Dataset | \`datasets/synthetic-tickets-500.jsonl\` |
| Eval report | \`evals/classifier-vs-gpt4.json\` |
| MLflow run | \`run-abc123\` |

1) Deploy this model
2) Build another agent
3) Back to overview`,
            nextStage: 'done'
          }
        }
        return { text: null, nextStage: 'reset' }
      }
    }
  },

  // ─────────────── Custom Chatbot ───────────────
  chatbot: {
    name: 'Custom Chatbot',
    stages: {
      clarify_1(userText, ctx) {
        ctx.originalRequest = userText
        return {
          text: `I'd love to help you build a custom chatbot! To design the right solution, tell me a bit more.

**What will your chatbot handle?** Describe the main types of questions or requests it should respond to.

For example:
- "Answer questions about our product pricing and features"
- "Help customers troubleshoot common issues"
- "Handle appointment scheduling and FAQ"`,
          nextStage: 'clarify_2'
        }
      },

      clarify_2(userText, ctx) {
        ctx.useCase = userText
        return {
          text: `Great — that gives me a clear picture.

**What tone should the chatbot use?** This affects how I generate training conversations.

- Professional and formal
- Friendly and conversational
- Technical and precise
- Warm but efficient`,
          nextStage: 'clarify_3'
        }
      },

      clarify_3(userText, ctx) {
        ctx.tone = userText
        return {
          text: `Perfect. Here's what I'll build:

**The Plan**

| Step | What | Time |
|------|------|------|
| 1 | Generate 500 synthetic conversations matching your use case | ~8 min |
| 2 | Fine-tune Llama 3.2 3B on the conversation data | ~25 min |
| 3 | Evaluate response quality vs GPT-4 | ~5 min |
| 4 | Package for deployment | ~2 min |

**Your chatbot will:**
- Handle the use cases you described
- Respond in a ${ctx.tone?.toLowerCase().includes('formal') ? 'professional' : ctx.tone?.toLowerCase().includes('technical') ? 'technical' : 'friendly'} tone
- Run locally on your infrastructure
- Cost ~$0.008/response vs $0.50 with GPT-4

**Projected savings** at 1,000 conversations/day:
- Current cost: **$15,000/month** (GPT-4)
- Your SLM: **$240/month**
- Savings: **$14,760/month**

1) Start building
2) Adjust the plan
3) I want to provide sample conversations`,
          nextStage: 'execute'
        }
      },

      execute(userText, ctx) {
        const lower = userText.toLowerCase()
        if (lower.includes('adjust') || lower.includes('sample') || lower.includes('provide')) {
          return {
            text: `In the full version, you can upload sample conversations as JSONL files and I'll use them as seeds. For now, I'll generate synthetic data based on your description.

1) Start building with synthetic data
2) Go back to the plan`,
            nextStage: 'execute'
          }
        }

        return {
          text: `**Step 1/4 — Conversation Generation**

Generating 500 multi-turn conversations for your use case...

✅ **Done!** Created 500 conversations (avg 4 turns each), covering your use case with edge cases, clarifying questions, and handoff scenarios. Tone calibrated to "${ctx.tone || 'friendly'}."

| Metric | Value |
|--------|-------|
| Conversations | 500 |
| Avg turns | 4 per conversation |
| Total messages | ~2,000 |
| Tone | ${ctx.tone || 'Friendly'} |

Ready to train the model on these conversations?

1) Continue to fine-tuning
2) View sample conversation
3) Regenerate with more data`,
          nextStage: 'execute_2'
        }
      },

      execute_2(userText, ctx) {
        if (userText.toLowerCase().includes('view') || userText.toLowerCase().includes('sample')) {
          return {
            text: `**Sample Generated Conversation**

**Customer**: Hi, I need help with my account.

**Chatbot**: Of course! I'd be happy to help. Could you let me know what you need?

**Customer**: I can't log in, it keeps saying invalid password.

**Chatbot**: I understand how frustrating that is. I'll send a password reset link to your registered email. Should I do that now?

**Customer**: Yes please.

**Chatbot**: Done! Check your inbox — the link should arrive within 2 minutes. Anything else I can help with?

---

This is 1 of 500 generated conversations. Each maintains your preferred tone.

1) Continue to fine-tuning
2) Regenerate conversations`,
            nextStage: 'execute_2'
          }
        }

        return {
          text: `**Step 2/4 — Fine-Tuning**

Training Llama 3.2 3B on your conversation data...

✅ **Done!** Model fine-tuned successfully.

| Parameter | Value |
|-----------|-------|
| Method | SFT with chat template |
| Epochs | 3 |
| Training loss | 0.51 → 0.09 |
| Duration | 22 minutes |

Ready to evaluate the chatbot's quality?

1) Continue to evaluation
2) Train for more epochs`,
          nextStage: 'execute_3'
        }
      },

      execute_3(userText, ctx) {
        return {
          text: `**Step 3/4 — Quality Evaluation**

Testing on 100 held-out conversations with LLM-as-judge scoring...

✅ **Done!** Quality scores are in.

| Dimension | Your SLM | GPT-4 |
|-----------|----------|-------|
| Relevance | 4.6 | 4.8 |
| Helpfulness | 4.5 | 4.7 |
| Tone Match | 4.7 | 4.5 |
| Accuracy | 4.4 | 4.8 |
| **Overall** | **4.55** | **4.70** |

Your chatbot scores *higher* on tone matching because it was trained specifically for your preferred style.

1) Show final results
2) Try to improve scores`,
          nextStage: 'execute_4'
        }
      },

      execute_4(userText, ctx) {
        return {
          text: `**Step 4/4 — Your Chatbot is Ready!**

**Performance Summary**

| Metric | Your SLM | GPT-4 |
|--------|----------|-------|
| Quality score | 4.55/5 | 4.70/5 |
| Latency (p50) | 62ms | 950ms |
| Cost/response | $0.008 | $0.50 |

**ROI at 1,000 conversations/day:**

| Metric | Value |
|--------|-------|
| Training cost | $2.00 (one-time) |
| Monthly savings | $14,760 |
| Annual savings | $177,120 |

What would you like to do next?

1) Deploy this chatbot
2) Build another agent
3) View sample conversations
4) Run more evaluations`,
          nextStage: 'done'
        }
      },

      done(userText, ctx) {
        const lower = userText.toLowerCase()
        if (lower.includes('sample') || lower.includes('conversation')) {
          return {
            text: `**Sample Generated Conversation**

---

**Customer**: Hi, I need help with my account.

**Your Chatbot**: Of course! I'd be happy to help with your account. Could you let me know what specifically you need assistance with? For example:
- Password reset
- Billing question
- Account settings change
- Something else

**Customer**: I can't log in, it keeps saying invalid password.

**Your Chatbot**: I understand how frustrating that is. Let me help you get back into your account. I'll send a password reset link to the email address on file. Would you like me to do that?

**Customer**: Yes please.

**Your Chatbot**: Done! I've sent a reset link to your registered email. It should arrive within 2 minutes. If you don't see it, please check your spam folder. Is there anything else I can help with?

---

This is one of 500 conversations generated. Each covers different scenarios and maintains your preferred tone.

1) Deploy this chatbot
2) Build another agent
3) Back to overview`,
            nextStage: 'done'
          }
        }
        if (lower.includes('another') || lower.includes('build') || lower.includes('new') || lower.includes('overview')) {
          return { text: null, nextStage: 'reset' }
        }
        if (lower.includes('deploy')) {
          return {
            text: `Your chatbot model is packaged and ready for deployment. In the full version, I'll deploy directly to your OpenShift AI cluster.

**Deployment bundle:** Model weights (LoRA adapter, 45MB), chat template configuration, serving config for vLLM, and OpenAI-compatible API endpoint spec.

1) Build another agent
2) Back to overview`,
            nextStage: 'done'
          }
        }
        return { text: null, nextStage: 'reset' }
      }
    }
  },

  // ─────────────── Invoice Data Extractor ───────────────
  extractor: {
    name: 'Invoice Data Extractor',
    stages: {
      clarify_1(userText, ctx) {
        ctx.originalRequest = userText
        return {
          text: `Invoice extraction is a perfect use case for fine-tuned SLMs — structured output with high accuracy.

**What format are your invoices?**

- PDF documents
- Scanned images (OCR needed)
- Email attachments
- Digital/electronic (already text)
- Mix of formats`,
          nextStage: 'clarify_2'
        }
      },

      clarify_2(userText, ctx) {
        ctx.format = userText
        return {
          text: `Got it. **What specific fields do you need extracted?**

Common fields I can handle:
- Vendor name and address
- Invoice number and date
- Line items (description, quantity, unit price)
- Subtotal, tax, total amount
- Payment terms and due date
- PO number

Tell me which fields matter most, or just say "all of the above."`,
          nextStage: 'clarify_3'
        }
      },

      clarify_3(userText, ctx) {
        ctx.fields = userText
        return {
          text: `Here's the plan:

**The Plan**

| Step | What | Time |
|------|------|------|
| 1 | Generate 300 synthetic invoices with structured output | ~5 min |
| 2 | Fine-tune Llama 3.2 3B for structured extraction | ~20 min |
| 3 | Validate field accuracy on test set | ~3 min |
| 4 | Package with JSON output schema | ~1 min |

**Your extractor will:**
- Parse invoices and output structured JSON
- Handle variations in layout and formatting
- Extract all requested fields with high precision

**Cost comparison** at 500 invoices/day:

| | GPT-4 | Your SLM |
|---|---|---|
| Per invoice | $0.60 | $0.01 |
| Monthly | $9,000 | $150 |
| **Savings** | — | **$8,850/month** |

1) Start building
2) Adjust the plan
3) I want to upload sample invoices`,
          nextStage: 'execute'
        }
      },

      execute(userText, ctx) {
        const lower = userText.toLowerCase()
        if (lower.includes('adjust') || lower.includes('upload') || lower.includes('sample')) {
          return {
            text: `In the full version, uploading sample invoices helps me generate more accurate synthetic data. For now, I'll use realistic templates.

1) Start building with synthetic data
2) Go back to the plan`,
            nextStage: 'execute'
          }
        }

        return {
          text: `**Step 1/4 — Synthetic Invoice Generation**

Generating 300 invoices with ground-truth JSON labels...

✅ **Done!** Created 300 invoices across 15 different layouts, with variations in formatting, currencies, and languages.

| Metric | Value |
|--------|-------|
| Invoices generated | 300 |
| Unique layouts | 15 |
| Edge cases | Multi-page, partial data, handwritten notes |
| Output format | JSONL with ground-truth labels |

Ready to train the extraction model?

1) Continue to fine-tuning
2) View sample invoice
3) Generate more invoices`,
          nextStage: 'execute_2'
        }
      },

      execute_2(userText, ctx) {
        if (userText.toLowerCase().includes('view') || userText.toLowerCase().includes('sample')) {
          return {
            text: `**Sample Extraction**

Input: A standard invoice from "Acme Supplies Inc."

Output:
\`\`\`json
{
  "vendor": "Acme Supplies Inc.",
  "invoice_number": "INV-2026-0847",
  "date": "2026-06-15",
  "line_items": [
    { "description": "Widget A", "qty": 100, "unit_price": 12.50 },
    { "description": "Widget B", "qty": 50, "unit_price": 24.00 }
  ],
  "subtotal": 2450.00,
  "tax": 196.00,
  "total": 2646.00
}
\`\`\`

1) Continue to fine-tuning
2) Generate more invoices`,
            nextStage: 'execute_2'
          }
        }

        return {
          text: `**Step 2/4 — Fine-Tuning**

Training Llama 3.2 3B for structured JSON extraction...

✅ **Done!** Model trained for structured output.

| Parameter | Value |
|-----------|-------|
| Method | SFT with function-calling format |
| Epochs | 4 |
| Training loss | 0.38 → 0.06 |
| Duration | 20 minutes |

Ready to validate extraction accuracy?

1) Continue to validation
2) Train for more epochs`,
          nextStage: 'execute_3'
        }
      },

      execute_3(userText, ctx) {
        return {
          text: `**Step 3/4 — Field Validation**

Testing extraction accuracy on 60 held-out invoices...

✅ **Done!** Field-level accuracy results:

| Field | Your SLM | GPT-4 |
|-------|----------|-------|
| Vendor name | 97.8% | 98.5% |
| Invoice number | 98.3% | 99.1% |
| Date | 96.2% | 97.8% |
| Line items | 94.1% | 96.3% |
| Total amount | 98.7% | 99.2% |
| **Overall** | **96.7%** | **98.0%** |

Your model matches 98.7% of GPT-4's accuracy on the highest-value field (total amount).

1) Show final results
2) Try to improve accuracy`,
          nextStage: 'execute_4'
        }
      },

      execute_4(userText, ctx) {
        return {
          text: `**Step 4/4 — Your Extractor is Ready!**

Model packaged with JSON output schema and ready for deployment.

**Monthly savings at 500 invoices/day: $8,850**

| Metric | Value |
|--------|-------|
| Training cost | $2.00 (one-time) |
| Overall accuracy | 96.7% |
| Latency | 45ms per invoice |
| Cost per invoice | $0.01 (vs $0.60 GPT-4) |

What would you like to do next?

1) Deploy this extractor
2) Build another agent
3) View more extraction examples`,
          nextStage: 'done'
        }
      },

      done(userText, ctx) {
        const lower = userText.toLowerCase()
        if (lower.includes('another') || lower.includes('build') || lower.includes('new') || lower.includes('overview')) {
          return { text: null, nextStage: 'reset' }
        }
        if (lower.includes('example') || lower.includes('more')) {
          return {
            text: `Here's another extraction example with a more complex invoice:

\`\`\`json
{
  "vendor": "Global Tech Solutions",
  "invoice_number": "GTS-2026-12445",
  "date": "2026-07-01",
  "po_number": "PO-8829",
  "line_items": [
    { "description": "Cloud hosting (July)", "qty": 1, "unit_price": 2400.00 },
    { "description": "Support tier upgrade", "qty": 1, "unit_price": 500.00 },
    { "description": "API calls overage", "qty": 15000, "unit_price": 0.02 }
  ],
  "subtotal": 3200.00,
  "tax": 256.00,
  "total": 3456.00,
  "payment_terms": "Net 15",
  "due_date": "2026-07-16"
}
\`\`\`

The model correctly handled: mixed unit types, overage charges, and a PO number field.

1) Deploy this extractor
2) Build another agent`,
            nextStage: 'done'
          }
        }
        return {
          text: `Your extractor is ready. In the full version, I'll deploy to your infrastructure automatically.

1) Build another agent
2) Back to overview`,
          nextStage: 'done'
        }
      }
    }
  },

  // ─────────────── Sentiment Analysis ───────────────
  sentiment: {
    name: 'Sentiment Analysis Model',
    stages: {
      clarify_1(userText, ctx) {
        ctx.originalRequest = userText
        return {
          text: `Sentiment analysis is one of the best tasks for small models — they match frontier accuracy easily.

**What type of text will you analyze?**

- Customer reviews (product, service)
- Social media posts (Twitter, Reddit)
- Survey responses (NPS, CSAT)
- Support chat transcripts
- Something else`,
          nextStage: 'clarify_2'
        }
      },

      clarify_2(userText, ctx) {
        ctx.textSource = userText
        return {
          text: `And **what level of granularity do you need?**

- Simple (Positive / Negative / Neutral)
- Detailed (add Mixed, Very Positive, Very Negative)
- Aspect-based (sentiment per feature: "price is good but support is slow")
- With confidence scores`,
          nextStage: 'clarify_3'
        }
      },

      clarify_3(userText, ctx) {
        ctx.granularity = userText
        const isAspect = userText.toLowerCase().includes('aspect')
        const modelSize = isAspect ? '3B' : '1B'
        ctx.modelSize = modelSize

        return {
          text: `Here's the plan:

**The Plan**

| Step | What | Time |
|------|------|------|
| 1 | Generate 1,000 synthetic ${ctx.textSource?.toLowerCase().includes('review') ? 'reviews' : 'texts'} with labels | ~6 min |
| 2 | Fine-tune Llama 3.2 ${modelSize} ${isAspect ? '(aspect-based needs more capacity)' : '(1B is perfect for sentiment!)'} | ~${isAspect ? '20' : '10'} min |
| 3 | Benchmark against GPT-4 | ~3 min |
| 4 | Package for real-time inference | ~1 min |

**Why ${modelSize}?** ${isAspect
  ? 'Aspect-based sentiment needs to identify entities AND classify, so the 3B model gives better results.'
  : 'Simple sentiment classification is well within the 1B model\'s capability — and it\'s the cheapest to run.'}

**Cost at 10,000 analyses/day:**

| | GPT-4 | Your SLM |
|---|---|---|
| Per analysis | $0.30 | $0.003 |
| Monthly | $90,000 | $900 |
| **Savings** | — | **$89,100/month** |

1) Start building
2) Adjust the plan
3) Use a larger model`,
          nextStage: 'execute'
        }
      },

      execute(userText, ctx) {
        const lower = userText.toLowerCase()
        if (lower.includes('adjust') || lower.includes('larger')) {
          return {
            text: `Available models for sentiment analysis:

| Model | Speed | Accuracy | Monthly cost (10k/day) |
|-------|-------|----------|----------------------|
| **Llama 3.2 1B** | 10ms | ~94% | $900 |
| Llama 3.2 3B | 25ms | ~95% | $2,400 |
| Mistral 7B | 45ms | ~96% | $5,400 |

1) Use Llama 3.2 1B (fastest, recommended)
2) Use Llama 3.2 3B (slightly more accurate)
3) Go back to the plan`,
            nextStage: 'execute'
          }
        }

        const textType = ctx.textSource?.toLowerCase().includes('review') ? 'reviews' : 'texts'
        return {
          text: `**Step 1/4 — Data Generation**

Generating 1,000 labeled ${textType} using SDG...

✅ **Done!** Created 1,000 labeled samples — balanced across sentiment categories with domain-specific vocabulary, including sarcasm, mixed sentiment, and edge cases.

| Metric | Value |
|--------|-------|
| Samples | 1,000 |
| Categories | Balanced |
| Includes | Sarcasm, mixed sentiment, edge cases |

Ready to train the sentiment model?

1) Continue to fine-tuning
2) View sample data
3) Generate more samples`,
          nextStage: 'execute_2'
        }
      },

      execute_2(userText, ctx) {
        if (userText.toLowerCase().includes('view') || userText.toLowerCase().includes('sample')) {
          return {
            text: `**Sample Generated Data**

> "The product quality is amazing but the shipping took forever. Mixed feelings overall."
> **Label: Mixed**

> "Absolutely terrible customer service. Will never buy again."
> **Label: Negative**

> "It's fine. Does what it's supposed to do. Nothing special."
> **Label: Neutral**

1) Continue to fine-tuning
2) Generate more samples`,
            nextStage: 'execute_2'
          }
        }

        return {
          text: `**Step 2/4 — Fine-Tuning**

Training Llama 3.2 ${ctx.modelSize || '1B'} with SFT classification head...

✅ **Done!** Model fine-tuned.

| Parameter | Value |
|-----------|-------|
| Model | Llama 3.2 ${ctx.modelSize || '1B'} |
| Epochs | 5 |
| Training loss | 0.35 → 0.04 |
| Duration | ${ctx.modelSize === '3B' ? '18' : '8'} minutes |

Ready to benchmark against GPT-4?

1) Continue to benchmarking
2) Train for more epochs`,
          nextStage: 'execute_3'
        }
      },

      execute_3(userText, ctx) {
        return {
          text: `**Step 3/4 — Benchmarking**

Comparing against GPT-4 on 200 test samples...

✅ **Done!** Results:

| Category | Your SLM | GPT-4 |
|----------|----------|-------|
| Positive | 95.8% | 96.2% |
| Negative | 94.2% | 95.8% |
| Neutral | 92.1% | 93.5% |
| **Overall** | **93.8%** | **94.9%** |

Your model runs at **${ctx.modelSize === '3B' ? '25ms' : '10ms'} per analysis** — fast enough for real-time dashboards.

1) Show final results
2) Try to improve accuracy`,
          nextStage: 'execute_4'
        }
      },

      execute_4(userText, ctx) {
        return {
          text: `**Step 4/4 — Your Sentiment Model is Ready!**

Optimized for real-time inference and ready to deploy.

**Monthly ROI at 10,000 analyses/day:**

| Metric | Value |
|--------|-------|
| Savings | $89,100/month |
| Training cost | $${ctx.modelSize === '3B' ? '2.00' : '0.50'} (one-time) |
| Latency | ${ctx.modelSize === '3B' ? '25ms' : '10ms'} per analysis |
| Accuracy | 93.8% (vs GPT-4's 94.9%) |

What would you like to do next?

1) Deploy this model
2) Build another agent
3) Run on my own test data`,
          nextStage: 'done'
        }
      },

      done(userText, ctx) {
        const lower = userText.toLowerCase()
        if (lower.includes('another') || lower.includes('build') || lower.includes('new') || lower.includes('overview')) {
          return { text: null, nextStage: 'reset' }
        }
        return {
          text: `Your sentiment model is ready for deployment. In the full version, I'll push it to your inference cluster.

1) Build another agent
2) Back to overview`,
          nextStage: 'done'
        }
      }
    }
  }
}

// ==================== Flow Engine ====================

function detectFlow(text) {
  const lower = text.toLowerCase()
  if (lower.includes('support ticket') || lower.includes('ticket classifier') || lower.includes('classify') || lower.includes('triage') || lower.includes('classifier')) return 'classifier'
  if (lower.includes('chatbot') || lower.includes('chat bot') || lower.includes('custom chatbot')) return 'chatbot'
  if (lower.includes('invoice') || lower.includes('extract') || lower.includes('parse') || lower.includes('data extraction')) return 'extractor'
  if (lower.includes('sentiment') || lower.includes('opinion') || lower.includes('review analysis')) return 'sentiment'
  return null
}

function generateResponse(session, userText) {
  // If session has no flow yet, try to detect one
  if (!session.flow) {
    const detectedFlow = detectFlow(userText)
    if (detectedFlow) {
      session.flow = detectedFlow
      session.stage = 'clarify_1'
      session.context = {}
    } else {
      // No flow detected — offer options
      return `I'm Morty, your AI assistant for building task-specific models that run on your infrastructure at a fraction of the cost.

**What would you like to build?**

1) Support ticket classifier
2) Custom chatbot for my business
3) Invoice data extractor
4) Sentiment analysis model`
    }
  }

  // Handle "reset" — user wants to start over
  if (session.stage === 'reset' || userText.toLowerCase().includes('start over')) {
    session.flow = null
    session.stage = null
    session.context = {}
    return `No problem! Let's start fresh.

**What would you like to build?**

1) Support ticket classifier
2) Custom chatbot for my business
3) Invoice data extractor
4) Sentiment analysis model`
  }

  // Handle "build another" from any stage
  const lower = userText.toLowerCase()
  if ((lower.includes('build another') || lower.includes('back to overview') || lower === 'start over') && session.stage !== 'clarify_1') {
    session.flow = null
    session.stage = null
    session.context = {}
    return `Great work on the ${FLOWS[session.flow]?.name || 'last project'}! Ready for the next one.

**What would you like to build?**

1) Support ticket classifier
2) Custom chatbot for my business
3) Invoice data extractor
4) Sentiment analysis model`
  }

  // Route to the current flow's stage handler
  const flow = FLOWS[session.flow]
  if (!flow) {
    session.flow = null
    session.stage = null
    return generateResponse(session, userText)
  }

  const handler = flow.stages[session.stage]
  if (!handler) {
    // Unknown stage — reset
    session.flow = null
    session.stage = null
    session.context = {}
    return generateResponse(session, userText)
  }

  const result = handler(userText, session.context)

  // Handle reset result
  if (result.nextStage === 'reset') {
    session.flow = null
    session.stage = null
    session.context = {}
    return `Ready for the next one!

**What would you like to build?**

1) Support ticket classifier
2) Custom chatbot for my business
3) Invoice data extractor
4) Sentiment analysis model`
  }

  // Advance stage
  session.stage = result.nextStage

  // If text is null (shouldn't happen normally but safety), provide fallback
  if (!result.text) {
    return generateResponse(session, userText)
  }

  return result.text
}

// ==================== Agent/Chat Endpoints ====================

app.post('/agent/session', (req, res) => {
  const sessionId = `session-${sessionCounter++}`
  sessions.set(sessionId, {
    id: sessionId,
    messages: [],
    flow: null,
    stage: null,
    context: {},
    created_at: new Date().toISOString()
  })
  console.log(`✅ Created session: ${sessionId}`)
  res.json({ id: sessionId })
})

app.post('/agent/session/:id/message', (req, res) => {
  const { id } = req.params
  const { parts } = req.body
  const userText = parts?.[0]?.text || ''

  const session = sessions.get(id)
  if (!session) {
    return res.status(404).json({ code: 'session_not_found', message: 'Session not found' })
  }

  console.log(`💬 [${id}] flow=${session.flow || 'none'} stage=${session.stage || 'none'} → "${userText.substring(0, 60)}"`)

  session.messages.push({ role: 'user', content: userText, timestamp: new Date().toISOString() })

  const responseText = generateResponse(session, userText)

  session.messages.push({ role: 'assistant', content: responseText, timestamp: new Date().toISOString() })

  console.log(`🤖 [${id}] flow=${session.flow || 'none'} stage=${session.stage || 'none'}`)

  res.json({
    parts: [{ type: 'text', text: responseText }],
    info: { providerID: 'amortized', modelID: 'morty-agent-v1' }
  })
})

// ==================== Other API Endpoints ====================

app.get('/api/v1/health', (req, res) => {
  res.json({
    status: 'ok',
    version: '3.0.0',
    timestamp: new Date().toISOString(),
    gpu: { available: false, count: 0, devices: [] }
  })
})

app.get('/api/v1/jobs', (req, res) => res.json([]))
app.get('/api/v1/jobs/:id', (req, res) => res.status(404).json({ code: 'not_found', message: 'Job not found' }))
app.get('/api/v1/recipes', (req, res) => res.json([
  { name: 'sft-llama-3.2', type: 'training', description: 'Supervised fine-tuning with Llama 3.2', category: 'training' }
]))

// MLflow mock endpoints (so datasets/models pages load without MLflow running)
app.post('/mlflow/api/2.0/mlflow/experiments/search', (req, res) => {
  res.json({ experiments: [] })
})
app.post('/mlflow/api/2.0/mlflow/runs/search', (req, res) => {
  res.json({ runs: [] })
})
app.get('/mlflow/api/2.0/mlflow/runs/get', (req, res) => {
  res.status(404).json({ error_code: 'RESOURCE_DOES_NOT_EXIST', message: 'Run not found' })
})
app.get('/mlflow/api/2.0/mlflow/registered-models/search', (req, res) => {
  res.json({ registered_models: [] })
})
app.get('/mlflow/api/2.0/mlflow/model-versions/search', (req, res) => {
  res.json({ model_versions: [] })
})
app.get('/mlflow/api/2.0/mlflow/metrics/get-history', (req, res) => {
  res.json({ metrics: [] })
})

// ==================== Start Server ====================

const PORT = 8001

app.listen(PORT, '127.0.0.1', () => {
  console.log(`
  ╔═════════════════════════════════════════════════╗
  ║  🤖 Morty Agent Backend                        ║
  ║  http://localhost:${PORT}                         ║
  ║                                                 ║
  ║  Interactive workflows:                         ║
  ║    • Support Ticket Classifier                  ║
  ║    • Custom Chatbot                             ║
  ║    • Invoice Data Extractor                     ║
  ║    • Sentiment Analysis                         ║
  ║                                                 ║
  ║  Each flow: clarify → plan → execute → results  ║
  ╚═════════════════════════════════════════════════╝
`)
})

process.on('SIGINT', () => {
  console.log('\n👋 Shutting down...')
  process.exit(0)
})
