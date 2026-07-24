import type { ChatMessage, OptionCard } from "../types"
import type { MlflowGatewayRoute } from "@/types/api"

export type WorkflowStep =
  | "sdg-domain"
  | "sdg-categories"
  | "sdg-urgency"
  | "sdg-samples"
  | "sdg-teacher-model"
  | "sdg-confirm"
  | "training-student-model"
  | "training-method"
  | "training-confirm"
  | "eval-method"
  | "eval-judge-model"
  | "eval-confirm"
  | null

function card(title: string, description: string, value?: string): OptionCard {
  return { title, description, value: value ?? `${title} — ${description}` }
}

const DOMAIN_CARDS: OptionCard[] = [
  card("Software/technical support", "Bug reports, feature requests, troubleshooting"),
  card("Billing & payments", "Invoices, refunds, subscription issues"),
  card("Customer service", "Account access, onboarding, general inquiries"),
  card("E-commerce", "Orders, shipping, returns, product questions"),
]

const CATEGORY_MAP: Record<string, OptionCard[]> = {
  software: [
    card("Bug reports & crashes", "Application errors, unexpected behavior, system failures"),
    card("Feature requests", "New functionality, enhancement suggestions, integrations"),
    card("Setup & configuration", "Installation, onboarding, environment setup"),
    card("All of the above", "Cover all technical support categories"),
  ],
  billing: [
    card("Invoice & payment issues", "Failed payments, missing invoices, overcharges"),
    card("Refunds & disputes", "Refund requests, chargebacks, billing errors"),
    card("Subscription management", "Plan changes, cancellations, renewals"),
    card("All of the above", "Cover all billing sub-categories"),
  ],
  customer: [
    card("Account access", "Login issues, password resets, account recovery"),
    card("Onboarding & setup", "Getting started, configuration, first-time help"),
    card("General inquiries", "Questions, feedback, complaints"),
    card("All of the above", "Cover all customer service categories"),
  ],
  ecommerce: [
    card("Orders & shipping", "Order status, tracking, delivery, lost packages"),
    card("Returns & refunds", "Return requests, refund status, exchanges"),
    card("Product questions", "Sizing, availability, compatibility, details"),
    card("All of the above", "Cover all e-commerce categories"),
  ],
}

const URGENCY_CARDS: OptionCard[] = [
  card("Yes, 3 levels", "Low, Medium, High"),
  card("Yes, 4 levels", "Low, Medium, High, Critical"),
  card("No, just categories", "Only classify by topic"),
]

const SAMPLE_CARDS: OptionCard[] = [
  card("100 samples", "Quick test run"),
  card("500 samples", "Good for most use cases"),
  card("1000 samples", "Higher quality, takes longer"),
]

const FALLBACK_TEACHER_MODEL_CARDS: OptionCard[] = [
  card("Claude Haiku", "Fast and affordable"),
  card("GPT-4o", "Strong reasoning ability"),
  card("Claude Sonnet", "Higher quality output"),
]

const FRIENDLY_MODEL_NAMES: Record<string, string> = {
  "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
  "claude-haiku-4-5@20251001": "Claude Haiku 4.5",
  "claude-sonnet-4-20250514": "Claude Sonnet 4",
  "claude-sonnet-4@20250514": "Claude Sonnet 4",
  "gpt-4o": "GPT-4o",
  "gpt-4o-mini": "GPT-4o Mini",
  "gpt-4o-mini-2024-07-18": "GPT-4o Mini",
}

export function friendlyModelLabel(route: MlflowGatewayRoute): string {
  const friendly = FRIENDLY_MODEL_NAMES[route.model.name]
  if (friendly) return friendly
  return route.name
    .replace(/-/g, " ")
    .replace(/\b\w/g, c => c.toUpperCase())
}

export function routesToModelCards(routes: MlflowGatewayRoute[]): OptionCard[] {
  return routes.map(r => card(
    friendlyModelLabel(r),
    `${r.model.provider} · ${r.model.name}`,
    `${r.name} — ${r.model.provider}/${r.model.name}`,
  ))
}

function getTeacherModelCards(routes?: MlflowGatewayRoute[]): OptionCard[] {
  if (routes && routes.length > 0) return routesToModelCards(routes)
  return FALLBACK_TEACHER_MODEL_CARDS
}

function getJudgeModelCards(routes?: MlflowGatewayRoute[]): OptionCard[] {
  if (routes && routes.length > 0) return routesToModelCards(routes)
  return FALLBACK_JUDGE_MODEL_CARDS
}

const CONFIRM_CARDS: OptionCard[] = [
  card("Yes, submit the job", "Start generating with the settings above", "Yes, submit the job"),
  card("Change something", "Adjust the settings before submitting", "Change something"),
]

const STUDENT_MODEL_CARDS: OptionCard[] = [
  card("Qwen3 0.6B", "Fastest, great for prototyping (~0.6B params)"),
  card("Qwen 2.5 1.5B", "Good default for production (~1.5B params)"),
  card("Qwen3 4B", "Best quality for complex tasks (~4B params)"),
  card("Llama 3.1 8B", "Largest, needs QLoRA (~8B params)"),
]

const TRAINING_METHOD_CARDS: OptionCard[] = [
  card("LoRA SFT", "Recommended — fastest and cheapest"),
  card("QLoRA SFT", "Lower memory footprint"),
  card("Full SFT", "Best quality, updates all weights"),
]

const EVAL_METHOD_CARDS: OptionCard[] = [
  card("Classification accuracy", "Precision, recall, F1 via deterministic field matching"),
  card("Response quality", "LLM judge assesses correctness and reasoning"),
  card("Both", "Classification accuracy plus LLM judge quality check"),
]

const FALLBACK_JUDGE_MODEL_CARDS: OptionCard[] = [
  card("GPT-4o Mini", "Fast and cheap, recommended default"),
  card("Claude Haiku", "Balanced cost and quality"),
  card("GPT-4o", "Higher quality judgments"),
  card("Claude Sonnet", "Highest quality, most expensive"),
]

function detectDomain(messages: ChatMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const lower = messages[i]!.content.toLowerCase()
    if (/software|technical|bug report|feature request|troubleshoot/i.test(lower)) return "software"
    if (/billing|payment|invoice|refund|subscription/i.test(lower)) return "billing"
    if (/customer service|account access|onboarding|general inquir/i.test(lower)) return "customer"
    if (/e-?commerce|orders|shipping|returns|product/i.test(lower)) return "ecommerce"
  }
  return null
}

export function detectWorkflowStep(
  content: string,
  messages?: ChatMessage[],
  gatewayRoutes?: MlflowGatewayRoute[],
): { step: WorkflowStep; cards: OptionCard[] } | null {
  // SDG flow
  if (/what.*(?:type|kind).*(?:support|ticket|customer)|what.*(?:kind|type).*handle|what.*customer.*support/i.test(content))
    return { step: "sdg-domain", cards: DOMAIN_CARDS }

  if (/what.*(?:specific|billing|support).*categor|(?:specific|billing).*categor|sub-?categor/i.test(content)) {
    const domain = messages ? detectDomain(messages) : null
    const cards = (domain && CATEGORY_MAP[domain]) || CATEGORY_MAP.ecommerce!
    return { step: "sdg-categories", cards }
  }

  if (/(?:should.*assign|also.*assign|want.*urgency|assign.*urgency|urgency.*level.*\?)/i.test(content))
    return { step: "sdg-urgency", cards: URGENCY_CARDS }

  if (/how many.*sample/i.test(content))
    return { step: "sdg-samples", cards: SAMPLE_CARDS }

  if (/which teacher model|pick.*teacher model|choose.*teacher model/i.test(content))
    return { step: "sdg-teacher-model", cards: getTeacherModelCards(gatewayRoutes) }

  // Training flow
  if (/which student model|would you like to fine-?tune|choose a student model/i.test(content))
    return { step: "training-student-model", cards: STUDENT_MODEL_CARDS }

  if (/training method option|which training method|breakdown for|LoRA SFT.*QLoRA|QLoRA.*Full SFT|LoRA SFT.*Full SFT/i.test(content))
    return { step: "training-method", cards: TRAINING_METHOD_CARDS }

  // Eval flow
  if (/what.*(?:would you like to )?evaluat|how.*evaluat|which.*evaluation method|evaluation method/i.test(content))
    return { step: "eval-method", cards: EVAL_METHOD_CARDS }

  if (/which judge model|which model.*judge|judge.*should.*evaluat/i.test(content))
    return { step: "eval-judge-model", cards: getJudgeModelCards(gatewayRoutes) }

  // Confirmation steps (must be checked AFTER specific steps)
  if (/ready to start training|ready to train/i.test(content))
    return { step: "training-confirm", cards: CONFIRM_CARDS }

  if (/ready to run eval|ready to evaluate/i.test(content))
    return { step: "eval-confirm", cards: CONFIRM_CARDS }

  if (/ready to (?:generate|go|run|start)|here'?s (?:the|your) plan/i.test(content))
    return { step: "sdg-confirm", cards: CONFIRM_CARDS }

  return null
}

export type CostStep =
  | "sdg-teacher-model"
  | "sdg-confirm"
  | "training-student-model"
  | "training-method"
  | "training-confirm"
  | "eval-judge-model"
  | "eval-confirm"
  | null

export function workflowStepToCostStep(step: WorkflowStep): CostStep {
  switch (step) {
    case "sdg-teacher-model":
    case "sdg-confirm":
    case "training-student-model":
    case "training-method":
    case "training-confirm":
    case "eval-judge-model":
    case "eval-confirm":
      return step
    default:
      return null
  }
}

// --- State machine: determine next step and inject instruction ---

const FULL_FLOW: WorkflowStep[] = [
  "sdg-domain", "sdg-categories", "sdg-urgency", "sdg-samples",
  "sdg-teacher-model", "sdg-confirm",
  "training-student-model", "training-method", "training-confirm",
  "eval-method", "eval-judge-model", "eval-confirm",
]

const PHASE_BOUNDARIES = new Set<WorkflowStep>(["sdg-confirm", "training-confirm", "eval-confirm"])

function buildStepMatchers(routes?: MlflowGatewayRoute[]): Record<string, (msg: string) => boolean> {
  const routeNames = (routes ?? []).map(r => r.name.toLowerCase())
  const modelMatcher = routeNames.length > 0
    ? (m: string) => routeNames.some(n => m.includes(n))
    : (m: string) => /claude haiku|gpt-4o|claude sonnet/i.test(m)
  const judgeMatcher = routeNames.length > 0
    ? (m: string) => routeNames.some(n => m.includes(n))
    : (m: string) => /gpt-4o mini|claude haiku|gpt-4o|claude sonnet/i.test(m)
  return {
    "sdg-domain": (m) => DOMAIN_CARDS.some((c) => m.includes(c.title.toLowerCase())),
    "sdg-categories": (m) => /invoice|refund|subscription|bug report|feature request|setup|order|return|product|account|onboarding|all of the above/i.test(m),
    "sdg-urgency": (m) => /yes, [34] level|no, just categor/i.test(m),
    "sdg-samples": (m) => /\b\d+\s*samples?\b/i.test(m) || /^\d+$/.test(m.trim()),
    "sdg-teacher-model": modelMatcher,
    "sdg-confirm": (m) => /yes.*submit|change something/i.test(m),
    "training-student-model": (m) => /qwen|llama/i.test(m),
    "training-method": (m) => /lora sft|qlora|full sft/i.test(m),
    "training-confirm": (m) => /yes.*submit|change something/i.test(m),
    "eval-method": (m) => /classification accuracy|response quality|both/i.test(m),
    "eval-judge-model": judgeMatcher,
    "eval-confirm": (m) => /yes.*submit|change something/i.test(m),
  }
}

const STEP_INSTRUCTIONS: Partial<Record<NonNullable<WorkflowStep>, string>> = {
  "sdg-domain": `[SYSTEM INSTRUCTION — follow exactly]
Ask what type of support tickets this classifier will handle.
Present EXACTLY these 4 options as a numbered list:
1) Software/technical support — Bug reports, feature requests, troubleshooting
2) Billing & payments — Invoices, refunds, subscription issues
3) Customer service — Account access, onboarding, general inquiries
4) E-commerce — Orders, shipping, returns, product questions
Say ONE short sentence before the options. Ask ONLY this question. Do NOT combine with any other question.`,

  "sdg-categories": `[SYSTEM INSTRUCTION — follow exactly]
Based on the user's domain choice, ask what specific sub-categories the classifier should use.
Present 3-4 relevant categories PLUS "All of the above" as a numbered list.
Say ONE short sentence before the options. Ask ONLY about categories. Do NOT ask about urgency, samples, or anything else in this message.`,

  "sdg-urgency": `[SYSTEM INSTRUCTION — follow exactly]
Ask if the classifier should also assign urgency levels to each ticket.
Present EXACTLY these 3 options as a numbered list:
1) Yes, 3 levels — Low, Medium, High
2) Yes, 4 levels — Low, Medium, High, Critical
3) No, just categories — Only classify by topic
Say ONE short sentence before the options. Ask ONLY this question. Do NOT combine with sample count or any other question.`,

  "sdg-samples": `[SYSTEM INSTRUCTION — follow exactly]
Ask how many training samples to generate.
Present EXACTLY these 3 options as a numbered list:
1) 100 samples — Quick test run
2) 500 samples — Good for most use cases
3) 1000 samples — Higher quality, takes longer
Say ONE short sentence before the options. Ask ONLY this question.`,

  "sdg-teacher-model": `[SYSTEM INSTRUCTION — follow exactly]
Ask which teacher model should generate the training data.
Present EXACTLY these 3 options as a numbered list:
1) Claude Haiku — Fast and affordable
2) GPT-4o — Strong reasoning ability
3) Claude Sonnet — Higher quality output
Say ONE short sentence before the options. Ask ONLY this question.`,

  "sdg-confirm": `[SYSTEM INSTRUCTION — follow exactly]
Show a confirmation summary using a markdown TABLE with one setting per row:

| Setting | Value |
|---------|-------|
| Recipe | (auto-selected recipe) |
| Domain | (selected domain) |
| Categories | (selected sub-categories) |
| Urgency Levels | (selected levels) |
| Teacher Model | (selected model) |
| Samples | (count) |

Do NOT include an "Est. Cost" row — cost is shown separately by the system.
Each setting MUST be on its own row. Do NOT put multiple settings on one line.
End with: Ready to generate? (yes / change something)
After job submission, show ONLY a brief summary (type, teacher model, samples, labels, job ID). Do NOT add "What would you like to do next?" or any numbered options — the system shows clickable action cards automatically.`,

  "training-student-model": `[SYSTEM INSTRUCTION — follow exactly]
Ask which student model to fine-tune.
Present EXACTLY these 4 options as a numbered list:
1) Qwen3 0.6B — Fastest, great for prototyping (~0.6B params)
2) Qwen 2.5 1.5B — Good default for production (~1.5B params)
3) Qwen3 4B — Best quality for complex tasks (~4B params)
4) Llama 3.1 8B — Largest, needs QLoRA (~8B params)
Ask ONLY this question.`,

  "training-method": `[SYSTEM INSTRUCTION — follow exactly]
Present training method options for the selected model.
Present EXACTLY these 3 options as a numbered list:
1) LoRA SFT — Recommended, fastest and cheapest
2) QLoRA SFT — Lower memory footprint
3) Full SFT — Best quality, updates all weights
Ask ONLY this question.`,

  "training-confirm": `[SYSTEM INSTRUCTION — follow exactly]
Show a confirmation table with training settings (student model, method, training data source/parent job, estimated time).
Do NOT include an "Est. Cost" row — cost is shown separately by the system.
End with: Ready to start training? (yes / change something)
After job submission, show ONLY a brief summary (model, method, epochs, job ID). Do NOT add "What would you like to do next?" or any numbered options — the system shows clickable action cards automatically.`,

  "eval-method": `[SYSTEM INSTRUCTION — follow exactly]
Ask what evaluation method(s) to use.
Present EXACTLY these 3 options as a numbered list:
1) Classification accuracy — Precision, recall, F1 via deterministic field matching
2) Response quality — LLM judge assesses correctness and reasoning
3) Both — Classification accuracy plus LLM judge quality check
Ask ONLY this question.`,

  "eval-judge-model": `[SYSTEM INSTRUCTION — follow exactly]
Ask which judge model to use for evaluation.
Present EXACTLY these 4 options as a numbered list:
1) GPT-4o Mini — Fast and cheap, recommended default
2) Claude Haiku — Balanced cost and quality
3) GPT-4o — Higher quality judgments
4) Claude Sonnet — Highest quality, most expensive
Ask ONLY this question.`,

  "eval-confirm": `[SYSTEM INSTRUCTION — follow exactly]
Show a confirmation table with eval settings (eval method, judge model, samples, parent job).
Do NOT include an "Est. Cost" row — cost is shown separately by the system.
End with: Ready to run eval? (yes / change something)
After job submission, show ONLY a brief summary (eval method, judge model, samples, job ID). Do NOT add "What would you like to do next?" or any numbered options — the system shows clickable action cards automatically.`,
}

function isWorkflowTrigger(content: string): boolean {
  return /build.*classifier|train.*model|classify.*ticket|create.*classifier|support ticket/i.test(content)
}

function detectTransitionReady(messages: ChatMessage[], fromStep: WorkflowStep): boolean {
  const assistantMsgs = messages.filter(m => m.role === "assistant")
  const lastAssistant = assistantMsgs[assistantMsgs.length - 1]
  const userMsgs = messages.filter(m => m.role === "user")
  const lastUser = userMsgs[userMsgs.length - 1]

  if (fromStep === "sdg-confirm") {
    const assistantReady = lastAssistant && /fine-?tun|student model|training|which.*model.*train|would you like to train|train a model/i.test(lastAssistant.content)
    const userReady = lastUser && /continue.*training|training step|train/i.test(lastUser.content)
    return !!(assistantReady || userReady)
  }
  if (fromStep === "training-confirm") {
    const assistantReady = lastAssistant && /evaluat|eval|judge|assess|how.*(?:well|good)|measure.*performance|test.*model/i.test(lastAssistant.content)
    const userReady = lastUser && /continue.*eval|evaluation step|evaluat/i.test(lastUser.content)
    return !!(assistantReady || userReady)
  }
  return false
}

function shouldSkipJudgeModel(messages: ChatMessage[]): boolean {
  const userMsgs = messages.filter(m => m.role === "user")
  for (let i = userMsgs.length - 1; i >= 0; i--) {
    const text = userMsgs[i]!.content.toLowerCase()
    if (/classification accuracy/i.test(text) && !/both/i.test(text)) return true
    if (/response quality|both/i.test(text)) return false
  }
  return false
}

function buildDynamicInstruction(step: NonNullable<WorkflowStep>, routes?: MlflowGatewayRoute[]): string | undefined {
  if (step === "sdg-teacher-model" && routes && routes.length > 0) {
    const list = routes.map((r, i) => `${i + 1}) ${r.name} — ${r.model.provider} / ${r.model.name}`).join("\n")
    return `[SYSTEM INSTRUCTION — follow exactly]
Ask which teacher model should generate the training data.
Present EXACTLY these options as a numbered list:
${list}
Say ONE short sentence before the options. Ask ONLY this question.`
  }
  if (step === "eval-judge-model" && routes && routes.length > 0) {
    const list = routes.map((r, i) => `${i + 1}) ${r.name} — ${r.model.provider} / ${r.model.name}`).join("\n")
    return `[SYSTEM INSTRUCTION — follow exactly]
Ask which judge model to use for evaluation.
Present EXACTLY these options as a numbered list:
${list}
Say ONE short sentence before the options. Ask ONLY this question.`
  }
  return STEP_INSTRUCTIONS[step]
}

export function getNextStepInstruction(
  messages: ChatMessage[],
  gatewayRoutes?: MlflowGatewayRoute[],
): { instruction: string; nextStep: WorkflowStep } | null {
  if (messages.length === 0) return null

  const lastUserMsg = messages[messages.length - 1]
  if (!lastUserMsg || lastUserMsg.role !== "user") return null
  const userText = lastUserMsg.content.toLowerCase()

  const userMessages = messages.filter((m) => m.role === "user")
  const matchers = buildStepMatchers(gatewayRoutes)

  // First message triggers the workflow
  if (userMessages.length === 1 && isWorkflowTrigger(userText)) {
    const inst = STEP_INSTRUCTIONS["sdg-domain"]
    return inst ? { instruction: inst, nextStep: "sdg-domain" } : null
  }

  // Walk through all completed steps across all phases
  let currentIdx = -1
  for (const um of userMessages) {
    const text = um.content.toLowerCase()
    const nextStep = FULL_FLOW[currentIdx + 1]
    if (nextStep && matchers[nextStep]?.(text)) {
      currentIdx++
    }
  }

  const currentStep = currentIdx >= 0 ? FULL_FLOW[currentIdx]! : null

  // At a phase boundary (confirm step): wait for natural transition
  if (currentStep && PHASE_BOUNDARIES.has(currentStep)) {
    const nextIdx = currentIdx + 1
    if (nextIdx >= FULL_FLOW.length) return null

    if (!detectTransitionReady(messages, currentStep)) {
      return null
    }

    let nextStep = FULL_FLOW[nextIdx]!
    // Skip judge model if user chose classification-only
    if (nextStep === "eval-judge-model" && shouldSkipJudgeModel(messages)) {
      const skipIdx = nextIdx + 1
      if (skipIdx >= FULL_FLOW.length) return null
      nextStep = FULL_FLOW[skipIdx]!
    }
    const inst = buildDynamicInstruction(nextStep, gatewayRoutes)
    return inst ? { instruction: inst, nextStep } : null
  }

  // Normal flow: advance to next step
  const nextIdx = currentIdx + 1
  if (nextIdx >= FULL_FLOW.length) return null

  let nextStep = FULL_FLOW[nextIdx]!
  // Skip judge model if user chose classification-only
  if (nextStep === "eval-judge-model" && shouldSkipJudgeModel(messages)) {
    const skipIdx = nextIdx + 1
    if (skipIdx >= FULL_FLOW.length) return null
    nextStep = FULL_FLOW[skipIdx]!
  }
  const inst = buildDynamicInstruction(nextStep, gatewayRoutes)
  return inst ? { instruction: inst, nextStep } : null
}

const PHASE_TAG_RE = /<phase>([\w:_-]+)<\/phase>/

export function extractPhase(text: string): { cleanText: string; phase: string | null } {
  const match = PHASE_TAG_RE.exec(text)
  if (!match) return { cleanText: text, phase: null }
  return {
    cleanText: text.replace(PHASE_TAG_RE, "").replace(/\n{2,}$/, "\n").trim(),
    phase: match[1]!,
  }
}

export function parsePhaseTag(phase: string): { phase: WorkflowPhase; step: string } | null {
  const parts = phase.split(":")
  if (parts.length !== 2) return null
  const [p, step] = parts as [string, string]
  if (p !== "sdg" && p !== "training" && p !== "eval") return null
  return { phase: p, step }
}

export function getLatestPhase(messages: ChatMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const phase = messages[i]!.phase
    if (phase) return phase
  }
  return null
}

export function friendlyModelName(nameOrId: string): string {
  const exact = FRIENDLY_MODEL_NAMES[nameOrId]
  if (exact) return exact
  const withoutProvider = nameOrId.replace(/^(openai|anthropic|vertex_ai)\//, "")
  const fromStripped = FRIENDLY_MODEL_NAMES[withoutProvider]
  if (fromStripped) return fromStripped
  if (/^[A-Z]/.test(nameOrId) && !nameOrId.includes("/")) return nameOrId
  return withoutProvider
    .replace(/-/g, " ")
    .replace(/\b\w/g, c => c.toUpperCase())
}

export type WorkflowPhase = "sdg" | "training" | "eval"

export function getCurrentWorkflowPhase(messages: ChatMessage[]): { phase: WorkflowPhase; stepIndex: number } | null {
  const userMessages = messages.filter(m => m.role === "user")
  const matchers = buildStepMatchers()
  let currentIdx = -1

  for (const um of userMessages) {
    const text = um.content.toLowerCase()
    const nextStep = FULL_FLOW[currentIdx + 1]
    if (nextStep && matchers[nextStep]?.(text)) {
      currentIdx++
    }
  }

  if (currentIdx < 0) return null

  const step = FULL_FLOW[currentIdx]!
  if (step.startsWith("eval-")) return { phase: "eval", stepIndex: currentIdx - 9 }
  if (step.startsWith("training-")) return { phase: "training", stepIndex: currentIdx - 6 }
  return { phase: "sdg", stepIndex: currentIdx }
}
