export interface TutorialStep {
  id: string
  route?: string
  target?: string
  title: string
  description: string
  placement?: "top" | "right" | "bottom" | "left"
}

export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    id: "welcome",
    title: "Welcome to Amortized Studio",
    description:
      "Build task-specific AI models that replace expensive API calls. This quick tour shows you around — it takes about a minute.",
  },
  {
    id: "sidebar",
    target: "sidebar-nav",
    title: "Navigation",
    description:
      "Everything lives in these sections: Chat with Morty, monitor Jobs, browse Datasets and Models, configure Recipes, and manage Settings.",
    placement: "right",
  },
  {
    id: "settings",
    route: "/settings",
    target: "settings-nav",
    title: "Check Your Connections",
    description:
      "Three services need to be running: the Backend API, MLflow for artifact tracking, and an AI Gateway for LLM access.",
    placement: "bottom",
  },
  {
    id: "documents",
    route: "/documents",
    target: "documents-header",
    title: "Upload Documents",
    description:
      "Upload PDFs and DOCX files to use as grounding for synthetic data generation. Parsed content feeds directly into the SDG pipeline.",
    placement: "bottom",
  },
  {
    id: "chat",
    route: "/chat",
    target: "chat-input",
    title: "Chat with Morty",
    description:
      "Describe your task in plain language. Morty handles the full pipeline — data generation, training, and evaluation — so you don't have to.",
    placement: "top",
  },
  {
    id: "recipes",
    route: "/recipes",
    target: "recipe-browser",
    title: "Recipes",
    description:
      "Pre-built pipelines for each stage — data generation, training, and evaluation. Follow the 5 steps on this page, or let Morty guide you in the chat.",
    placement: "left",
  },
  {
    id: "jobs",
    route: "/jobs",
    target: "job-header",
    title: "Monitor Jobs",
    description:
      "Track every job in real time — view logs, training metrics, and status updates as they run on your infrastructure.",
    placement: "bottom",
  },
  {
    id: "datasets",
    route: "/datasets",
    target: "datasets-header",
    title: "Browse Datasets",
    description:
      "Inspect your generated training data — preview samples, check row counts, and link datasets to the SDG jobs that created them.",
    placement: "bottom",
  },
  {
    id: "models",
    route: "/models",
    target: "models-header",
    title: "Browse Models",
    description:
      "View your fine-tuned models, track versions in MLflow, and check which models are ready to deploy.",
    placement: "bottom",
  },
  {
    id: "complete",
    route: "/",
    title: "You're Ready!",
    description:
      "You're all set! Start by chatting with Morty or picking a recipe. Your first custom model is a conversation away.",
  },
]
