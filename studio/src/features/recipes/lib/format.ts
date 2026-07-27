export function formatRecipeType(type: string): string {
  const map: Record<string, string> = {
    training: "Training",
    sdg: "SDG",
    eval: "Eval",
  }
  return map[type.toLowerCase()] ?? type
}

export function formatRecipeName(name: string): string {
  const basename = name.split("/").pop() ?? name
  return basename
    .replace(/[-_]/g, " ")
    .replace(/\.[^.]+$/, "")
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

export function inferRecipeType(name: string): string | null {
  if (name.includes("/eval/")) return "eval"
  if (name.includes("/sdg/")) return "sdg"
  if (name.includes("/training/") || name.includes("/train/")) return "training"
  return null
}

export function getEffectiveType(recipe: { name: string; type: string }): string {
  const raw = recipe.type?.trim()
  if (raw && raw !== "—") return raw
  return inferRecipeType(recipe.name) ?? ""
}

export function isUsefulRecipe(recipe: { name: string; type: string; description: string }): boolean {
  const hasDescription = !!recipe.description?.trim()
  const hasType = !!getEffectiveType(recipe)
  return hasDescription || hasType
}

export function recipeTypeVariant(
  type: string,
): "default" | "secondary" | "outline" {
  switch (type.toLowerCase()) {
    case "training":
      return "default"
    case "sdg":
      return "secondary"
    default:
      return "outline"
  }
}

export function recipeTypeClassName(type: string): string {
  switch (type.toLowerCase()) {
    case "sdg":
      return "border-transparent bg-[#ece6ff] text-[#5e40be] dark:bg-[#1b0d33]/40 dark:text-[#876fd4]"
    case "training":
      return "border-transparent bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]"
    case "eval":
      return "border-transparent bg-[#daf2f2] text-[#147878] dark:bg-[#003333]/40 dark:text-[#37a3a3]"
    default:
      return ""
  }
}
