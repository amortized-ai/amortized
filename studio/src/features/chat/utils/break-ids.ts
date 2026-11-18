import type { ElementContent, Root, RootContent } from "hast"

// A long id such as `bigbench_code_line_description_multiple_choice` has no space to wrap at,
// so a table holding it grows past the chat bubble. This rehype plugin adds a line break
// opportunity (<wbr>) after each _ : . / - of an id inside table cells. Ids then wrap at
// their own separators, and ordinary words like "Questions" are never cut. A token without
// two letters in a row is a number or a date (0.967, -0.25, 2026-11-18T09:30:00Z), so it
// stays whole.
const SEPARATOR = /(?<=[_:./-])(?=.)/
const ID = /\p{L}{2}/u

function breakIds(text: string): ElementContent[] {
  const nodes: ElementContent[] = []
  let pending = ""
  for (const token of text.split(/(\s+)/)) {
    const parts = ID.test(token) ? token.split(SEPARATOR) : [token]
    parts.forEach((part, i) => {
      if (i > 0) {
        nodes.push(
          { type: "text", value: pending },
          { type: "element", tagName: "wbr", properties: {}, children: [] },
        )
        pending = ""
      }
      pending += part
    })
  }
  return [...nodes, { type: "text", value: pending }]
}

function breakText(nodes: ElementContent[]): ElementContent[] {
  return nodes.flatMap((node): ElementContent[] => {
    if (node.type === "element") return [{ ...node, children: breakText(node.children) }]
    if (node.type !== "text") return [node]
    return breakIds(node.value)
  })
}

function visit(node: Root | RootContent): void {
  if (node.type !== "root" && node.type !== "element") return
  node.children.forEach((child, i) => {
    if (child.type === "element" && (child.tagName === "td" || child.tagName === "th")) {
      node.children[i] = { ...child, children: breakText(child.children) }
    } else {
      visit(child)
    }
  })
}

export function rehypeBreakIds() {
  return (tree: Root) => visit(tree)
}
