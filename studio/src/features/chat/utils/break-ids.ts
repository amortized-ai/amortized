import type { ElementContent, Root, RootContent } from "hast"

// A long id such as `bigbench_code_line_description_multiple_choice` has no space to wrap at,
// so a table holding it grows past the chat bubble. This rehype plugin adds a line break
// opportunity (<wbr>) after each _ : . / - inside table cells. Ids then wrap at their own
// separators, and ordinary words like "Questions" are never cut.
const SEPARATOR = /(?<=[_:./-])(?=.)/

function breakText(nodes: ElementContent[]): ElementContent[] {
  return nodes.flatMap((node): ElementContent[] => {
    if (node.type === "element") return [{ ...node, children: breakText(node.children) }]
    if (node.type !== "text") return [node]
    return node.value.split(SEPARATOR).flatMap((part, i): ElementContent[] => [
      ...(i > 0 ? [{ type: "element" as const, tagName: "wbr", properties: {}, children: [] }] : []),
      { type: "text", value: part },
    ])
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
