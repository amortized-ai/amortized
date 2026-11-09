import { render } from "@testing-library/react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { describe, expect, it } from "vitest"
import { rehypeBreakIds } from "./break-ids"

function renderMarkdown(markdown: string) {
  return render(
    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeBreakIds]}>
      {markdown}
    </ReactMarkdown>,
  ).container
}

describe("rehypeBreakIds", () => {
  it("lets ids in table cells wrap after their separators only", () => {
    const container = renderMarkdown(
      "| ID | Questions |\n|---|---|\n| bigbench_code_line | 1000 |\n| `lcb:codegeneration_v6` | 10 |",
    )
    const cells = Array.from(container.querySelectorAll("td"))

    expect(cells[0]?.innerHTML).toBe("bigbench_<wbr>code_<wbr>line")
    expect(cells[0]?.textContent).toBe("bigbench_code_line")
    expect(cells[2]?.innerHTML).toBe("<code>lcb:<wbr>codegeneration_<wbr>v6</code>")
    expect(container.querySelector("th")?.innerHTML).toBe("ID")
  })

  it("leaves text outside tables alone", () => {
    const container = renderMarkdown("Pick arc_easy or `arc:easy`.")

    expect(container.querySelector("wbr")).toBeNull()
  })
})
