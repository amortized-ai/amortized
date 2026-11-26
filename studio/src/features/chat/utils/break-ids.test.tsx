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

  it("leaves numbers and dates whole", () => {
    const container = renderMarkdown(
      "| Metric | Score | Delta | Date |\n|---|---|---|---|\n| acc_norm | 0.967 | -0.25 | 2026-11-18 |\n| f1 | 1e-4 | 12.5% | 2026-11-18T09:30:00Z |",
    )
    const cells = Array.from(container.querySelectorAll("td"))

    expect(cells[0]?.innerHTML).toBe("acc_<wbr>norm")
    expect(cells.slice(1).map((cell) => cell.innerHTML)).toEqual([
      "0.967",
      "-0.25",
      "2026-11-18",
      "f1",
      "1e-4",
      "12.5%",
      "2026-11-18T09:30:00Z",
    ])
  })

  it("breaks each id in a cell that also holds words and numbers", () => {
    const container = renderMarkdown(
      "| Note |\n|---|\n| ran arc_easy on 2026-11-18, score 0.4 |\n| f03e4df5-91d7-4e1d |",
    )
    const cells = Array.from(container.querySelectorAll("td"))

    expect(cells[0]?.innerHTML).toBe("ran arc_<wbr>easy on 2026-11-18, score 0.4")
    expect(cells[1]?.innerHTML).toBe("f03e4df5-<wbr>91d7-<wbr>4e1d")
  })

  it("breaks a UUID that has no two letters in a row", () => {
    const container = renderMarkdown("| Job |\n|---|\n| 123e4567-e89b-12d3-a456-426614174000 |")

    expect(container.querySelector("td")?.innerHTML).toBe(
      "123e4567-<wbr>e89b-<wbr>12d3-<wbr>a456-<wbr>426614174000",
    )
  })

  it("leaves text outside tables alone", () => {
    const container = renderMarkdown("Pick arc_easy or `arc:easy`.")

    expect(container.querySelector("wbr")).toBeNull()
  })
})
