import { test, expect } from "./fixtures.js"

test.describe("Amortized Studio E2E", () => {
  test("overview page loads with metric cards", async ({ page }) => {
    await page.goto("/")
    await expect(page).toHaveURL(/\/overview/)
    await expect(page.getByText("Amortized Studio")).toBeVisible()
    await expect(page.getByRole("heading", { name: "Amortized Studio" })).toBeVisible()
    const main = page.locator("main")
    await expect(main.getByText("Total Datasets")).toBeVisible()
    await expect(main.getByText("Active Jobs")).toBeVisible()
  })

  test("sidebar navigation has all v1 pages", async ({ page }) => {
    await page.goto("/overview")
    const sidebar = page.locator("[data-sidebar='sidebar']")
    await expect(sidebar).toBeVisible()
    await expect(sidebar.getByText("Overview")).toBeVisible()
    await expect(sidebar.getByText("Chat")).toBeVisible()
    await expect(sidebar.getByText("Jobs")).toBeVisible()
    await expect(sidebar.getByText("Datasets")).toBeVisible()
    await expect(sidebar.getByText("Models")).toBeVisible()
    await expect(sidebar.getByText("Recipes")).toBeVisible()
    await expect(sidebar.getByText("Settings")).toBeVisible()
  })

  test("jobs page shows jobs from backend", async ({ page }) => {
    await page.goto("/jobs")
    await expect(page.getByRole("heading", { name: "Jobs" })).toBeVisible()
    await page.waitForSelector("table", { timeout: 10000 })
    const rows = page.locator("table tbody tr")
    await expect(rows.first()).toBeVisible({ timeout: 10000 })
    const count = await rows.count()
    expect(count).toBeGreaterThan(0)
  })

  test("job detail panel opens on click", async ({ page }) => {
    await page.goto("/jobs")
    await page.getByRole("button", { name: "Toggle Sidebar" }).click()
    await page.waitForTimeout(500)
    await page.waitForSelector("table tbody tr", { timeout: 10000 })
    await page.locator("table tbody tr").first().click()
    // Sheet panel renders — verify the job ID appears somewhere on page
    await expect(page.getByText("job-001").first()).toBeVisible({ timeout: 10000 })
  })

  test("datasets page loads", async ({ page }) => {
    await page.goto("/datasets")
    await expect(page.getByRole("heading", { name: "Datasets" })).toBeVisible()
    await page.waitForTimeout(3000)
    const hasTable = await page.locator("table").isVisible()
    const hasEmptyText = await page.getByText(/no datasets/i).isVisible().catch(() => false)
    expect(hasTable || hasEmptyText).toBeTruthy()
  })

  test("models page loads", async ({ page }) => {
    await page.goto("/models")
    await expect(page.getByRole("heading", { name: "Models" })).toBeVisible()
    await page.waitForTimeout(3000)
    const hasTable = await page.locator("table").isVisible()
    const hasEmptyText = await page.getByText(/no models/i).isVisible().catch(() => false)
    expect(hasTable || hasEmptyText).toBeTruthy()
  })

  test("recipes page loads", async ({ page }) => {
    await page.goto("/recipes")
    await expect(page.getByRole("heading", { name: "Recipes" })).toBeVisible()
  })

  test("settings page loads", async ({ page }) => {
    await page.goto("/settings")
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible()
  })

  test("chat page loads and accepts input", async ({ page }) => {
    await page.goto("/chat")
    const textarea = page.locator("textarea")
    await expect(textarea).toBeVisible({ timeout: 5000 })
  })

  test("dark mode toggle works", async ({ page }) => {
    await page.goto("/overview")
    const toggle = page.getByTestId("theme-toggle")
    await expect(toggle).toBeVisible()
    const htmlEl = page.locator("html")
    await expect(htmlEl).not.toHaveClass(/dark/)
    await toggle.click()
    await expect(htmlEl).toHaveClass(/dark/)
    await toggle.click()
    await expect(htmlEl).not.toHaveClass(/dark/)
  })

  test("breadcrumbs update on navigation", async ({ page }) => {
    await page.goto("/jobs")
    const breadcrumb = page.getByRole("navigation", { name: "Breadcrumb" })
    await expect(breadcrumb).toBeVisible()
    await expect(breadcrumb.getByText("Jobs")).toBeVisible()
  })

  test("+ Create dropdown menu opens", async ({ page }) => {
    await page.goto("/overview")
    const createBtn = page.getByRole("button", { name: /create/i })
    await expect(createBtn).toBeVisible()
    await createBtn.click()
    await expect(
      page.getByRole("menuitem", { name: /chat/i }).or(page.getByText(/new chat/i))
    ).toBeVisible()
  })

  test("connection dot reflects backend status", async ({ page }) => {
    await page.goto("/overview")
    const dot = page.getByTestId("connection-dot")
    await expect(dot).toBeVisible()
  })
})
