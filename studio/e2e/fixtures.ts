import { test as base, expect } from "@playwright/test"
import { type AnyHandler } from "msw"
import {
  defineNetworkFixture,
  type NetworkFixture,
} from "@msw/playwright"
import { handlers } from "../src/mocks/handlers.js"

interface MockFixtures {
  handlers: Array<AnyHandler>
  network: NetworkFixture
}

export const test = base.extend<MockFixtures>({
  handlers: [handlers, { option: true }],

  network: [
    async ({ context, handlers }, use) => {
      const network = defineNetworkFixture({
        context,
        handlers,
        onUnhandledRequest: "bypass",
        skipAssetRequests: true,
      })

      await network.enable()
      await use(network)
      await network.disable()
    },
    { auto: true },
  ],
})

export { expect }
