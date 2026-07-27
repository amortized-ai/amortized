import { Logger } from "tslog"
import type { ILogObj } from "tslog"

const isProd = import.meta.env.PROD

const rootLogger = new Logger<ILogObj>({
  name: "studio",
  type: isProd ? "json" : "pretty",
  minLevel: isProd ? 2 : 0,
})

rootLogger.info("Amortized Studio logger initialized", {
  mode: isProd ? "production" : "development",
  logLevel: isProd ? 2 : 0,
})

export function getLogger(name: string): Logger<ILogObj> {
  return rootLogger.getSubLogger({ name })
}

export { rootLogger as logger }
