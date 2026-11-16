/** Label for a classification eval group, so a head's held-out accuracy and a router's
 * separability proxy (anchors drawn from the eval set) aren't read as the same number. */
export function classifierKindLabel(kind: "head" | "router"): string {
  return kind === "head" ? "classifier head" : "embedding router (separability proxy)"
}
