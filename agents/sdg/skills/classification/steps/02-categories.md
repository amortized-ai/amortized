### Step 2 — Categories (ask AFTER user picks domain)

What labels should the classifier predict? Based on the domain they
chose, suggest specific category labels:

"What categories should it classify into?"

For customer support, call `present_options` with step="sdg-categories" and these options:
- title: "Standard categories", description: "Billing, Technical, Account, General Inquiry", value: "Standard categories — Billing, Technical, Account, General Inquiry"
- title: "Detailed categories", description: "Billing, Technical, Account, Shipping, Returns, Product Questions", value: "Detailed categories — Billing, Technical, Account, Shipping, Returns, Product Questions"
- title: "Custom categories", description: "I'll define my own labels", value: "Custom categories — I'll define my own labels"

For other domains, suggest 3-4 relevant groupings.

STOP here. Do NOT continue to step 3 in this message.

When you ask the user about this step, call present_options with step: "categories".
