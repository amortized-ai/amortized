### Step 1 — Domain

What kind of content will this classifier handle? e.g., customer support
tickets, user messages, content moderation.

Call `present_options` with step="sdg-domain" and these options:
- title: "Software/technical support", description: "Bug reports, feature requests, troubleshooting", value: "Software/technical support — Bug reports, feature requests, troubleshooting"
- title: "Billing & payments", description: "Invoices, refunds, subscription issues", value: "Billing & payments — Invoices, refunds, subscription issues"
- title: "Customer service", description: "Account access, onboarding, general inquiries", value: "Customer service — Account access, onboarding, general inquiries"
- title: "E-commerce", description: "Orders, shipping, returns, product questions", value: "E-commerce — Orders, shipping, returns, product questions"

STOP here. Do NOT continue to step 2 in this message.

When you ask the user about this step, call present_options with step: "domain".
