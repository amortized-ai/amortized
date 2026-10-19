### Step 6 — System prompt for the trained model

Do NOT ask the user to write a system prompt. Generate a domain-specific
default based on the document content and include it in the config.
Only mention it in the confirmation table so the user can adjust if
they want to.

When you ask the user about this step, call present_options with step: "system-prompt".
