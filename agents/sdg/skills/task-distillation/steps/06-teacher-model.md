### Step 6 — Teacher model

Call `list_models` to get the available teacher models. Each entry has
a `name` and a `provider`. Copy both values exactly as returned — do
not shorten, re-prefix, or modify the `name` string (e.g., if
`list_models` returns `openai/openai/gpt-oss-120b`, use that full
string, not `openai/gpt-oss-120b`). Present ONLY those models as
options. If no models are returned, stop — no teacher-model provider
is configured on the server.

When you ask the user about this step, call present_options with step: "teacher-model".
