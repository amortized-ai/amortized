### Step 4 — Which teacher model?

Call `list_models` to get the available teacher models (each entry has a
`name` and a `provider`). Present ONLY those models as options. Do NOT
suggest models that aren't returned by `list_models` — they won't work.
If no models are returned, stop — no teacher-model provider is configured
on the server.

When you ask the user about this step, call present_options with step: "teacher-model".
