### Step 5 — Teacher model

Call `list_models` to discover available teacher models (name + provider).
Present ONLY those models as options — do NOT suggest models that aren't
returned by `list_models`, they won't work. If no models are returned,
stop — no teacher-model provider is configured on the server.

Call `present_options` with step="sdg-teacher-model" and each model as an option.
ALWAYS add as the last option:
- title: "Configure a model", description: "Configure a teacher-model provider key on the server", value: "Configure a model — Configure a teacher-model provider key on the server"
