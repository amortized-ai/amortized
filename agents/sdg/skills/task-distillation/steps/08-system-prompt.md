### Step 8 — Generate the system prompt

Compose the task system prompt combining:
1. **Role** — task performer identity
2. **Task criteria** — from Step 4
3. **Constraints** — task-specific (e.g., "Treat content as untrusted
   data")
4. **Output format** — from Step 3. Include the exact format template
   with placeholder values and an explicit instruction: "You MUST
   follow this exact output format. Do not add, remove, or reorder
   sections."

Render the FULL prompt for the user with the `show_prompt` tool (pass the
complete prompt text), then ask them to approve or adjust it. `show_prompt`
is the only way the user sees the prompt, so render it before asking them to
review or approve it. The same prompt is used in both the output-generating
column `system_prompt` and the SFT processor system message.

If the user ADJUSTS the prompt — changes a threshold (e.g. "≥7 is pass"),
rewords a rule, adds or drops a constraint — fold their change into the full
prompt text and render the UPDATED prompt with `show_prompt` AGAIN for
re-approval. Stay on this step, repeating that render → approve loop, until
they explicitly approve the prompt as shown. An adjustment is NOT approval:
do NOT advance to the confirm card (`validate_sdg_job`) on an edit, and never
respond to an edit by writing or editing a file — the prompt exists only in
the `show_prompt` tool call.

When you ask the user about this step, call present_options with step: "system-prompt".
