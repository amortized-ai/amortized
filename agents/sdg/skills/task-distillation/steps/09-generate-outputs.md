### Step 9 — Generate outputs for raw inputs (optional)

If the user provided only raw inputs (without outputs) in Step 2,
use the system prompt from Step 8 to generate outputs for each raw
input now. Present the generated (input, output) pairs to the user
for review. These become the demonstration examples used in the SDG
config.

When you ask the user about this step, call present_options with step: "generate-outputs".
