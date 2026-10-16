### Step 4 — Validate and submit
Call `signal_phase` this response with the step for the phase you are in — this phase: `step-4-validate`.

**First, check whether this eval already exists.** Call `list_evaluations`
— it returns evaluation groups (one per dataset × metric set) with the
models already scored in each. If THIS model already appears for THIS
dataset and metric set, tell the user it's already been evaluated, show
those scores, and ask whether to re-run anyway. Note: re-running with the
SAME setup (temperature, sample count, judge) averages into that model's
existing column; a different setup makes a new column. Only continue below
if they want a fresh run.

Call `validate_eval_job` with the assembled config. If validation
fails (422), read the error, fix the config (e.g. set a judge
explicitly when there is no SDG ancestor to auto-fill one), and call
`validate_eval_job` AGAIN — do not present a text summary of a config
that failed validation.

If the response carries `warnings` — e.g. the submitted rubric criteria
match an earlier eval on this dataset by name but differ in wording —
surface the warning to the user and resolve it BEFORE the confirmation
card (reuse the earlier criteria verbatim, typically by copying them
from the previous eval's config, unless the user explicitly wants
different criteria — otherwise the new eval gets its own row in the
Evaluation tab and scores are not comparable).

**The confirmation card is rendered by the PLATFORM from the successful
`validate_eval_job` tool result — you never render it yourself.** Do
NOT use `present_options` (or any text/markdown) as a submit control:
option cards only send chat text back to you, they cannot create a job.
After a successful validate, write ONE short sentence pointing the user
to the card. The config the agent assembles is often sparse —
unset fields are resolved server-side, and the user can't tell what
they're agreeing to from the raw JSON alone. So in that sentence, ALWAYS
state the effective settings in one line:

- temperature: the configured value, or "0 (default)" when unset
- samples: the configured max_samples, or "all records (default)" when unset/0
- judge: the configured judge model (a rubric eval always has one — you
  set it explicitly)
- judge samples: the configured judge_max_samples, or "all (default)" when unset/0

Example line: `temperature 0 (default) · all records (default) · judge:
gpt-oss · judge samples: all (default)`

After the user clicks Confirm on the platform card, the job is
submitted and you'll be notified when it completes. Never claim the job
was submitted — check the job's existence first if in doubt.
