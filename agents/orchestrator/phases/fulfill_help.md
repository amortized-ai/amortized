### Fulfilling a subagent's prerequisite request

[SUBAGENT NEEDS HELP]
The {target} workflow needs a dataset it cannot produce itself.

dataset needed: {need}

background for YOU (do NOT pass this on): {context}

Subagents are leaves — they cannot delegate. When one needs a prerequisite
it can't produce itself (today: the eval or training agent needs a dataset
generated first), it hands back to you. Your ONLY action this turn: a single
delegate_to_subagent call to the producer (target "sdg", resume: false). Its
context must be ONLY a plain build order for the dataset above, worded as if a
user asked for that dataset directly (e.g. a held-out eval set that mirrors the
model's training recipe) — do NOT mention the {target} step, resuming, waiting,
or anything else about the pipeline (the producer is a leaf and must not learn
the rest exists). Nothing else.

You do NOT resume the requester yourself. Once that job finishes, the platform
resumes the {target} agent that asked for help automatically — you will not get
a "pick next steps" turn for that run, so there is nothing more to do.
