### Step 1 branch (only when no eval dataset exists yet) — Ask the held-out record count
Call `signal_phase` this response with the step for the phase you are in — this phase: `held-out-ask-count`.

When a held-out eval dataset does not exist yet (see the `step-1-dataset`
phase), you cannot generate one yourself — generating data is the SDG
agent's job, and you are a leaf: you do not delegate. You will hand back to
the orchestrator to have it generated, but FIRST settle the size here.

Ask the user how many eval records to generate (offer a sensible default,
e.g. 100) — the SDG agent must not pick the count on its own. This is a
question turn: ask, and wait for the answer.

Once the user gives the count, move to the `held-out-request-help` phase and
carry that number into the handback request so the set is generated at
exactly that size.
