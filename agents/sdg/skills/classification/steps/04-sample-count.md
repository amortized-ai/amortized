### Step 4 — Sample count

"How many training examples should we generate?" Scale based on category
count and desired coverage — at least 50 samples per category for basic
coverage, 150+ per category for production quality (N = number of
categories × urgency levels).

Call `present_options` with step="sdg-samples" and these options:
- title: "100 samples", description: "Quick prototype", value: "100 samples — Quick prototype"
- title: "500 samples", description: "Good coverage across categories", value: "500 samples — Good coverage across categories"
- title: "1000 samples", description: "Best model quality, more diverse examples", value: "1000 samples — Best model quality, more diverse examples"

Distribution: keep categories roughly balanced unless the real-world
distribution is known, in which case weight to match it.
