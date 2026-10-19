### Step 5 — How many samples?

Documents are chunked at upload time. Use
`get_document_chunks(doc_id)` to get the chunk count and token
statistics.

Use ALL chunks in the calculation — the worker sends every chunk
to DataDesigner, so do not filter or exclude any chunks here.

The goal is for total training tokens to be a multiple of total
source tokens. Research suggests ~5x source coverage as a good
target. Compute the per-chunk multiplier from the document's
actual chunk statistics:

```
avg_qa_tokens ≈ 200  (empirical median across past runs; ranges 80-300
                      depending on prompt style and answer depth)
mean_chunk_tokens = mean of num_tokens across ALL chunks
multiplier = coverage × mean_chunk_tokens / avg_qa_tokens
num_samples = multiplier × num_chunks  (num_chunks = total, not filtered)
```

Present three coverage tiers and show the computed sample counts:

1) 3x source coverage — Good starting point
2) 5x source coverage — Recommended (research-backed default)
3) 8x source coverage — Best quality, thorough coverage

For example, a document with 50 chunks of mean 400 tokens:
- 3x: multiplier = 3 × 400 / 200 = 6x → 300 samples
- 5x: multiplier = 5 × 400 / 200 = 10x → 500 samples
- 8x: multiplier = 8 × 400 / 200 = 16x → 800 samples

Show the user the actual numbers, the multiplier, and the coverage
tier so they understand the reasoning.
