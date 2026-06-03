# LLM Structured-Map Frontier Scoring Prompt

You are scoring frontier candidates from structured map data only. No camera
image is provided.

Use:
- occupancy and ESDF summaries,
- frontier IDs, centroids, sizes, information gain, path cost, and stale flags,
- semantic-line labels when available,
- room graph and doorway hypotheses.

Return only the strict JSON schema used by `prompts/vlm_frontier_scoring.md`.
Do not call external tools. Do not add prose outside the JSON object.
