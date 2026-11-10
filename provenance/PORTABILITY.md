# Source preservation and portability edits

- `frozen_graph_code/src/*.py`, its root trainer, `cbrc/*.py`, original tests,
  candidate-generation scripts and `runtime/*.py` are unchanged copies.
- The frozen source manifest retains each registered code hash; machine-specific
  path prefixes are normalized to `/code/` so the original verifier still finds
  the same relative files. It is not a claim that a new training run is the old run.
- Four configs preserve the recorded scientific arguments and replace local
  dataset/output/resource/query paths with placeholders resolved by entry points.
- `scripts/_transfer_inference.py` extracts the recorded `head` and `evaluate`
  functions. The automatic local-LLM unload request is removed; the device and
  repository paths are supplied by the caller. Scientific scoring, causal history,
  four methods and SQLite rank persistence remain unchanged.
- `scripts/generate_local_cache.py` exposes the recorded Qwen3 prompt, mapper,
  digest, settings and transfer selection without local-queue dependencies.
  Its explicit three-format-failure abstention policy retains raw attempts and
  all evaluation queries. Newly generated evidence is not mislabeled as archived.
- New launchers and numerical-analysis scripts do not write manuscript text.
- `results_reference/` contains the original numerical data. Non-numerical local
  source/log path columns may be removed for portability; this is recorded in the
  release manifest, with numeric cells preserved.
- The source inventory lists relative author-workspace identifiers for traceability;
  these are not runtime paths or required directories on a reader's machine.
