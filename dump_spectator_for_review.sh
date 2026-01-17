#!/usr/bin/env bash
set -euo pipefail

OUT="spectator_full_dump.txt"
ROOT="$(pwd)"

echo "Writing $OUT ..."
echo "" > "$OUT"

emit() {
  local title="$1"
  local file="$2"
  echo "" >> "$OUT"
  echo "================================================================================" >> "$OUT"
  echo "### $title" >> "$OUT"
  echo "### PATH: $file" >> "$OUT"
  echo "================================================================================" >> "$OUT"
  echo "" >> "$OUT"
  sed 's/\t/    /g' "$file" >> "$OUT"
}

emit_block() {
  local title="$1"
  shift
  echo "" >> "$OUT"
  echo "================================================================================" >> "$OUT"
  echo "### $title" >> "$OUT"
  echo "================================================================================" >> "$OUT"
  echo "" >> "$OUT"
  for f in "$@"; do
    emit "$(basename "$f")" "$f"
  done
}

# ------------------------------------------------------------------------------
# 1. High-level docs & project metadata
# ------------------------------------------------------------------------------

emit_block "PROJECT OVERVIEW & CONTRACTS" \
  README.md \
  pyproject.toml \
  Makefile \
  docs/architecture.md \
  docs/contracts.md \
  docs/roadmap_steps.md \
  docs/testing.md

# ------------------------------------------------------------------------------
# 2. Prompts (CRITICAL)
# ------------------------------------------------------------------------------

emit_block "ROLE PROMPTS" \
  src/spectator/prompts/roles/reflection.txt \
  src/spectator/prompts/roles/planner.txt \
  src/spectator/prompts/roles/critic.txt \
  src/spectator/prompts/roles/governor.txt

emit_block "SYSTEM / LLAMA RULES" \
  src/spectator/prompts/system/llama_rules.txt \
  src/spectator/prompts/system/test_llama_rules.txt

emit "Prompt loader" src/spectator/prompts/loader.py

# ------------------------------------------------------------------------------
# 3. Runtime core & pipeline
# ------------------------------------------------------------------------------

emit_block "RUNTIME CORE" \
  src/spectator/runtime/pipeline.py \
  src/spectator/runtime/controller.py \
  src/spectator/runtime/sanitize.py \
  src/spectator/runtime/notes.py \
  src/spectator/runtime/tool_calls.py \
  src/spectator/runtime/condense.py \
  src/spectator/runtime/capabilities.py \
  src/spectator/runtime/memory_feedback.py

# ------------------------------------------------------------------------------
# 4. Backends (focus on llama)
# ------------------------------------------------------------------------------

emit_block "BACKENDS" \
  src/spectator/backends/llama_server.py \
  src/spectator/backends/registry.py \
  src/spectator/backends/fake.py

# ------------------------------------------------------------------------------
# 5. Tools & sandbox
# ------------------------------------------------------------------------------

emit_block "TOOLS & SANDBOX" \
  src/spectator/tools/executor.py \
  src/spectator/tools/registry.py \
  src/spectator/tools/results.py \
  src/spectator/tools/sandbox.py \
  src/spectator/tools/fs_tools.py \
  src/spectator/tools/shell_tool.py \
  src/spectator/tools/http_tool.py \
  src/spectator/tools/http_cache.py

# ------------------------------------------------------------------------------
# 6. Memory & retrieval
# ------------------------------------------------------------------------------

emit_block "MEMORY & RETRIEVAL" \
  src/spectator/memory/context.py \
  src/spectator/memory/retrieval.py \
  src/spectator/memory/vector_store.py \
  src/spectator/memory/embeddings.py

# ------------------------------------------------------------------------------
# 7. Tracing & telemetry
# ------------------------------------------------------------------------------

emit_block "TRACING & TELEMETRY" \
  src/spectator/core/tracing.py \
  src/spectator/core/telemetry.py \
  src/spectator/core/types.py

# ------------------------------------------------------------------------------
# 8. Tests that define behavior (selective, architecture-relevant)
# ------------------------------------------------------------------------------

emit_block "KEY TESTS (ARCHITECTURE DEFINING)" \
  tests/test_pipeline.py \
  tests/test_pipeline_sanitizes_visible_output.py \
  tests/test_history_prompt.py \
  tests/test_governor_tool_loop.py \
  tests/test_llama_backend_prompts.py \
  tests/test_sanitize.py \
  tests/test_tool_calls.py \
  tests/test_http_get_tool.py \
  tests/test_smoke_run_pipeline.py

echo ""
echo "DONE."
echo "Output file: $OUT"

