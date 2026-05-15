#!/bin/bash
set -euo pipefail

USER_PROMPT="${1:-princess captured dungeon wizard}"
SEED="${2:-}"

# Build command
CMD="python PlanningV2/expanders/storyboard_expander.py '$USER_PROMPT'"
[[ -n "$SEED" ]] && CMD="$CMD --seed $SEED"

# Expand → generate
EXPANDED=$(eval "$CMD")
python lib/graphics_gen.py -P "$EXPANDED" -O "output/${USER_PROMPT// /_}.png"

echo "✓ Generated: output/${USER_PROMPT// /_}.png"