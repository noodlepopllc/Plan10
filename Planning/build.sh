#!/bin/bash
set -euo pipefail

mkdir -p Planning/outputs

# 1. Generate Asset Registry
python lib/qwen_llm.py -P "$(cat "$1")" -S Planning/assets.txt | tail -n +2 > Planning/outputs/registry.json

# 2. Build Base Sequence Prompt (rules + schema)
python Planning/build_sequence_prompt.py Planning/outputs/registry.json "$1" > Planning/outputs/combined.txt

# 3. Generate & Append Dialog
echo -e "\n### PROVIDED DIALOG (MAP EXACTLY)\n" >> Planning/outputs/combined.txt
DIALOG_PROMPT=$(python Planning/dialog_headshot.py "$1" Planning/outputs/registry.json 14)
python lib/qwen_llm.py -P "$DIALOG_PROMPT" -S Planning/dialog_system.txt | tail -n +2 >> Planning/outputs/combined.txt

# 4. Generate Final Sequence JSON
python lib/qwen_llm.py -P "$(cat Planning/outputs/combined.txt)" -S Planning/sequence.txt | tail -n +2 > Planning/outputs/sequence.json

# 5. Final Build
python Planning/build.py Planning/outputs/registry.json Planning/outputs/sequence.json > "$2"

echo "✅ Pipeline complete: $2"