#!/bin/bash
set -euo pipefail

rm -rf Planning/outputs
mkdir -p Planning/outputs

touch Planning/outputs/empty.txt

# 1. Asset Registry (unchanged)
python lib/qwen_llm.py -P "$(cat "$1")" -S Planning/assets.txt | tail -n +2 > Planning/outputs/registry.json

# 2. Plain English Visual Expansion
python Planning/beat_expansion.py "$(cat "$1")" 4 > Planning/outputs/visual_prompt.txt
python lib/qwen_llm.py -P "$(cat Planning/outputs/visual_prompt.txt)" -S Planning/sequence.txt | tail -n +2 > Planning/outputs/beats_raw.txt

# 3. Generate Dialog (EXACTLY as you had it)
DIALOG_PROMPT=$(python Planning/dialog_headshot.py "$1" Planning/outputs/registry.json 4)
python lib/qwen_llm.py -P "$DIALOG_PROMPT" -S Planning/dialog_system.txt | tail -n +2 > Planning/outputs/dialog.txt

# 4. Schema Mapper (combines visuals + dialog + registry → final JSON)
python Planning/beat_mapper.py Planning/outputs/registry.json Planning/outputs/empty.txt Planning/outputs/dialog.txt "$1" > Planning/outputs/map_prompt_dialog.txt
python lib/qwen_llm.py -P "$(cat Planning/outputs/map_prompt_dialog.txt)" -S Planning/sequence.txt | tail -n +2 > Planning/outputs/sequence_dialog.json

python Planning/beat_mapper.py Planning/outputs/registry.json Planning/outputs/beats_raw.txt Planning/outputs/empty.txt "$1" > Planning/outputs/map_prompt_action.txt
python lib/qwen_llm.py -P "$(cat Planning/outputs/map_prompt_action.txt)" -S Planning/sequence.txt | tail -n +2 > Planning/outputs/sequence_action.json

# 5. Final Build
python Planning/build.py Planning/outputs/registry.json Planning/outputs/sequence_action.json Planning/outputs/sequence_dialog.json > "$2"
python Planning/cinematic_establishing.py Planning/outputs/registry.json >> "$2"

echo "✅ Pipeline complete: $2"