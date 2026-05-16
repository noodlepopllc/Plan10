#!/bin/bash
set -euo pipefail

mkdir -p $2/output
output="$2/output"

export SEED=$RANDOM

touch $output/empty.txt

if [[ ! -f "$output/registry.json" ]]; then
    # 1. Asset Registry (unchanged)
    echo "creating $output/registry.json"
    python lib/qwen_llm.py -P "$(cat "$1")" -S PlanningV2/prompts/assets.txt | tail -n +2 > $output/registry.json
fi

if [[ ! -f "$output/visual_prompt.txt" || ! -f "$output/action.txt" ]]; then
    # 2. Plain English Visual Expansion
    echo "creating $output/visual_prompt.txt"
    python PlanningV2/expanders/beat_expansion.py "$(cat "$1")" 8 > $output/visual_prompt.txt

fi

if [[ ! -f "$output/action.txt" ]]; then
    echo "creating $output/action.txt"
    python lib/qwen_llm.py -P "$(cat $output/visual_prompt.txt)" | tail -n +2 > $output/action.txt
fi

if [[ ! -f "$output/dialog.txt"  ]]; then
    # 3. Generate Dialog (EXACTLY as you had it)
    echo "creating $output/dialog.txt"
    DIALOG_PROMPT=$(python PlanningV2/expanders/dialog_headshot.py "$1" $output/registry.json 8)
    python lib/qwen_llm.py -P "$DIALOG_PROMPT" | tail -n +2 > $output/dialog.txt
fi

if [[ ! -f "$output/map_prompt_dialog.txt" || ! -f "$output/sequence_dialog.json" ]]; then
    # 4. Schema Mapper (combines visuals + dialog + registry → final JSON)
    echo "creating $output/map_prompt_dialog.txt"
    python PlanningV2/expanders/beat_mapper.py $output/registry.json $output/empty.txt $output/dialog.txt "$1" > $output/map_prompt_dialog.txt

    echo "creating $output/sequence_dialog.json"
    python lib/qwen_llm.py -P "$(cat $output/map_prompt_dialog.txt)" -S PlanningV2/prompts/sequence.txt | tail -n +2 > $output/sequence_dialog.json
fi

if [[ ! -f "$output/map_prompt_action.txt" || ! -f "$output/sequence_action.json" ]]; then
    echo "creating $output/map_prompt_action.txt"
    python PlanningV2/expanders/beat_mapper.py $output/registry.json $output/action.txt $output/empty.txt "$1" > $output/map_prompt_action.txt
    
    echo "creating $output/sequence_action.json"
    python lib/qwen_llm.py -P "$(cat $output/map_prompt_action.txt)" -S PlanningV2/prompts/sequence.txt | tail -n +2 > $output/sequence_action.json
fi

if [[ ! -f "$output/map_prompt_combined.txt" || ! -f "$output/sequence_combined.json" ]]; then
    echo "creating $output/map_prompt_action.txt"
    python PlanningV2/expanders/beat_mapperV2.py $output/registry.json $output/action.txt $output/dialog.txt "$1" > $output/map_prompt_combined.txt
    
    echo "creating $output/sequence_action.json"
    python lib/qwen_llm.py -P "$(cat $output/map_prompt_action.txt)" -S PlanningV2/prompts/sequence.txt | tail -n +2 > $output/sequence_combined.json
fi

rm -f $2/complete.txt
python PlanningV2/builders/identity.py $output/registry.json > "$2/complete.txt"
python PlanningV2/builders/actions.py $output/registry.json $output/sequence_combined.json >> "$2/complete.txt"
python PlanningV2/builders/dialog.py $output/registry.json $output/sequence_combined.json >> "$2/complete.txt"


echo "#!/bin/bash" > $2/final.sh
echo "set -euo pipefail" >> $2/final.sh
chmod 0777 $2/final.sh

echo "rm -f $2/final.txt" > $2/final.sh

if [[ ! -f "$2/identity.txt" ]]; then
    python PlanningV2/builders/identity.py $output/registry.json > $2/identity.txt
fi
echo "cat $2/identity.txt > $2/final.txt" >> $2/final.sh

echo "### ACTION SHOTS (Pick One)" >> $2/final.sh
if [[ ! -f "$2/action_images.txt" ]]; then
    python PlanningV2/builders/actions.py $output/registry.json $output/sequence_action.json --images-only > "$2/action_images.txt"
fi
echo "# cat $2/action_images.txt >> $2/final.txt" >> $2/final.sh

if [[ ! -f "$2/action_videos.txt" ]]; then
    python PlanningV2/builders/actions.py $output/registry.json $output/sequence_action.json > "$2/action_videos.txt"
fi
echo "# cat $2/action_videos.txt >> $2/final.txt" >> $2/final.sh

echo "### DIALOG SHOTS (Pick One)" >> $2/final.sh
if [[ ! -f "$2/headshots.txt" ]]; then
    python PlanningV2/builders/dialog.py $output/registry.json $output/sequence_dialog.json --headshots-only > "$2/headshots.txt"
fi
echo "# cat $2/headshots.txt >> $2/final.txt" >> $2/final.sh

if [[ ! -f "$2/static_dialog.txt" ]]; then
    python PlanningV2/builders/dialog.py $output/registry.json $output/sequence_dialog.json --images-only > "$2/static_dialog.txt"
fi
echo "# cat $2/static_dialog.txt >> $2/final.txt" >> $2/final.sh

if [[ ! -f "$2/all_dialog.txt" ]]; then
    python PlanningV2/builders/dialog.py $output/registry.json $output/sequence_dialog.json > "$2/all_dialog.txt"
fi
echo "# cat $2/all_dialog.txt >> $2/final.txt" >> $2/final.sh

echo "✅ Pipeline complete: $2"