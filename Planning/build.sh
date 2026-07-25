#!/bin/bash
set -euo pipefail

mkdir -p $2/output
output="$2/output"

basepath="Planning"

if [[ ! -f "$output/story.txt" ]]; then
    python $basepath/builders/storywriter.py -S $1 -O $output/story.txt
fi

if [[ ! -f "$output/script.txt" ]]; then
    python $basepath/builders/script.py $output
fi

if [[ ! -f "$output/complete.json" ]]; then
    python $basepath/builders/scriptwriter.py $1 $output
fi

python $basepath/renderer/renderer.py $2 $3 > $2/scene.txt


echo "✅ Pipeline complete: scene $2"
