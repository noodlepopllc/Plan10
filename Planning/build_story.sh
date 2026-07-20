#!/bin/bash
set -euo pipefail

mkdir -p $2/output
output="$2/output"

cp $1 $output/story.txt

basepath="Planning"

if [[ ! -f "$output/complete.json" ]]; then
    python $basepath/builders/scriptwriter.py $1 $output
fi

python $basepath/renderer/renderer.py $2 $3 > $2/scene.txt


echo "✅ Pipeline complete: scene $2"
