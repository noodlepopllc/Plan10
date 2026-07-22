#!/bin/bash
set -euo pipefail
if [ ! -d "tests/$1" ]; then 
    python tests/character_builder.py -D -N $1 -R "latinx_mestizo" -C "red" -T "tan" -H "random" -S "long waves"
fi

if [ ! -d "tests/$2" ]; then
    python tests/character_builder.py -D -N $2 -R "east_asian" -C "blonde" -T "fair" -H "random" -S "soft bob"
fi

if [ ! -d "tests/$1_$2" ]; then
   python tests/persons.py $1 $2
fi

OUTDIR="tests/$1_$2"
mkdir -p "$OUTDIR"
BG="$OUTDIR/location.png"
A="$OUTDIR/char1.png"
B="$OUTDIR/char2.png"
BG_REV="$OUTDIR/location_reverse.png"
BG_LEFT="$OUTDIR/location_left.png"
BG_RIGHT="$OUTDIR/location_right.png"

python lib/config.py -R

source .env
HEIGHT=$HEIGHT
WIDTH=$WIDTH

SEED=${SEED:-$RANDOM}
echo "🎲 Seed: $SEED | Date: $(date)" > "$OUTDIR/run_manifest.txt"

shot() {
    local bg="$1" char1="$2" char2="$3" shot_type="$4" action="$5" out_suffix="$6" vid_prompt="$7"
    local out="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}.png"
    local out_vid="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}.mp4"

    if [[ ! -f "$out" ]]; then
        echo "🎨 Generating T2I: $out_suffix"
        
        # Always pass both chars. Your Python patch handles routing/ignoring.
        python lib/compositor.py -B "$bg" -C "$char1" -C "$char2" \
            -S "$shot_type" -A "$action" \
            -O "$out" -E "$SEED" -H "$HEIGHT" -W "$WIDTH" || { echo "❌ Compositor failed: $out_suffix"; exit 1; }
            
        touch "$out"  # ✅ Refreshes OS thumbnail cache
        
        echo "✅ $out_suffix | T2I: $action" >> "$OUTDIR/run_manifest.txt"
    else
        echo "⏭️ Skipping $out_suffix (exists)"
    fi
}

zoom() {
    local input1="$1"
    local input2="$2"
    local person="$3"
    local out_suffix="$4"

    local target="$OUTDIR/${WIDTH}_${HEIGHT}_${input1}.png" 
    local reference="$OUTDIR/${WIDTH}_${HEIGHT}_${input2}.png" 
    local out="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}.png"
    local out_vid="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}.mp4"

    if [[ ! -f "$out" ]]; then
        echo " Zooming: $out_suffix"
        python lib/camera.py -I  "$target" -I "$reference" -T "$person" -S 30 -E "$SEED" -O "$out"
        touch "$out"
        python lib/image_to_video.py -I "$target" -I "$out" -P "Camera zooms in on the subject" -O "$out_vid" -W "$WIDTH" -H "$HEIGHT" -S "$SEED" -D 5.0
         
        echo "✅ $out_suffix | ZOOM: ${input1} ${input2}" >> "$OUTDIR/run_manifest.txt" 
    else
        echo "⏭️ Skipping $out_suffix (exists)"
    fi
}

pan() {
    local input1="$1"
    local movetype="$2"
    local out_suffix="$3"

    local target="$OUTDIR/${WIDTH}_${HEIGHT}_${input1}.png" 
    local out="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}.png"
    local out_vid="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}.mp4"

    if [[ ! -f "$out" ]]; then
        echo " Zooming: $out_suffix"
        python lib/camera.py -I  "$target" -C "$movetype" -S 30 -E "$SEED" -O "$out"
        touch "$out" 
        python lib/image_to_video.py -I "$target" -I "$out" -P "Camera $movetype slowly" -O "$out_vid" -W "$WIDTH" -H "$HEIGHT" -S "$SEED" -D 5.0
        
        echo "✅ $out_suffix | Pan: ${input1} ${movetype}" >> "$OUTDIR/run_manifest.txt" 
    else
        echo "⏭️ Skipping $out_suffix (exists)"
    fi
}

# ─── BACKGROUNDS ───
echo "=== BACKGROUNDS ==="
if [ ! -f "$BG_LEFT" ]; then 
    python lib/compositor.py -B $BG -Z "left" -O "$BG_LEFT" -R 
fi

if [ ! -f "$BG_RIGHT" ]; then 
    python lib/compositor.py -B $BG -Z "right" -O "$BG_RIGHT" -R
fi

# ─── SHOTS ───

echo "=== CLOSEUPS & REACTIONS ==="
shot "$BG_LEFT" "$A" "$A" "closeup" "She smiles happily." "reaction_A" "eyes blinking naturally, subtle head tilt"
shot "$BG_RIGHT" "$B" "$B" "closeup" "She smiles happily." "reaction_B" "eyes blinking naturally, soft exhale"

echo "=== SINGLES & PROFILES ==="
shot "$BG_LEFT" "$A" "$A" "medium" "She poses like a model." "single_A" "subtle stance shift"
shot "$BG_RIGHT" "$B" "$B" "medium" "She poses like a model." "single_B" "subtle breathing"

zoom "single_A" "reaction_A" "" "zoom_A"
zoom "single_B" "reaction_B" "" "zoom_B"

echo "=== MASTER ==="
shot "$BG" "$A" "$B" "two_shot" "The women look towards each other 3/4 view" "master_close" "hair gently swaying, subtle weight shift"

pan "master_close" "pan-left" "master_close_pan_left"
pan "master_close" "pan-right" "master_close_pan_right"

echo "=== OVER-SHOULDER ==="
shot "$BG_RIGHT" "$A" "$B" "ots" "She speaks in a friendly manner" "ots_A_to_B" "hair softly swaying, subtle breathing"
shot "$BG_LEFT" "$B" "$A" "ots" "She speaks in a friendly manner" "ots_B_to_A" "hair softly swaying, relaxed posture"

zoom "ots_A_to_B" "reaction_B" "" "zoom_ots_A_to_B"
zoom "ots_B_to_A" "reaction_A" "" "zoom_ots_B_to_A"

echo "✅ All shots generated into '$OUTDIR/'"
