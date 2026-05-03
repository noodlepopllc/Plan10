#!/bin/bash
set -euo pipefail

OUTDIR="tests/$1"
mkdir -p "$OUTDIR"
BG="$OUTDIR/location.png"
A="$OUTDIR/char1.png"
B="$OUTDIR/char2.png"
BG_REV="$OUTDIR/location_reverse.png"
HEIGHT=832
WIDTH=480

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

# ─── SHOTS ───
echo "=== MASTER ==="
shot "$BG" "$A" "$B" "two_shot" "The women greet each other." "master_close" "hair gently swaying, subtle weight shift"
shot "$BG_REV" "$B" "$A" "two_shot" "The women begin arguing." "master_close_rev" "hair gently swaying, subtle posture tension"

echo "=== CLOSEUPS & REACTIONS ==="
shot "$BG" "$A" "$A" "closeup" "She smiles happily." "reaction_A" "eyes blinking naturally, subtle head tilt"
shot "$BG_REV" "$B" "$B" "closeup" "She smiles happily." "reaction_B" "eyes blinking naturally, soft exhale"

pan "master_close" "pan-left" "master_close_pan_left"
pan "master_close" "pan-right" "master_close_pan_right"

pan "master_close_rev" "pan-left" "master_close_pan_left_rev"
pan "master_close_rev" "pan-right" "master_close_pan_right_rev"

echo "=== OVER-SHOULDER ==="
shot "$BG_REV" "$A" "$B" "ots" "She speaks in a friendly manner" "ots_A_to_B" "hair softly swaying, subtle breathing"
shot "$BG" "$B" "$A" "ots" "She speaks in a friendly manner" "ots_B_to_A" "hair softly swaying, relaxed posture"

zoom "ots_A_to_B" "reaction_B" "" "zoom_ots_A_to_B"
zoom "ots_B_to_A" "reaction_A" "" "zoom_ots_B_to_A"

echo "=== SINGLES & PROFILES ==="
shot "$BG" "$A" "$A" "medium" "She poses like a model." "single_A" "subtle stance shift"
shot "$BG_REV" "$B" "$B" "medium" "She poses like a model." "single_B" "subtle breathing"

zoom "single_A" "reaction_A" "" "zoom_A"
zoom "single_B" "reaction_B" "" "zoom_B"

echo "✅ All shots generated into '$OUTDIR/'"