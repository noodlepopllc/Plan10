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

        echo "🎬 Generating I2V: $out_suffix"
        python lib/image_to_video.py -P "$vid_prompt" -I "$out" -O "$out_vid" -W "$WIDTH" -H "$HEIGHT" -S "$SEED" -D 3.0 || { echo "❌ I2V failed: $out_suffix"; exit 1; }
        
        echo "✅ $out_suffix | T2I: $action | I2V: $vid_prompt" >> "$OUTDIR/run_manifest.txt"
    else
        echo "⏭️ Skipping $out_suffix (exists)"
    fi
}

# ─── SHOTS ───
echo "=== MASTER ==="
shot "$BG" "$A" "$B" "two_shot" "The women greet each other." "master_close" "hair gently swaying, subtle weight shift"
shot "$BG_REV" "$B" "$A" "two_shot" "The women begin arguing." "master_close_rev" "hair gently swaying, subtle posture tension"

echo "=== OVER-SHOULDER ==="
shot "$BG_REV" "$A" "$B" "ots" "She speaks angrily" "ots_A_to_B" "hair softly swaying, subtle breathing"
shot "$BG" "$B" "$A" "ots" "She speaks in a friendly manner" "ots_B_to_A" "hair softly swaying, relaxed posture"

echo "=== CLOSEUPS & REACTIONS ==="
shot "$BG" "$A" "$A" "closeup" "She smiles happily." "reaction_A" "eyes blinking naturally, subtle head tilt"
shot "$BG_REV" "$B" "$B" "closeup" "She frowns unhappily." "reaction_B" "eyes blinking naturally, soft exhale"

echo "=== SINGLES & PROFILES ==="
shot "$BG" "$A" "$A" "profile_right" "She points to something out of frame." "profile_A" "hair gently swaying, arm relaxed"
shot "$BG" "$A" "$A" "medium" "She poses like a model." "single_A" "fabric rippling softly, subtle stance shift"
shot "$BG" "$B" "$B" "profile_left" "She looks up above her." "profile_B" "hair gently swaying, subtle head lift"
shot "$BG_REV" "$B" "$B" "medium" "She poses like an idol." "single_B" "fabric rippling softly, subtle breathing"

echo "✅ All shots generated into '$OUTDIR/'"