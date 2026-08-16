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

uv run config -R

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
        uv run compositor -B "$bg" -C "$char1" -C "$char2" \
            -S "$shot_type" -A "$action" \
            -O "$out" || { echo "❌ Compositor failed: $out_suffix"; exit 1; }
            
        touch "$out"  # ✅ Refreshes OS thumbnail cache

        echo "🎬 Generating I2V: $out_suffix"
        uv run image_to_video -P "$vid_prompt" -I "$out" -O "$out_vid" -D 4 || { echo "❌ I2V failed: $out_suffix"; exit 1; }
        
        echo "✅ $out_suffix | T2I: $action | I2V: $vid_prompt" >> "$OUTDIR/run_manifest.txt"
    else
        echo "⏭️ Skipping $out_suffix (exists)"
    fi
}

# ─── BACKGROUNDS ───
echo "=== BACKGROUNDS ==="
if [ ! -f "$BG_LEFT" ]; then 
    uv run compositor -B $BG -Z "left" -O "$BG_LEFT" -R 
fi

if [ ! -f "$BG_RIGHT" ]; then 
    uv run compositor -B $BG -Z "right" -O "$BG_RIGHT" -R
fi

# ─── SHOTS ───
echo "=== MASTER ==="
shot "$BG" "$A" "$B" "two_shot" "The women face each other." "master_close" "hair gently swaying, subtle weight shift"

echo "=== OVER-SHOULDER ==="
shot "$BG_RIGHT" "$A" "$B" "ots" "She speaks angrily" "ots_A_to_B" "hair softly swaying, subtle breathing"
shot "$BG_LEFT" "$B" "$A" "ots" "She speaks in a friendly manner" "ots_B_to_A" "hair softly swaying, relaxed posture"

echo "=== CLOSEUPS & REACTIONS ==="
shot "$BG_LEFT" "$A" "$A" "closeup" "She smiles happily." "reaction_A" "eyes blinking naturally, subtle head tilt"
shot "$BG_RIGHT" "$B" "$B" "closeup" "She frowns unhappily." "reaction_B" "eyes blinking naturally, soft exhale"

echo "=== SINGLES & PROFILES ==="
shot "$BG_LEFT" "$A" "$A" "profile_right" "She points to something out of frame." "profile_A" "hair gently swaying, arm relaxed"
shot "$BG_LEFT" "$A" "$A" "medium" "She poses like a model." "single_A" "fabric rippling softly, subtle stance shift"
shot "$BG_RIGHT" "$B" "$B" "profile_left" "She looks up above her." "profile_B" "hair gently swaying, subtle head lift"
shot "$BG_RIGHT" "$B" "$B" "medium" "She poses like an idol." "single_B" "fabric rippling softly, subtle breathing"

echo "✅ All shots generated into '$OUTDIR/'"
