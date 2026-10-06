#!/bin/bash

uv run config -R

source .env

#set -euo pipefail
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

SEED=${SEED:-$RANDOM}
echo "🎲 Seed: $SEED | Date: $(date)" > "$OUTDIR/run_manifest.txt"
VISION_BACKEND="qwen" 

shot() {
    local bg="$1" char1="$2" char2="$3" shot_type="$4" action="$5" out_suffix="$6" vid_prompt="$7"
    local out="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}.png"
    local out_vid="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}.mp4"

    if [[ ! -f "$out" ]]; then
        echo "🎨 Generating T2I: $out_suffix"
        
        if [ "$char1" = "$char2" ]; then 
            # Always pass both chars. Your Python patch handles routing/ignoring.
            uv run qwen21 --composite -B "$bg" --chars "$char1" \
                -S "$shot_type" -A "$action" \
                -O "$out" -W $WIDTH -H $HEIGHT
        else
                    # Always pass both chars. Your Python patch handles routing/ignoring.
            uv run qwen21 --composite -B "$bg" --chars "$char1" --chars "$char2" \
                -S "$shot_type" -A "$action" \
                -O "$out" -W $WIDTH -H $HEIGHT
        fi
            
        touch "$out"  # ✅ Refreshes OS thumbnail cache

        if [ "$char1" = "$char2" ]; then 

            uv run video_creator -I "$out" -R "$char1" -C "$vid_prompt" -O "$OUTDIR/$out_suffix" -D 8 -M
        else
            uv run video_creator -I "$out" -R "$char1" -R "$char2" -C "$vid_prompt" -O "$OUTDIR/$out_suffix" -D 8 -M
        fi
        uv run video_runner -O "$OUTDIR/$out_suffix" -M
        uv run director -I "$OUTDIR/$out_suffix/beat_001_script.txt" -O "$OUTDIR" -W $WIDTH -H $HEIGHT -S 8 --wangp
        cp $OUTDIR/$out_suffix/beat_001_script.mp4 $out_vid
        
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
shot "$BG" "$A" "$B" "two_shot" "The char1 and char2 face each other." "master_close" "hair gently swaying, subtle weight shift"

echo "=== OVER-SHOULDER ==="
shot "$BG_RIGHT" "$A" "$B" "ots" "the two characters face each other" "ots_A_to_B" "char1 back is to the viewer, char2 faces them hair softly swaying, subtle breathing"
shot "$BG_LEFT" "$B" "$A" "ots" "the two characters face each other" "ots_B_to_A" "char1 back is to the viewer, char2 faces them hair softly swaying, relaxed posture"

echo "=== CLOSEUPS & REACTIONS ==="
shot "$BG_LEFT" "$A" "$A" "closeup" "char1 smiles happily." "reaction_A" "eyes blinking naturally, subtle head tilt"
shot "$BG_RIGHT" "$B" "$B" "closeup" "char1 frowns unhappily." "reaction_B" "eyes blinking naturally, soft exhale"

echo "=== SINGLES & PROFILES ==="
shot "$BG_LEFT" "$A" "$A" "profile_right" "char1 points to something out of frame." "profile_A" "hair gently swaying, arm relaxed"
shot "$BG_LEFT" "$A" "$A" "medium" "char1 poses like a model." "single_A" "fabric rippling softly, subtle stance shift"
shot "$BG_RIGHT" "$B" "$B" "profile_left" "char1 looks up above her." "profile_B" "hair gently swaying, subtle head lift"
shot "$BG_RIGHT" "$B" "$B" "medium" "char1 poses like an idol." "single_B" "fabric rippling softly, subtle breathing"

echo "✅ All shots generated into '$OUTDIR/'"

# ────────────────────────────────────────────────
# Emotion Action Table (diffusion‑friendly verbs)
# ────────────────────────────────────────────────
EMOTIONS=(
  " maintains a neutral expression."
  " smiles softly."
  " smiles brightly."
  " opens her mouth in surprise."
  " eyes widen in shock."
  " frowns slightly."
  " frowns deeply."
  " tilts her head in confusion."
  " glances away nervously."
  " smirks subtly."
  " stares intensely."
  " looks embarrassed, gaze lowering."
)

DIALOG_LINES=(
  "I’m here, just taking things in and staying calm today."
  "It’s really nice being here with you right now."
  "I can’t help smiling; everything feels genuinely good today."
  "Wait—hold on, I didn’t expect that to happen at all."
  "No way… seriously? I can’t believe what I’m seeing here."
  "Something feels off, but I’m trying to understand it clearly."
  "This isn’t right, and I’m tired of pretending otherwise now."
  "I’m trying to follow you, but none of this makes sense."
  "I’m not sure about this… something doesn’t feel completely safe."
  "Oh really? That’s the best you can offer today?"
  "Say it clearly. I want the truth without hesitation now."
  "Please don’t look at me like that… it’s embarrassing honestly."
)

# Micro‑motion for I2V
VID_MICRO="eyes blinking naturally, subtle breathing"

# ────────────────────────────────────────────────
# EMOTION TESTS — CHAR 1
# ────────────────────────────────────────────────
echo "=== EMOTION TEST: CHAR 1 ==="
i=0
for EMO in "${EMOTIONS[@]}"; do
    DIALOG="${DIALOG_LINES[$i]}"
    echo "EMO='$EMO'"
    echo "DIALOG='$DIALOG'"

    # closeup still
    shot "$BG_LEFT" "$A" "$A" "closeup" "$EMO" "char1_emotion_$i" "char1 speaks \"$DIALOG\" $VID_MICRO"
    shot "$BG_LEFT" "$A" "$A" "medium" "$EMO" "char1_emotion_medium_$i" "Camera push-in on char1 as char1 speaks \"$DIALOG\" $VID_MICRO"

    # two-person OTS + motion
    shot "$BG_RIGHT" "$B" "$A" "ots" "$EMO" "char1_emotion_ots_${i}" "Camera crash-in on char2 as char2 speaks \"$DIALOG\" $VID_MICRO"

    ((++i))
done

# ────────────────────────────────────────────────
# EMOTION TESTS — CHAR 2
# ────────────────────────────────────────────────
echo "=== EMOTION TEST: CHAR 2 ==="
i=0
for EMO in "${EMOTIONS[@]}"; do
    DIALOG="${DIALOG_LINES[$i]}"
    echo "EMO='$EMO'"
    echo "DIALOG='$DIALOG'"

    # closeup still
    shot "$BG_RIGHT" "$B" "$B" "closeup" "$EMO" "char2_emotion_$i" "char1 speaks \"$DIALOG\" $VID_MICRO"
    shot "$BG_RIGHT" "$B" "$B" "medium" "$EMO" "char2_emotion_medium_$i" "Camera Push-In on char1 as char1 speaks \"$DIALOG\" $VID_MICRO"


    # two-person OTS + motion
    shot "$BG_LEFT" "$A" "$B" "ots" "$EMO" "char2_emotion_ots_${i}" "Camera Crash-In on char2 as char2 speaks \"$DIALOG\" $VID_MICRO"

    ((++i))
done

echo "✅ Emotion closeups generated in '$OUTDIR/'"