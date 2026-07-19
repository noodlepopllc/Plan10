#!/bin/bash
set -euo pipefail
#set -x   # debug
set -euo pipefail

if [ ! -d "tests/$1" ]; then 
    python tests/character_builder.py -D -N $1
fi

if [ ! -d "tests/$2" ]; then
    python tests/character_builder.py -D -N $2
fi

if [ ! -d "tests/$1_$2" ]; then
   python tests/personsV2.py $1 $2
fi

# ────────────────────────────────────────────────
# Environment / setup
# ────────────────────────────────────────────────
#source ~/.bashrc

#eval "$(conda shell.bash hook)"
#conda activate plan10

OUTDIR="tests/$1_$2"

mkdir -p "$OUTDIR"

BG="$OUTDIR/location.png"
BG_REV="$OUTDIR/location_reverse.png"
A="$OUTDIR/char1.png"
B="$OUTDIR/char2.png"
BG_LEFT="$OUTDIR/location_left.png"
BG_RIGHT="$OUTDIR/location_right.png"

python lib/config.py -R

source .env
echo "DEBUG: HEIGHT='$HEIGHT' WIDTH='$WIDTH'"

HEIGHT=$HEIGHT
WIDTH=$WIDTH

SEED=${SEED:-$RANDOM}
echo "🎲 Seed: $SEED | Date: $(date)" > "$OUTDIR/run_manifest.txt"

# ────────────────────────────────────────────────
# Shot wrapper
# ────────────────────────────────────────────────
shot() {
    local bg="$1" char="$2" shot_type="$3" action="$4" out_suffix="$5" vid_prompt="$6"
    local out="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}.png"
    local out_vid="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}.mp4"

    if [[ ! -f "$out" ]]; then
        echo "🎨 T2I: $out_suffix"
        python lib/compositor.py \
            -B "$bg" \
            -C "$char" \
            -S "$shot_type" \
            -A "$action" \
            -O "$out" \
            || { echo "❌ Compositor failed: $out_suffix"; exit 1; }

        touch "$out"

        #echo "🎬 I2V: $out_suffix"
        #python lib/image_to_video.py \
        #    -P "$vid_prompt" \
        #    -I "$out" \
        #    -O "$out_vid" \
        #    || { echo "❌ I2V failed: $out_suffix"; exit 1; }

        #echo "✅ $out_suffix | T2I: $action | I2V: $vid_prompt" >> "$OUTDIR/run_manifest.txt"
        echo "✅ $out_suffix | T2I: $action >> "$OUTDIR/run_manifest.txt"
    else
        echo "⏭️ Skipping $out_suffix (exists)"
    fi
}

# ────────────────────────────────────────────────
# Dialog → speech-to-video wrapper
# ────────────────────────────────────────────────
dialog() {
    local dialog_text="$1" voice="$2" action="$3" out_suffix="$4"
    local closeup="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}.png"
    local voice_vid="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}_voice.mp4"

    if [[ ! -f "$voice_vid" ]]; then
        echo "🎨 S2V: $voice_vid"
        python lib/speech_to_video.py \
            -P "$action" \
            -I "$closeup" \
            -T "$dialog_text" \
            -A "$voice" \
            -O "$voice_vid" \
            || { echo "❌ S2V failed: $voice_vid"; exit 1; }

        echo "✅ S2V: $voice_vid" >> "$OUTDIR/run_manifest.txt"
    else
        echo "⏭️ Skipping $voice_vid (exists)"
    fi
}

# ────────────────────────────────────────────────
# Two-person OTS + medium + closeup pipeline
# ────────────────────────────────────────────────
two_person() {
    local bg="$1" char_fg="$2" char_bg="$3" shot_type="$4" action="$5" out_suffix="$6" vid_prompt="$7"
    local closeup="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}.png"
    local medium="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}_medium.png"
    local out="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}_ots.png"
    local removed_vid="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}_removed.mp4"
    local out_vid="$OUTDIR/${WIDTH}_${HEIGHT}_${out_suffix}_ots.mp4"

    # Medium of background character
    if [[ ! -f "$medium" ]]; then
        echo "🎨 T2I: $medium"
        python lib/compositor.py \
            -B "$bg" \
            -C "$char_bg" \
            -S "medium" \
            -A "$action" \
            -O "$medium" \
            || { echo "❌ Compositor failed: $out_suffix (medium)"; exit 1; }
    fi

    # OTS with foreground + medium background
    if [[ ! -f "$out" ]]; then
        echo "🎨 T2I: $out_suffix (OTS)"
        python lib/compositor.py \
            -B "$bg" \
            -C "$char_fg" \
            -C "$medium" \
            -S "$shot_type" \
            -A "$action" \
            -O "$out" \
            || { echo "❌ Compositor failed: $out_suffix (ots)"; exit 1; }

        touch "$out"

        echo "🎬 I2V Exits frame: $out_suffix"
        python lib/image_to_video.py \
          -P "Over the entire duration, the woman with her back to the camera in the foreground slowly walks forward and completely out of frame in a smooth, continuous motion, no cuts, no teleporting. The woman in the background steps forward. The final frame should match the second image, with only the background woman visible." \
          -I "$out" \
          -I "$medium" \
          -O "$removed_vid" \

        echo "🎬 I2V Closeup: $out_suffix"
        python lib/image_to_video.py \
            -P "She moves forward as the camera zooms in on her face. $vid_prompt" \
            -I "$medium" \
            -I "$closeup" \
            -O "$out_vid" \
            || { echo "❌ I2V failed: $out_suffix (closeup)"; exit 1; }

        echo "✅ $out_suffix | T2I: $action | I2V: $vid_prompt" >> "$OUTDIR/run_manifest.txt"
    else
        echo "⏭️ Skipping $out_suffix (ots exists)"
    fi
}

# ────────────────────────────────────────────────
# Emotion Action Table (diffusion‑friendly verbs)
# ────────────────────────────────────────────────
EMOTIONS=(
  "She maintains a neutral expression."
  "She smiles softly."
  "She smiles brightly."
  "She opens her mouth in surprise."
  "Her eyes widen in shock."
  "She frowns slightly."
  "She frowns deeply."
  "She tilts her head in confusion."
  "She glances away nervously."
  "She smirks subtly."
  "She stares intensely."
  "She looks embarrassed, gaze lowering."
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

# ─── BACKGROUNDS ───
echo "=== BACKGROUNDS ==="
if [ ! -f "$BG_LEFT" ]; then 
    python lib/compositor.py -B $BG -Z "left" -O "$BG_LEFT" -R 
fi

if [ ! -f "$BG_RIGHT" ]; then 
    python lib/compositor.py -B $BG -Z "right" -O "$BG_RIGHT" -R
fi

VOICE1="$OUTDIR/char1.wav"
VOICE2="$OUTDIR/char2.wav"

# ────────────────────────────────────────────────
# Generate base voices
# ────────────────────────────────────────────────

if [[ ! -f $VOICE1 ]]; then
python lib/dialog.py \
  -I "female, young adult, moderate pitch, canadian accent" \
  -O "$VOICE1"
fi

if [[ ! -f $VOICE2 ]]; then
python lib/dialog.py \
  -I "female, young adult, high pitch, portuguese accent" \
  -O "$VOICE2"
fi

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
    shot "$BG_RIGHT" "$A" "closeup" "$EMO" "char1_emotion_$i" "$VID_MICRO"

    # dialog-driven video from closeup
    dialog "$DIALOG" "$VOICE1" "$EMO" "char1_emotion_${i}"

    # two-person OTS + motion
    #two_person "$BG_RIGHT" "$B" "$A" "ots" "$EMO" "char1_emotion_${i}" "$VID_MICRO"

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
    shot "$BG_LEFT" "$B" "closeup" "$EMO" "char2_emotion_$i" "$VID_MICRO"

    # dialog-driven video from closeup
    dialog "$DIALOG" "$VOICE2" "$EMO" "char2_emotion_${i}"

    # two-person OTS + motion
    #two_person "$BG_LEFT" "$A" "$B" "ots" "$EMO" "char2_emotion_${i}" "$VID_MICRO"

    ((++i))
done

echo "✅ Emotion closeups generated in '$OUTDIR/'"

