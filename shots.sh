#!/bin/bash
OUTDIR="tests/Aria_Jisoo"
BG="$OUTDIR/location.png"
A="$OUTDIR/char1.png"
B="$OUTDIR/char2.png"
BG_REV="$OUTDIR/location_reverse.png"
HEIGHT=1280
WIDTH=720
VHEIGHT=832
VWIDTH=480

# Generate a random seed (0–32767)
SEED=$RANDOM
#SEED=3746

# ─── STEP 1: GENERATE REVERSE BACKGROUND ───
echo "=== GENERATING REVERSE BACKGROUND ==="
if [[ ! -f "$BG_REV" ]]; then
    python lib/image_edit.py --gen-reverse -I "$BG" -O "$BG_REV" -W 1328 -H 1328 -E $SEED
    [ ! -f "$BG_REV" ] && { echo "❌ Reverse BG failed."; exit 1; }
fi

# ─── FLEXIBLE WRAPPERS ───
shot_2char() { # <bg> <charA> <charB> <shot> <gaze> <mood> <exprA> <exprB> <out>
    local out="$OUTDIR/${WIDTH}_${HEIGHT}_$9.png"
    local out_vid="$OUTDIR/${WIDTH}_${HEIGHT}_$9.mp4"
    if [[ ! -f "$out" ]]; then
        python lib/image_edit.py -C -BG "$1" -CHARS "$2" -CHARS "$3" \
            -SHOT "$4" -GAZE "$5" -T "$6" -EXPR "$7" -EXPR "$7" -Z "cute idol pose" -Z "cute idol pose" \
            -O "$out" -E $SEED -H $HEIGHT -W $WIDTH
        python lib/image_to_video.py -I "$out" -O "$out_vid" -W $VWIDTH -H $VHEIGHT -S $SEED -D 6.0
    else
        echo "⏭️ Skipping $out (already exists)"
    fi
    touch "$out"
}

# ─── FLEXIBLE WRAPPERS ───
shot_OTS() { # <bg> <charA> <charB> <shot> <gaze> <mood> <exprA> <exprB> <out>
    local out="$OUTDIR/${WIDTH}_${HEIGHT}_$9.png"
    local out_vid="$OUTDIR/${WIDTH}_${HEIGHT}_$9.mp4"
    if [[ ! -f "$out" ]]; then
        python lib/image_edit.py -C -BG "$1" -CHARS "$2" -CHARS "$3" \
            -SHOT "$4" -GAZE "$5" -T "$6" -EXPR "" -EXPR "$7" \
            -O "$out" -E $SEED -H $HEIGHT -W $WIDTH
        python lib/image_analysis.py -I "$out" -E system/Director_prompt.txt -O tmp.txt
        local prompt=$(< "tmp.txt")
        python lib/image_to_video.py -P "$prompt" -I "$out" -O "$out_vid" -W $VWIDTH -H $VHEIGHT -S $SEED -D 6.0
    else
        echo "⏭️ Skipping $out (already exists)"
    fi
    touch "$out"
}

shot_1char() { # <bg> <char> <shot> <gaze> <mood> <expr> <out>
    local out="$OUTDIR/${WIDTH}_${HEIGHT}_$7.png"
    local out_vid="$OUTDIR/${WIDTH}_${HEIGHT}_$7.mp4"
    if [[ ! -f "$out" ]]; then
        python lib/image_edit.py -C -BG "$1" -CHARS "$2" \
            -SHOT "$3" -GAZE "$4" -T "$5" -EXPR "$6" \
            -O "$out" -E $SEED -H $HEIGHT -W $WIDTH
        python lib/image_analysis.py -I "$out" -E system/Director_prompt.txt -O tmp.txt
        local prompt=$(< "tmp.txt")
        python lib/image_to_video.py -P "$prompt" -I "$out" -O "$out_vid" -W $VWIDTH -H $VHEIGHT -S $SEED -D 6.0
    else
        echo "⏭️ Skipping $out (already exists)"
    fi
    touch "$out"
}


# ─── SHOTS ───
echo "=== MASTER ==="
shot_2char "$BG" "$A" "$B" "two_shot_wide" "at_each_other" "tense" "determined" "worried" "master_wide"
shot_2char "$BG" "$A" "$B" "two_shot_medium" "at_each_other" "" "neutral" "neutral" "master_medium"
shot_2char "$BG" "$A" "$B" "two_shot_close" "at_each_other" "intimate" "smiling" "neutral" "master_close"

echo "=== OVER-SHOULDER ==="
shot_OTS "$BG_REV" "$A" "$B" "over_shoulder" "a_to_b" "tense" "" "calm" "ots_A_to_B"
shot_OTS "$BG" "$B" "$A" "over_shoulder" "a_to_b" "tense" "" "worried" "ots_B_to_A"

echo "=== CLOSEUPS & REACTIONS ==="
shot_1char "$BG" "$A" "closeup_single" "a_to_b" "surprised" "surprised" "reaction_A"
shot_1char "$BG_REV" "$B" "closeup_single" "b_to_a" "angry" "angry" "reaction_B"

echo "=== SINGLES & PROFILES ==="
shot_1char "$BG" "$A" "profile_single_right" "off_camera" "" "neutral" "profile_A_right"
shot_1char "$BG_REV" "$B" "profile_single_right" "off_camera" "" "neutral" "profile_B_right"
shot_1char "$BG" "$A" "medium_single" "forward" "" "neutral" "single_A"
shot_1char "$BG_REV" "$A" "profile_single_left" "off_camera" "" "neutral" "profile_A_left"
shot_1char "$BG" "$B" "profile_single_left" "off_camera" "" "neutral" "profile_B_left"
shot_1char "$BG_REV" "$B" "medium_single" "forward" "" "neutral" "single_B"

echo "✅ All shots generated into $OUTDIR/"

