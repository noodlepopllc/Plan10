#!/bin/bash
OUTDIR="tests/$1"
BG="$OUTDIR/location.png"
A="$OUTDIR/char1.png"
B="$OUTDIR/char2.png"
BG_REV="$OUTDIR/location_reverse.png"
HEIGHT=832
WIDTH=480

# Generate a random seed (0–32767)
SEED=$RANDOM


# ─── FLEXIBLE WRAPPERS ───
shot_2char() { # <bg> <charA> <charB> <shot> <gaze> <mood> <exprA> <exprB> <out>
    local out="$OUTDIR/${WIDTH}_${HEIGHT}_$9.png"
    if [[ ! -f "$out" ]]; then
        python lib/compositor.py -BG "$1" -CHARS "$2" -CHARS "$3" \
            -SHOT "$4" -GAZE "$5" -T "$6" -EXPR "$7" -EXPR "$8" -Z "cute idol pose" -Z "cute idol pose" \
            -O "$out" -E $SEED -H $HEIGHT -W $WIDTH
        touch "$out"
    else
        echo "⏭️ Skipping $out (already exists)"
    fi
}

# ─── FLEXIBLE WRAPPERS ───
shot_OTS() { # <bg> <charA> <charB> <shot> <gaze> <mood> <exprA> <exprB> <out>
    local out="$OUTDIR/${WIDTH}_${HEIGHT}_$9.png"
    if [[ ! -f "$out" ]]; then
        python lib/compositor.py -BG "$1" -CHARS "$2" -CHARS "$3" \
            -SHOT "$4" -GAZE "$5" -T "$6" -EXPR "$7" -EXPR "$8" \
            -O "$out" -E $SEED -H $HEIGHT -W $WIDTH
        touch "$out"
    else
        echo "⏭️ Skipping $out (already exists)"
    fi
}

shot_1char() { # <bg> <char> <shot> <gaze> <mood> <expr> <out>
    local out="$OUTDIR/${WIDTH}_${HEIGHT}_$7.png"
    if [[ ! -f "$out" ]]; then
        python lib/compositor.py -BG "$1" -CHARS "$2" \
            -SHOT "$3" -GAZE "$4" -T "$5" -EXPR "$6" \
            -O "$out" -E $SEED -H $HEIGHT -W $WIDTH
        touch "$out"
    else
        echo "⏭️ Skipping $out (already exists)"
    fi
}


# ─── SHOTS ───
echo "=== MASTER ==="
shot_2char "$BG" "$A" "$B" "two_shot_close" "at_each_other" "" "neutral" "neutral" "master_close"
shot_2char "$BG_REV" "$B" "$A" "two_shot_close" "at_each_other" "" "neutral" "neutral" "master_close_rev"

echo "=== OVER-SHOULDER ==="
shot_OTS "$BG_REV" "$A" "$B" "over_shoulder" "a_to_b" "" "" "neutral" "ots_A_to_B"
shot_OTS "$BG" "$B" "$A" "over_shoulder" "a_to_b" "" "" "neutral" "ots_B_to_A"

echo "=== CLOSEUPS & REACTIONS ==="
shot_1char "$BG" "$A" "closeup_single" "a_to_b" "determined" "determined" "reaction_A"
shot_1char "$BG_REV" "$B" "closeup_single" "b_to_a" "angry" "angry" "reaction_B"

echo "=== SINGLES & PROFILES ==="
shot_1char "$BG" "$A" "profile_single_right" "off_camera" "" "neutral" "profile_A_right"
shot_1char "$BG" "$A" "medium_single" "forward" "" "neutral" "single_A"
shot_1char "$BG" "$B" "profile_single_left" "off_camera" "" "neutral" "profile_B_left"
shot_1char "$BG_REV" "$B" "medium_single" "forward" "" "neutral" "single_B"

echo "✅ All shots generated into $OUTDIR/"

