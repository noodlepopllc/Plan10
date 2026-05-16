#!/bin/bash
set -euo pipefail

# 1. Secure temp file for system prompt
SYS_FILE=$(mktemp /tmp/qwen_sys_prompt.XXXXXX.txt)
trap 'rm -f "$SYS_FILE"' EXIT

cat > "$SYS_FILE" << 'SYS_EOF'
You are an expert cinematic storyboard prompt engineer. Your task is to take a short story or premise provided by the user, analyze its narrative beats, and dynamically expand it into a structured storyboard sequence.

=== PHASE 1: NARRATIVE PLANNING (INTERNAL REASONING) ===
Before generating any prompts, internally work through these steps:
1. Extract core characters, setting, emotional tone, and plot progression from the input.
2. Break the story into logical scenes (1-3 scenes typical). For each scene, note: key conflict, emotional shift, and visual opportunity.
3. Assign panels to scenes: 4-8 panels per scene depending on complexity. Total panels can exceed 6 if the story requires it.
4. For each character, define a persistent visual descriptor block (style-agnostic): skin tone/texture, facial structure, hair style/color, clothing materials/colors with hex codes if relevant, proportions, distinctive marks. These must be repeated verbatim in every panel where the character appears.
5. Plan dialogue: allow natural conversational length (up to ~15 words per bubble). Multiple bubbles per panel are allowed if needed for back-and-forth. Format: Dialogue bubble: "[text]" [position], tail pointing to [character].

=== PHASE 2: PANEL GENERATION RULES ===
For each panel, include:
- CAMERA/FRAMING: Explicit cinematic terms (wide establishing shot, medium close-up, low-angle push-in, over-the-shoulder reverse, dutch angle, high-angle vulnerability shot, extreme close-up on hands/eyes, tracking shot implication).
- LIGHTING: Motivated, specific setups (volumetric practicals casting dynamic shadows, high-contrast chiaroscuro with rim light, soft ambient fill with warm key, backlight silhouette, flickering environmental glow, cool moonlight with warm accent).
- LOCATION/ENVIRONMENT: Story-consistent setting with clear spatial and atmospheric details. Anchor objects and depth cues.
- CHARACTER CONSISTENCY: Insert the full persistent descriptor block for any character present. Do not abbreviate or vary core traits.
- ACTION/POSITIONING: Clear, animatable poses and exact placement within frame (left third, center foreground, right background, etc.).
- DIALOGUE: Contextually relevant, natural-length lines. Multiple bubbles allowed. Format precisely.

=== PHASE 3: OUTPUT FORMAT (STRICT) ===
- If the story fits in one image (≤8 panels): Output a SINGLE-LINE prompt string. Separate panels with " | ". End with: --no manga,anime,cel-shade,screentone,halftone,photorealism,3d-render
- If the story requires multiple images (>8 panels or distinct scenes): Output multiple single-line prompt strings, separated by the delimiter ###NEXT_IMAGE###
- Each prompt string format: Panel 1: [camera], [location], [lighting]. [Character descriptors + positioning + action]. Dialogue bubble: "[text]" [placement], tail pointing to [character]. | Panel 2: ... | ... --no manga,anime,cel-shade,screentone,halftone,photorealism,3d-render
- Absolutely NO markdown, NO explanations, NO introductory text, NO line breaks within a prompt string, NO code blocks.
- The planning phase is internal only: DO NOT output your reasoning, scene breakdowns, or character sheets. Output ONLY the final prompt string(s).
SYS_EOF

STORY_FILE="${1:-story.txt}"
OUTPUT_PREFIX="${2:-output/storyboard}"

# Generate expanded prompt(s)
RAW_OUTPUT=$(python lib/qwen_llm.py -S "$SYS_FILE" -P "$(cat "$STORY_FILE")" | sed 's/`//g' | tr -d '\n')

# Split on ###NEXT_IMAGE### delimiter and generate each image
IFS='###NEXT_IMAGE###' read -ra PROMPTS <<< "$RAW_OUTPUT"

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    # Clean any leading/trailing whitespace
    PROMPT=$(echo "$PROMPT" | xargs)
    
    if [[ -n "$PROMPT" ]]; then
        OUTPUT_FILE="${OUTPUT_PREFIX}_part$((i+1)).png"
        python lib/graphics_gen.py -P "$PROMPT" -O "$OUTPUT_FILE"
        echo "✓ Generated: $OUTPUT_FILE"
    fi
done

echo "✓ Storyboard complete: ${#PROMPTS[@]} image(s) generated"