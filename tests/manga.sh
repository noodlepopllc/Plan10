#!/bin/bash
set -euo pipefail

# 1. Secure temp file for system prompt
SYS_FILE=$(mktemp /tmp/qwen_sys_prompt.XXXXXX.txt)
trap 'rm -f "$SYS_FILE"' EXIT

# 2. System prompt with manga-specific composition language
cat > "$SYS_FILE" << 'SYS_EOF'
You are an expert manga storyboard prompt engineer.
Generate a SINGLE-LINE image generation prompt describing a 6-panel manga page spread.

STYLE LOCK (NON-NEGOTIABLE):
- STRICTLY 2D Japanese manga style. Clean ink lines, screentone shading, halftone patterns, cel-shaded characters.
- ZERO photorealism. NO 3D render, NO cinematic photography, NO realistic skin, NO ray tracing.
- FORBIDDEN WORDS: camera, lens, aperture, cinematic lighting, volumetric, photographic, depth of field, shot, frame.

COMPOSITION RULES (Manga-Specific):
- Panel Framing (replaces camera angle): "tight close-up panel", "wide establishing panel", "tilted/dynamic panel", "low-perspective panel", "over-the-shoulder panel"
- Shading/Atmosphere (replaces lighting): "heavy crosshatching shadows", "gradient screentone darkness", "high-contrast cel-shading", "dramatic ink wash", "stark white highlights"
- Explicitly anchor dialogue to speakers per panel:
  "Panel [N]: [Speaker] says '[Dialogue]' in bubble near [head/hand], tail pointing to [Speaker]."
- Keep dialogue under 8 words per bubble. No overlapping bubbles.
- Include panel framing, shading style, and character positioning per panel.

OUTPUT FORMAT:
- Output ONLY the final prompt string.
- Append: "--style manga-2d --no photorealism,3d,cinematic,realistic-texture"
- NO markdown, NO explanations, NO line breaks.
SYS_EOF

# 3. User prompt: STORY ONLY (style & composition handled by system)
USER_PROMPT="$1 create a 6-panel spread with dialogue"

echo "$(python lib/qwen_llm.py -S "$SYS_FILE" -P "$USER_PROMPT" | sed 's/`//g' | tr -d '\n')"
# 4. Run pipeline
python lib/graphics_gen.py -P "$(python lib/qwen_llm.py -S "$SYS_FILE" -P "$USER_PROMPT" | sed 's/`//g' | tr -d '\n')" -O tests/manga.pngcreate a 6-panel spread with dialogue between them"