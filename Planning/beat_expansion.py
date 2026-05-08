#!/usr/bin/env python3
import sys

def main():
    story = sys.argv[1]
    beats = int(sys.argv[2]) if len(sys.argv) > 2 else 14

    prompt = f"""Generate exactly {beats} visual beats for this scene.
OUTPUT FORMAT (STRICT):
[ACTION] <plain english description of what happens visually>

RULES:
- Output ONLY the {beats} lines. NO markdown, NO numbers, NO bullets, NO extra text.
- Each line = exactly one image/video clip (~3-5s).
- Describe ONLY visible screen content. No camera jargon.
- Include character positioning, spatial changes, and facial/body expressions.
- Start with a clear physical establishing beat. Alternate focus naturally.
- Match scene context: {story}

EXAMPLE:
[ACTION] Both characters step into frame, standing side-by-side in torchlight
[ACTION] Teacher adjusts glasses, smiles warmly toward viewer
[ACTION] Assistant tilts head, purrs softly, shifts weight forward
[ACTION] Teacher crosses arms, leans back against stone wall
[ACTION] Assistant looks down, fidgets with dress hem, blushes

BEGIN OUTPUT:"""
    print(prompt)

if __name__ == "__main__":
    main()