#!/usr/bin/env python3
import sys, json

# 🔑 CLOSED MOOD VOCABULARY (Source of truth for asset routing)
ALLOWED_MOODS = [
    "neutral", "confident", "skeptical", "encouraging", "curious", 
    "supportive", "shy", "reassuring", "amused", "eager", 
    "nervous", "patient", "relieved", "stern", "defensive", 
    "frustrated", "overwhelmed", "playful"
]
MOOD_LIST_STR = ", ".join(ALLOWED_MOODS)

def main():
    scene_path = sys.argv[1]
    registry_path = sys.argv[2]
    beats = int(sys.argv[3]) if len(sys.argv) > 3 else 14

    with open(registry_path) as f:
        registry = json.load(f)

    chars = registry.get("characters", [])
    if len(chars) < 2:
        print("Error: Registry must contain at least 2 characters.", file=sys.stderr)
        sys.exit(1)

    name1 = chars[0]["name"]
    name2 = chars[1]["name"]
    env = registry.get("environment", "")

    with open(scene_path) as f:
        scene = f.read().strip()

    prompt = f"""Generate exactly {beats} dialog/reaction beats for this scene.
OUTPUT FORMAT (STRICT):
{name1}: <mood> <spoken line OR [nonverbal headshot action]>
{name2}: <mood> <spoken line OR [nonverbal headshot action]>

RULES:
- Output ONLY the {beats} lines. NO markdown, NO numbers, NO bullets, NO extra text.
- <mood> MUST be EXACTLY ONE word from this list: {MOOD_LIST_STR}
- NEVER invent new moods. If emotional nuance is needed, convey it through word choice, punctuation, or pacing.
- One utterance = one line. Multiple consecutive lines for the same character are allowed when the scene demands it.
- Spoken lines: natural conversational length. Let TTS handle pacing (~2-5s per line).
- Nonverbal: brackets ONLY for head/eye/mouth micro-movements. Example: [nods slowly], [averts gaze, exhales]
- At least 30% of beats should be bracketed reactions.
- Flow naturally. Match scene context: {scene}
- Setting: {env}

EXAMPLE:
{name1}: confident [adjusts glasses] Welcome to our Vietnamese lesson.
{name2}: playful [tilts head] Chào mừng đến với lớp học.
{name1}: thoughtful That means welcome to class.
{name2}: shy [fidgets with dress hem]

BEGIN OUTPUT:"""
    print(prompt)

if __name__ == "__main__":
    main()