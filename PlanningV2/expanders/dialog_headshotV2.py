#!/usr/bin/env python3
import sys, json

ALLOWED_MOODS = [
    "neutral", "confident", "skeptical", "encouraging", "curious", 
    "supportive", "shy", "reassuring", "amused", "eager", 
    "nervous", "patient", "relieved", "stern", "defensive", 
    "frustrated", "overwhelmed", "playful"
]
MOOD_LIST_STR = ", ".join(ALLOWED_MOODS)

def main():
    if len(sys.argv) < 3:
        print("Usage: python dialog_gen.py scene.txt registry.txt beats [goal_a] [goal_b] [dynamic]", file=sys.stderr)
        sys.exit(1)

    scene_path = sys.argv[1]
    registry_path = sys.argv[2]
    beats = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    goal_a = sys.argv[4] if len(sys.argv) > 4 else "Understand the truth"
    goal_b = sys.argv[5] if len(sys.argv) > 5 else "Maintain control"
    dynamic = sys.argv[6] if len(sys.argv) > 6 else "competitive"

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
{name1}: <mood> <spoken line OR [nonverbal action]>
{name2}: <mood> <spoken line OR [nonverbal action]>

CONTEXT:
- Character A ({name1}) Goal: {goal_a}
- Character B ({name2}) Goal: {goal_b}
- Dynamic: {dynamic}
- Setting: {env}
- Scene: {scene}

RULES:
- Output ONLY the {beats} lines. NO markdown, NO numbers, NO bullets, NO extra text.
- <mood> MUST be EXACTLY ONE word from: {MOOD_LIST_STR}
- NEVER alternate mechanically. Turn-taking must follow goal pressure:
  • If pushing an agenda → consecutive lines allowed
  • If defending/overwhelmed → shorter lines or bracketed reactions
  • If power shifts → immediate reaction beat from other character
- Structure the arc naturally: Establish → Push/Counter → Peak/Turn → Resolve/Linger.
- At least 30% of beats must be bracketed nonverbal reactions: [nods slowly], [looks away, exhales], [tightens jaw], etc.
- Spoken lines: natural conversational length. Let TTS handle pacing.
- Dynamic pacing: {dynamic} → {
    'competitive': 'interruptions, sharp turns, shorter lines, defensive postures',
    'cooperative': 'overlapping support, longer explanations, shared pauses, reassuring gestures',
    'defensive': 'evasive answers, delayed reactions, frequent bracketed beats, hesitant pacing',
    'persuasive': 'rhetorical questions, steady eye contact, gradual escalation, confident framing'
  }.get(dynamic, 'balanced tension')
- Match goals: Every line either advances a goal, defends against pressure, or reacts to a shift.
- Setting influences tone and word choice naturally.

EXAMPLE:
{name1}: confident You knew about this from the start.
{name2}: defensive [crosses arms] I didn't have a choice.
{name1}: frustrated Then why didn't you tell me?
{name2}: [averts gaze, exhales] I was protecting you.
{name1}: [steps closer, voice drops] You lied instead.
{name2}: overwhelmed [rubs temple] Just... give me a minute.

BEGIN OUTPUT:"""
    print(prompt)

if __name__ == "__main__":
    main()