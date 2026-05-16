#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python beat_planner.py '<story>' '<location_desc>' <num_beats> '<char1,char2>'", file=sys.stderr)
        sys.exit(1)
        
    story = sys.argv[1]
    location = sys.argv[2] if len(sys.argv) > 2 else "neutral interior space"
    beats = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    chars = sys.argv[4].split(",") if len(sys.argv) > 4 else ["Character A", "Character B"]
    char_names = ", ".join(chars)

    prompt = f"""You are a Goal-Oriented Action Director for AI compositing.

CONTEXT:
- Story Context: {story}
- Location: {location}
- Characters: {char_names}
- Total Beats: {beats}

TASK:
1. Define a clear, location-motivated goal for each character based on the story.
2. Create an escalation arc: Establish → Challenge → Push → Shift/Resolve.
3. Output exactly {beats} beats that form a continuous action chain toward those goals.

OUTPUT FORMAT (STRICT - PIPE PARSEABLE):
[GOAL_A] <goal for first character>
[GOAL_B] <goal for second character>
[BEAT 1] [CHAR] <name> | [ACTION] <location-motivated action using visible props/staging> | [DURATION] <seconds> | [HOLD] <stable end pose>
[BEAT 2] ...
[BEAT {beats}] ...

RULES:
- Actions MUST reference location features (e.g., railing, counter, floor, wall, window).
- Each beat advances a goal, reacts to the other character, or shifts tension.
- Duration: 1.0-4.0s. HOLD must end with "mouth closed, head still, gaze <direction>".
- Escalate physically: proximity → posture shift → weight transfer → eye contact/gaze break.
- NO camera jargon. Describe ONLY visible screen content.
- Output ONLY the formatted lines. NO markdown, NO extra text, NO explanations.

BEGIN OUTPUT:"""
    print(prompt)

if __name__ == "__main__":
    main()