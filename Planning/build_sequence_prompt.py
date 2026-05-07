#!/usr/bin/env python3
import sys

SCHEMA = '''
{
  "metadata": {
    "story_context": "string",
    "execution_note": "Pipeline reorders beats by generation dependencies (assets → composites → voices → dialog → video). Original narrative sequence is preserved in the 'beats' array via the 'order' field.",
    "flow_rationale": "string"
  },
  "beats": [
    {
      "order": "number",
      "type": "dialog or action",
      "visible_chars": "array of 1 or 2 integers",
      "text": "string or null",
      "facial_action": "string or null",
      "starting_pose": "string or null",
      "motion_prompt": "string or null",
      "shot_type": "string",
      "motion_type": "string",
      "motion_target": "string or null"
    }
  ]
}
'''

def main():
    registry_path = sys.argv[1]
    story_path = sys.argv[2]

    with open(registry_path) as f:
        registry = f.read()
    with open(story_path) as f:
        story = f.read().strip()

    prompt = f"""[ARC:setup→conflict→resolution]
{registry}

### SCENE CONTEXT
{story}

### NARRATIVE SEQUENCING RULES
- Arc: SETUP(1-2) → CONFLICT(3-6) → RESOLUTION(7-8)
- Beat 1 MUST be ACTION showing physical segue. DIALOG starts at Beat 2.
- Never start with exposition. Show → react → speak.

### MOTION CHAINING & SEGMENTATION (CRITICAL)
- AI video models ONLY handle 1-2 simple kinematic actions per 3s clip.
- NEVER chain 3+ actions in a single motion_prompt. SPLIT them across sequential beats.
- CONTINUITY LOCK: Beat N+1's starting_pose MUST exactly match the physical end-state of Beat N's motion_prompt.
- Keep visible_chars IDENTICAL across segmented beats unless a character physically enters/exits frame.
- Example of correct segmentation:
  Beat 4: starting_pose="standing facing toaster", motion_prompt="looks down at toaster, blinks"
  Beat 5: starting_pose="still facing toaster", motion_prompt="turns head left toward door"
  Beat 6: starting_pose="facing door", motion_prompt="takes two steps forward, shoulders relax"

### FIELD-LEVEL ENFORCEMENT
🔴 DIALOG BEATS:
  • len(visible_chars)==1 → shot_type = "closeup"
  • starting_pose = null, motion_prompt = null, motion_type = "static"
  • ONLY text and facial_action are filled.

🔵 ACTION BEATS:
  • text = null, facial_action = null
  • starting_pose REQUIRED (exact FRAME 0 state)
  • motion_prompt REQUIRED (max 2 verbs, max 10 words)
  • len(visible_chars)==1 → shot_type MUST be "closeup", "medium", "profile_left", or "profile_right"
  • len(visible_chars)==2 → shot_type MUST be "ots" (over-the-shoulder: camera behind one character, focusing on the other) or "two_shot" (both characters fully in frame)
  • NEVER use "ots" or "two_shot" for single-character action beats.

### OUTPUT CONSTRAINTS
- Output ONLY raw JSON. NO markdown, NO backticks, NO explanations.
- Use literal JSON null, not string "null".
- Start with {{ and end with }}. Nothing else.

SCHEMA:
{SCHEMA}

Respond ONLY with the JSON:"""

    print(prompt)

if __name__ == "__main__":
    main()
