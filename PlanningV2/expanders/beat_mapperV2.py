#!/usr/bin/env python3
import sys

SCHEMA = '''
{
  "metadata": { "story_context": "string", "execution_note": "Pipeline reorders beats by generation dependencies (assets → composites → voices → dialog → video). Original narrative sequence is preserved in the 'beats' array via the 'order' field.", "flow_rationale": "string" },
  "beats": [
    { "order": "number", "type": "dialog or action", "visible_chars": "array of 1 or 2 integers", "text": "string or null", "facial_action": "string or null", "starting_pose": "string or null", "motion_prompt": "string or null", "shot_type": "string", "motion_type": "string", "motion_target": "string or null", "base_composite": "string or null", "props": "string or null" }
  ]
}
'''

def main():
    registry_path, beats_path, dialog_path, story_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    
    with open(registry_path) as f: registry = f.read().strip()
    with open(beats_path) as f: beats_text = f.read().strip()
    with open(dialog_path) as f: dialog_text = f.read().strip()
    with open(story_path) as f: story = f.read().strip()

    # Explicitly mark empty inputs so the LLM knows what's missing
    action_input = beats_text if beats_text else "NONE"
    dialog_input = dialog_text if dialog_text else "NONE"

   prompt = f"""Convert the provided inputs into a strict JSON sequence.

   TOTAL BEATS REQUIRED: {len(action_input.splitlines()) + len(dialog_input.splitlines())} (Preserve EVERY line from BOTH inputs)

   INPUT 1 - ACTION BEATS:
   {action_input}

   INPUT 2 - DIALOG LINES:
   {dialog_input}

   ASSET REGISTRY:
   {registry}

   SCENE CONTEXT:
   {story}

   STRICT GENERATION RULES:
   0. INPUT SCOPE LOCK: If an input is "NONE", output ZERO beats of that type. NEVER invent missing content.
   1. EXACT MERGE & COUNT ENFORCEMENT: You MUST output EXACTLY {len(action_input.splitlines()) + len(dialog_input.splitlines())} beats. Preserve EVERY action line and EVERY dialog line. NEVER truncate, skip, pair 1:1, or replace. Arrange all beats chronologically based on SCENE CONTEXT. Assign sequential "order" values 1 through {len(action_input.splitlines()) + len(dialog_input.splitlines())}.
   2. CHRONOLOGICAL FLOW: Typical sequence: establishing action → dialog → reaction action → dialog → ... All beats must appear. If dialog logically follows an action, place it immediately after. Maintain narrative causality.
   3. TYPE-SCOPED FIELD RULES:
      • type == "dialog": text = spoken words only. facial_action = mood + expression. visible_chars = [speaker_index]. shot_type = "closeup"|"medium_closeup". starting_pose = "neutral speaking posture". motion_prompt = null, motion_type = "static", props = null, motion_target = null.
      • type == "action": text = null, facial_action = null. visible_chars = [1], [2], or [1,2]. starting_pose = initial stance. motion_prompt = "Char: [verb], [verb]". motion_type = "dynamic"|"subtle"|"static". shot_type = match visible count. props = extract explicit objects or null.
   4. CHARACTER MAPPING: Match names/descriptions to EXACT registry indices [1] or [2]. Do not invent aliases.
   5. FIELD COMPLETENESS: EVERY beat must contain ALL schema keys. Set unused fields to null. NEVER omit keys.
   6. OUTPUT VERIFICATION: Before finishing, count your beats. If fewer than {len(action_input.splitlines()) + len(dialog_input.splitlines())}, your output is incomplete. Continue until all are represented.
   7. OUTPUT: ONLY valid JSON matching the schema. NO markdown, NO explanations.

   SCHEMA:
   {SCHEMA}

   Respond ONLY with the JSON:"""

    print(prompt)

if __name__ == "__main__":
    main()