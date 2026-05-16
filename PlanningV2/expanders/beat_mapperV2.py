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
1. MERGE & SEQUENCING: When BOTH inputs are provided, interleave them into a single chronological array matching the SCENE CONTEXT. Preserve the relative order within each input. Typical flow: action/context → dialog → reaction/dialog → action. Assign sequential "order" values (1, 2, 3...) to the final merged array.
2. TYPE-SCOPED FIELD RULES:
   • If beat.type == "dialog": Apply ONLY these:
     - text = ONLY spoken words. Strip speaker labels, moods, brackets.
     - facial_action = Mood + expression from source (e.g., "confident, smiles thinly").
     - visible_chars = [speaker_index]. Map using registry aliases.
     - shot_type = "closeup" or "medium_closeup".
     - starting_pose = "neutral speaking posture" (or inherit if continuous).
     - motion_prompt = null, motion_type = "static", props = null, motion_target = null.
   • If beat.type == "action": Apply ONLY these:
     - text = null, facial_action = null
     - visible_chars = Map subjects to registry indices [1], [2], or [1,2].
     - starting_pose = Initial physical stance described.
     - motion_prompt = Explicit kinematic tags per character (e.g., "Sorceress: [reach_down], [grip_chin]").
     - motion_type = "dynamic" | "subtle" | "static"
     - shot_type = len(visible_chars)==1 → "closeup"|"medium"|"profile"; len==2 → "two_shot"|"ots"
     - props = Extract explicit objects mentioned. null if none.
3. CHARACTER MAPPING: Match names/descriptions in BOTH inputs to EXACT registry indices. Do not invent new identifiers.
4. FIELD COMPLETENESS: EVERY beat must contain ALL schema fields. Set unused fields to null. NEVER omit keys.
5. OUTPUT: ONLY valid JSON matching the schema. NO markdown, NO explanations.

SCHEMA:
{SCHEMA}
Respond ONLY with the JSON:"""

    print(prompt)

if __name__ == "__main__":
    main()