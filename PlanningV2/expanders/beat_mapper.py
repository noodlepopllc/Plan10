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
0. INPUT SCOPE LOCK (CRITICAL): If an input is marked "NONE", you MUST output ZERO beats of that type. NEVER invent, hallucinate, or infer missing actions or dialog.
1. EVERY PROVIDED [ACTION] LINE → 1 JSON beat with type="action", text=null, facial_action=null.
2. DIALOG BEAT RULES (APPLY ONLY IF DIALOG INPUT IS PROVIDED):
   • text = ONLY the final spoken words. NEVER include "Speaker:", Mood, or [brackets].
   • facial_action = Mood + Action combined (e.g., "confident, adjusts glasses, smiles").
   • visible_chars = Speaker ONLY. "Teacher"/"Instructor"/"Wife" → [1]. "Assistant"/"Kitten"/"Husband" → [2].
   • shot_type = "closeup" (ALWAYS). NEVER use [1,2] for dialog beats.
   • motion_prompt = null, motion_type = "static".
3. ACTION CHARACTER IDENTIFICATION: Map descriptive phrases in [ACTION] lines to EXACT registry characters.
   • "teacher", "woman in blazer", "instructor", "glasses" → visible_chars: [1]
   • "assistant", "kitten dress", "red dress", "cat ears" → visible_chars: [2]
   • If both described → visible_chars: [1, 2]
4. ACTION MOTION EXTRACTION (APPLY ONLY IF ACTION INPUT IS PROVIDED):
   • Convert physical actions into EXPLICIT, TAGGED kinematic instructions.
   • Assign to motion_prompt: "Teacher: [verb], [verb] | Assistant: [verb], [verb]"
   • Set motion_type = "dynamic" (overt movement), "subtle" (micro-adjustments), or "static" (held pose only).
   • NEVER use pronouns. MAX 2 verbs per character.
5. SHOT TYPE (ACTIONS): len(visible_chars)==1 → "closeup"|"medium"|"profile_left"|"profile_right". len==2 → "two_shot"|"ots".
6. ASSET ROUTING: base_composite MUST be an EXACT registry alias. If no match, set null.
7. PROP EXTRACTION (ACTIONS ONLY):
   • Scan [ACTION] lines for physical objects/furniture explicitly mentioned.
   • Extract to "props" field: "glowing book, wooden table".
   • If none mentioned, set props: null. NEVER invent props.
8. ORDERING: Maintain exact narrative sequence. Do not skip, merge, or reorder.
9. OUTPUT ONLY RAW JSON. NO MARKDOWN. NO EXPLANATIONS.

SCHEMA:
{SCHEMA}
Respond ONLY with the JSON:"""

    print(prompt)

if __name__ == "__main__":
    main()