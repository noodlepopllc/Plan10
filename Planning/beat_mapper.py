#!/usr/bin/env python3
import sys

SCHEMA = '''
{
  "metadata": { "story_context": "string", "execution_note": "Pipeline reorders beats by generation dependencies (assets → composites → voices → dialog → video). Original narrative sequence is preserved in the 'beats' array via the 'order' field.", "flow_rationale": "string" },
  "beats": [
    { "order": "number", "type": "dialog or action", "visible_chars": "array of 1 or 2 integers", "text": "string or null", "facial_action": "string or null", "starting_pose": "string or null", "motion_prompt": "string or null", "shot_type": "string", "motion_type": "string", "motion_target": "string or null", "base_composite": "string or null" }
  ]
}
'''

def main():
    registry_path, beats_path, dialog_path, story_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    with open(registry_path) as f: registry = f.read()
    with open(beats_path) as f: beats_text = f.read()
    with open(dialog_path) as f: dialog_text = f.read()
    with open(story_path) as f: story = f.read().strip()

    prompt = f"""Convert the provided ACTION BEATS and DIALOG LINES into a strict JSON sequence.
      INPUT 1 - ACTION BEATS (MUST BECOME ACTION JSON BEATS):
      {beats_text}

      INPUT 2 - DIALOG LINES (MUST BECOME DIALOG JSON BEATS):
      {dialog_text}

      ASSET REGISTRY:
      {registry}

      SCENE CONTEXT:
      {story}

      MAPPING RULES (NON-NEGOTIABLE):
      1. EVERY [ACTION] LINE → 1 JSON beat with type="action", text=null, facial_action=null.
      2. DIALOG BEAT RULES (APPLY ONLY TO TYPE="dialog"):
        • text = ONLY the final spoken words. NEVER include "Speaker:", Mood, or [brackets].
        • facial_action = Mood + Action combined (e.g., "confident, adjusts glasses, smiles").
        • visible_chars = Speaker ONLY. "Teacher"/"Instructor"/"Wife" → [1]. "Assistant"/"Kitten"/"Husband" → [2].
        • shot_type = "closeup" (ALWAYS). NEVER use [1,2] for dialog beats.
        • motion_prompt = null, motion_type = "static" (DIALOG IS ALWAYS STATIC).
      3. ACTION CHARACTER IDENTIFICATION: Map descriptive phrases in [ACTION] lines to EXACT registry characters.
        • "teacher", "woman in blazer", "instructor", "glasses" → visible_chars: [1]
        • "assistant", "kitten dress", "red dress", "cat ears" → visible_chars: [2]
        • If both described → visible_chars: [1, 2]
      4. ACTION MOTION EXTRACTION (APPLY ONLY TO TYPE="action"):
        • Convert physical actions into EXPLICIT, TAGGED kinematic instructions.
        • Assign to motion_prompt: "Teacher: [verb], [verb] | Assistant: [verb], [verb]"
        • Set motion_type = "dynamic" (overt movement), "subtle" (micro-adjustments), or "static" (held pose only).
        • NEVER use pronouns ("she", "her", "they"). MAX 2 verbs per character. If only one moves, omit the other tag.
      5. SHOT TYPE (ACTIONS): len(visible_chars)==1 → "closeup"|"medium"|"profile_left"|"profile_right". len==2 → "two_shot"|"ots".
      6. ASSET ROUTING: base_composite MUST be an EXACT registry alias matching the visual description. Keyword-match starting_pose against registry descriptions. If no match, set null.
      7. ORDERING: Maintain exact narrative sequence. Do not skip, merge, or reorder lines.
      8. OUTPUT ONLY RAW JSON. NO MARKDOWN. NO EXPLANATIONS.

      SCHEMA:
      {SCHEMA}
      Respond ONLY with the JSON:"""
    print(prompt)

if __name__ == "__main__":
    main()