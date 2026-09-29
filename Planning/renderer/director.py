import re, math
from plan10.lib.qwen_llm import llm_analyze_media

def llm(prompt):
    response = llm_analyze_media('', prompt=prompt, max_tokens=8192, temperature=0.4)['analysis']
    return response.strip()

shot_planner_prompt = '''
You are the shot planner. Your job is to convert the director’s semantic
shot plan into final Minimax-ready shot lines.

INPUT:
- Director shot plan: {director_shot_plan}

Your output IS the final shot list the renderer will use.

------------------------------------------------------------
SHOT PLANNER RULES (TIGHTENED)
------------------------------------------------------------

1. Format
   Each shot MUST be a single line:

   shot | foley. camera description. physical description. dialog (if any). | duration_seconds

   - MUST begin with literal prefix: "shot |"
   - Duration MUST be the final pipe-delimited integer (e.g., | 6)
   - Do NOT output "duration:" lines.

2. Foley
   - One short ambient cue only.
   - No invented or dramatic sounds.
   - Must match environment.

3. Camera
   - Use EXACT camera angle/movement from director.
   - No new moves.
   - No reframing not in director plan.

   Each camera moment should be as short as possible.

    Target:
    2-4 seconds

    Only exceed 4 seconds if:
    - uninterrupted speech requires it
    - complex continuous action requires it

    Maximum 6 seconds.

4. Physical Description
   - Describe ONLY the active character’s visible actions.
   - Passive characters may appear visually but MUST NOT perform actions.
   - Passive characters MUST be described with STATIC language:
       “visible at frame edge, static”
       “background presence only”
       “unmoving silhouette”
       “still, no actions”

5. Dialog
   - If a character speaks:
       • They MUST be the ONLY active character.
       • Other characters may appear visually but MUST NOT act.
       • Format:
         char speaks [English] "text"
         They close their mouth and are silent.

6. Actor Isolation (CRITICAL)
   - A shot may contain ONLY ONE active character.
   - A second character may appear ONLY IF:
       • they perform ZERO actions, and
       • they are described with STATIC language.

   - Two active characters are allowed ONLY IF they share ONE synchronized physical action
     (e.g., both running together, both lifting an object together).
   - If they are not sharing an action, isolate them into separate shots.

7. Duration
   - Use director’s duration.
   - MUST be 2–10 seconds.
   - Do NOT change duration.

8. Continuity
   - Maintain lighting, environment, character positions.
   - Maintain camera angle across merged moments.


------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------
shot | foley + description | duration_seconds

Example (FORMAT ONLY — DO NOT COPY CONTENT):
shot | soft wind. Medium shot. Alora shifts her stance, glancing toward Bartender. | 3
shot | glass clinks. Medium-close static. Bartender speaks [English] "What can I get you?" They close their mouth and are silent. | 4

------------------------------------------------------------
NOW PRODUCE THE FINAL SHOT LIST.
'''


camera_prompt = '''
You are a professional camera operator filming a scene in real time.

INPUTS:
- Scene description: {scene_description}
- Characters present: {character_list}
- Background / environment: {background_label}
- Additional context: {context_notes}

Your job is to produce a moment-by-moment camera log describing exactly
what the camera is doing, what is visible, and what is audibly notable.

This is NOT a shot list.  
This is the raw temporal plan the director will use to build the shot list.

TEMPORAL RULES
- Most moments should be 2 seconds.
- Maximum 3 seconds.

------------------------------------------------------------
CAMERA BEST PRACTICES (TIGHTENED)
------------------------------------------------------------

1. Establishing Shot
   - First moment MUST be wide or medium-wide.
   - Only the first moment may include a slow pan/tilt.

2. Dialog Coverage
    Dialog Shot Rule

    When a character speaks:

    - Frame only the speaker.
    - Do not show listeners unless explicitly required.
    - Prefer close-up or medium-close coverage.
    - Do not show listeners in the same frame.
    - Use over-the-shoulder framing only if required by the scene.

    Conversation Eyeline Rule

    If character A is speaking to character B:

    - Character A looks toward B.
    - Character A never addresses the camera.
    - Character A never looks directly into lens unless the script
    explicitly specifies breaking the fourth wall.

3. Actor Isolation
   - A moment may contain ONLY ONE active character.
   - If two characters appear:
       • Only ONE may perform actions.
       • The other MUST be static, passive, unmoving.

   - Two active characters allowed ONLY IF they share ONE synchronized physical action.

4. Camera Movement
   - Slow, intentional, motivated.
   - No sudden reframes.
   - No new moves not implied by scene.

5. Visual Description
   - Describe ONLY what the camera sees.
   - Passive characters MUST be described with STATIC language:
       “visible at frame edge, static”
       “background presence only”
       “still, no actions”

6. Audio Notes
   - Only natural diegetic sounds.
   - No invented foley.

Action Economy Rule

Characters perform actions only when
those actions advance the scene.

Do not add idle motions simply to create movement.

A stationary character is preferred over an unnecessary action.

Character Action Source Rule

Do not invent new actions.

Only use actions explicitly stated in:
- Scene description
- Context notes

If an action is not provided, the character remains still.

------------------------------------------------------------
AUDIO NOTES
------------------------------------------------------------
Include only what the camera operator would naturally hear:
- footsteps, clothing rustle, breath
- glasses, doors, chairs, objects
- environmental noise (wind, crowd, traffic)
- diegetic music (radio, jukebox)

Do NOT include:
- soundtrack
- foley design
- invented sounds
- internal monologue

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------
moment N | duration_seconds
camera: angle + movement
visual: what is visible + character actions
audio: notable sounds heard by the camera operator

------------------------------------------------------------
NOW PRODUCE THE CAMERA LOG.
'''

director_prompt = '''
You are the director. Your job is to convert the camera operator’s
moment-by-moment camera log into semantic shots.

INPUTS:
- Camera operator log: {camera_log}
- Scene description: {scene_description}
- Characters present: {character_list}
- Background / environment: {background_label}
- Additional context (previous beat continuity): {context_notes}

Your output is NOT the final shot list.
Your output is the semantic shot plan the shot planner will use.

DEFAULT STATE OF ALL CHARACTERS:

motionless
neutral posture
maintaining eyeline

until an explicit action is specified.

Action Preservation Rule

Do not create new actions.

A director shot may only contain actions
explicitly present in the camera log.

------------------------------------------------------------
DIRECTOR RULES (TIGHTENED)
------------------------------------------------------------

1. Shot Boundaries
   Prefer shorter shots.

    Start a new shot whenever:
    - action intent changes
    - gaze target changes
    - speech begins
    - speech ends
    - object interaction begins
    - object interaction ends

2. Shot Merging
   Merge ONLY IF:
   - camera angle is identical
   - motion is part of same phase
   - dialog belongs to same turn
   - no character enters/exits
   - only ONE active character is present

3. Actor Isolation (CRITICAL)
   - A shot may contain ONLY ONE active character.
   - If someone speaks:
       • They MUST be the ONLY active character.
       • Other characters may appear visually but MUST NOT act.
       • Passive characters MUST be described with STATIC language.

   - Two active characters allowed ONLY IF they share ONE synchronized physical action.

4. Shot Type
   - establishing: wide/medium-wide
   - dialog: speaker isolated
   - action: one active performer
   - reaction: one active performer

5. Duration
   - Sum of merged moments.
   - Clamp to 2–10 seconds.

6. Purpose
   - Each shot must have a clear purpose.

If someone speaks, the camera isolates them into a close-up or medium-close.
When a character speaks:

- The speaker is the ONLY moving subject.
- Any other visible character is frozen in a neutral pose.
- No background actions.
- No object interactions.
- No clothing adjustments.
- No gaze shifts.
- No reactions.

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------
shot N
type: establishing / action / dialog / reaction
moments: [list of moment numbers]
duration: estimated duration
purpose: what this shot accomplishes
camera: summary of angles/movement
visual: summary of visible elements
audio: summary of notable sounds

------------------------------------------------------------
NOW PRODUCE THE DIRECTOR SHOT PLAN.
'''

def run_camera_operator(scene_description, characters, background, context_notes):
    prompt = camera_prompt.format(
        scene_description=scene_description,
        character_list=characters,
        background_label=background,
        context_notes=context_notes
    )
    return llm(prompt)  # returns camera log text


def extract_dialog(entry):
    """
    Returns the dialog line if any character in the beat speaks.
    If multiple characters speak, return the first one.
    If none speak, return None.
    """
    for char in entry['characters']:
        if char.get('dialog'):
            return char['dialog']
    return None

def extract_action(entry):
    """
    Returns the first non-empty action from the characters list.
    If none exist, returns None.
    """
    for char in entry['characters']:
        if char.get('action'):
            return char['action']
    return None


# ------------------------------------------------------------
# 1. Split long actions into 5–15 second units
# ------------------------------------------------------------

def split_action_into_units(action: str):
    """
    Splits a long action into multiple units based on major verbs
    and conjunctions. Each unit should roughly map to 5–15 seconds
    of screen time.

    Heuristic: ~12–18 words ≈ 10–15 seconds.
    """

    if not action:
        return ""

    # Split on common sequential connectors
    chunks = re.split(r'\b(?:and|then|,)\b', action)
    units = []

    current = ""
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Add chunk to current unit
        if current:
            current += " " + chunk
        else:
            current = chunk

        # If unit is too long, finalize it
        if len(current.split()) >= 8:  # ~15 seconds
            units.append(current.strip())
            current = ""

    # Add final unit
    if current.strip():
        units.append(current.strip())

    return units


# ------------------------------------------------------------
# 2. Pad short actions (<5 seconds)
# ------------------------------------------------------------

def pad_if_too_short(action: str):
    """
    Pads an action if it is too short to fill 5 seconds.
    Heuristic: ~8 words ≈ 5 seconds.
    """

    if len(action.split()) >= 8:
        return action

    # Safe, continuity-friendly padding
    padding = ", steadying herself as the wind pushes against her"
    return action + padding


# ------------------------------------------------------------
# 3. Build director-ready entries
# ------------------------------------------------------------

def build_director_entries(entry: dict):
    location = entry['location']
    zone = entry['zone']
    characters = entry['characters']
    background = entry['background']

    dialog = extract_dialog(entry)
    action = extract_action(entry)

    action_units = split_action_into_units(action)
    director_entries = []

    for idx, unit in enumerate(action_units):
        #padded_action = pad_if_too_short(unit)
        padded_action = unit

        director_entries.append({
            'location': location,
            'zone': zone,
            'characters': characters,
            'background': background,   # <-- REQUIRED FIX
            'action': padded_action,
            'dialog': dialog if dialog and len(action_units) == 1 else None
        })

    return director_entries

def summarize_continuity_from_director_shots(director_shots_text: str) -> str:
    """
    Produces a short continuity summary from the previous beat's director output.
    This summary is passed as context_notes to the camera operator and director.
    """

    lines = director_shots_text.splitlines()
    camera_desc = []
    visual_desc = []
    audio_desc = []

    for line in lines:
        l = line.lower()

        if l.startswith("camera:"):
            camera_desc.append(line.replace("camera:", "").strip())

        elif l.startswith("visual:"):
            visual_desc.append(line.replace("visual:", "").strip())

        elif l.startswith("audio:"):
            audio_desc.append(line.replace("audio:", "").strip())

    # Build continuity summary
    summary_parts = []

    if visual_desc:
        summary_parts.append(f"Previously visible: {visual_desc[-1]}.")

    if camera_desc:
        summary_parts.append(f"Camera was positioned as: {camera_desc[-1]}.")

    if audio_desc:
        summary_parts.append(f"Ambient audio included: {audio_desc[-1]}.")

    # Final continuity paragraph
    continuity_summary = " ".join(summary_parts)

    return continuity_summary.strip()

import re

def quoted_word_count(text):
    quotes = re.findall(r'"([^"]*)"', text)
    return sum(len(q.split()) for q in quotes)

def direct(beat_entry: dict, notes=''):
    if notes: 
        notes = summarize_continuity_from_director_shots(notes)
    director_entries = build_director_entries(beat_entry)
    scene_description = " ".join([d["action"] for d in director_entries])
    camera_log = run_camera_operator(scene_description, beat_entry['characters'], beat_entry['background'], notes)
    director_shots = llm(
        director_prompt.format(
            camera_log=camera_log,
            scene_description=scene_description,
            character_list=beat_entry["characters"],
            background_label=beat_entry["background"],
            context_notes=notes
        )
    )

    final_shotlist = llm(
        shot_planner_prompt.format(
            director_shot_plan=director_shots
        )
    )
    fixed_shotlist = []
    for line in final_shotlist.split('\n'):
        parts = line.split('|')

        dialog_words = quoted_word_count(line)
        total_words = len(parts[1].split())

        if dialog_words:
            duration = max(2, min(6, math.ceil(dialog_words / 2.5)))
        else:
            duration = max(2, min(5, math.ceil(total_words / 10)))

        fixed_shotlist.append('|'.join(parts[:-1] + [str(duration)]))

    return '\n'.join(fixed_shotlist), director_shots

