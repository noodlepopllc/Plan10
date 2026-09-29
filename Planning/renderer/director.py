import re
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
SHOT PLANNER RULES
------------------------------------------------------------
1. Format
   Each shot must be written as:

   shot | foley + physical description + camera angle | duration_seconds

   Every output line MUST begin with the literal prefix: "shot |".
   Even if the director’s input already resembles a shot line.

2. Foley
   - Use a single short, natural ambient cue at the start of each shot.
   - Examples: "soft wind", "quiet footsteps", "glass clinks", "chair creaks".
   - Foley must match the environment described by the director.
   - Do NOT invent unrealistic or dramatic sounds.

3. Camera
   - Use the camera angle and movement described by the director.
   - Do NOT add new camera moves.
   - Do NOT change the shot type (medium, wide, close, etc.).

4. Physical Description
   - Describe only what the director says is visible.
   - Include small natural actions (blink, shift weight, adjust clothing).
   - Do NOT add new major actions.
   - Do NOT add new characters.

5. Dialog
   - If the director indicates dialog, format it as:
     char speaks [English] "text"
   - After speaking: "They close their mouth and are silent."

   If a character speaks, isolate them as the only active character in that shot.
Other characters may appear visually but must not perform actions.
   

6. Duration
   - Use the director’s duration.
   - Do NOT change it.
   - Do NOT exceed 10 seconds.

7. Continuity
   - Maintain consistent lighting, weather, and environment.
   - Maintain consistent character positions.
   - Maintain consistent camera angle across merged moments.

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

------------------------------------------------------------
TEMPORAL RULES
------------------------------------------------------------
- Break the scene into sequential camera moments.
- Each moment must be between 2 and 8 seconds long.
- A moment ends when:
  • the camera changes angle or position
  • a character enters or exits frame
  • a major action phase completes
  • a new conversational turn begins

------------------------------------------------------------
CAMERA BEST PRACTICES
------------------------------------------------------------
1. Establishing Shot
   - Begin with a wide or medium-wide shot showing environment + characters.
   - Slow pan or tilt allowed only in the first moment.

2. Introducing Characters
   - When a character enters frame, describe:
     • where they enter from
     • how the camera adjusts (pan, tilt, slight dolly)
     • how they settle into the composition

3. Dialog Coverage
   - Use medium or medium-close framing.
   - Ensure clear view of the speaker’s face.
   - Include the listener’s reaction (OS, two-shot, or slight angle shift).
   - Avoid rapid cutting; hold stable angles for clarity.

   If someone speaks, the camera isolates them into a close-up or medium-close.
Other characters may appear but must not perform actions.

4. Camera Movement
   - Movement must be slow, intentional, and motivated.
   - No sudden angle changes.
   - No whip-pans or fast reframes.
   - After the establishing shot, prefer static framing unless action demands movement.

------------------------------------------------------------
CHARACTER ACTIONS
------------------------------------------------------------
You may add small natural actions if they fit the environment:
- waitress wipes table
- bartender hands menu
- character adjusts clothing
- character shifts weight
- character reacts physically

These actions must be plausible and consistent with the scene.

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

------------------------------------------------------------
DIRECTOR RULES
------------------------------------------------------------
1. Shot Boundaries
   - Start a new shot when:
     • the camera angle changes
     • a character enters or exits frame
     • a new conversational turn begins
     • a major action phase begins

2. Shot Merging
   - Merge consecutive moments when:
     • camera angle is similar
     • motion is part of the same phase
     • dialog belongs to the same turn
     • no character enters/exits

3. Shot Duration
   - Duration = sum of merged moment durations
   - Clamp duration to 2–10 seconds
   - Long shots (7–10s) only allowed when ≥4 distinct phases exist

4. Shot Type
   - establishing: wide or medium-wide, introduces environment
   - dialog: someone speaks, show face + listener reaction
   - action: physical movement dominates
   - reaction: emotional or physical response dominates

5. Purpose
   - Each shot must have a clear purpose:
     • introduce environment
     • show motion
     • show dialog
     • show reaction

If someone speaks, the camera isolates them into a close-up or medium-close.
Other characters may appear but must not perform actions.

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
        if len(current.split()) >= 18:  # ~15 seconds
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
        padded_action = pad_if_too_short(unit)

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
    return final_shotlist, director_shots




