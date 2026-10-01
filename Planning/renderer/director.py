import re, math, time, os
from plan10.lib.qwen_llm import llm_analyze_media, LLMContext
from plan10.lib.config import load_config
load_config()

def llm(prompt, cooloff=10, processor=None, model=None):
    if os.environ.get("LLM_BACKEND", "transformers") == "ollama":
        print(f'Cool off period: {cooloff} seconds')
        time.sleep(cooloff)
    response = llm_analyze_media('', prompt=prompt, max_tokens=8192, temperature=0.4, processor=processor, model=model)['analysis']
    return response.strip()

shot_planner_prompt = '''
You are the shot planner.

Your job is to convert the approved director shot plan into final renderer-ready shot lines.

INPUT:
- Director shot plan: {director_shot_plan}

------------------------------------------------------------
SHOT PLANNER ROLE
------------------------------------------------------------

The director shot plan is authoritative.
Convert the approved director shot plan into renderer-ready syntax.
Preserve the director shot plan exactly as written.

------------------------------------------------------------
PRESERVATION REQUIREMENTS
------------------------------------------------------------

Keep all actions, camera descriptions, dialog, audio, and durations exactly as they appear in the director shot plan.
Maintain all camera angles, movements, shot sizes, and compositions without modification.
Use only ambient audio explicitly present in the director shot plan.
Apply the director duration exactly as specified.

------------------------------------------------------------
DIALOG FORMATTING
------------------------------------------------------------

Extract exact quoted dialog from DIALOG: "..." lines in the director shot plan.
Format all spoken dialog using this renderer syntax:

character speaks [English] "dialog text"
They close their mouth and are silent.

The [English] tag is renderer metadata required for all dialog lines.
Preserve the quoted dialog text exactly without paraphrasing or summarizing.

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

Format each shot as a single line:

shot | audio. camera. visual. dialog (if any). | duration

Requirements:
- Begin each line with "shot |"
- Place duration as the final pipe-delimited integer
- Write one shot per line
- Provide only the formatted shot lines without additional content

------------------------------------------------------------
NOW PRODUCE THE SHOT LIST.
'''

camera_prompt = '''
You are a professional camera operator filming a scene in real time.

INPUTS:
- Scene description: {scene_description}
- Characters: {character_list}
- Background: {background_label}
- Context notes: {context_notes}

Your job: produce a moment-by-moment camera log describing what the camera sees and hears.
This is the raw temporal plan for the director, not a final shot list.

------------------------------------------------------------
CREATIVE CINEMATOGRAPHY
------------------------------------------------------------

Enrich the visual presentation through framing, composition, camera movement, facial expression, body language, and subject emphasis.

Enhance observable details that are consistent with the scene:
- Facial expressions matching the character's delivery and emotion
- Body language reinforcing the described action
- Camera angles emphasizing the emotional tone

Keep all character actions and expressions strictly aligned with the scene description.
The scene_description is the authoritative source for all character behavior.

------------------------------------------------------------
CONTINUOUS CAMERA MOVEMENT
------------------------------------------------------------

The camera moves through the scene as a continuous flowing presence.

Connect every moment with smooth, motivated camera movement.
Frame transitions describe the camera moving to its new position.
Each moment ends where the next moment begins, maintaining spatial continuity.
Maintain continuous spatial tracking when a character performs a multi-part action.

Every camera movement serves a specific narrative, emotional, or spatial purpose:
- Character movement motivates the camera to follow the action
- Gaze shifts motivate the camera to pan toward the new focus
- Spatial relationships motivate the camera to reveal the environment
- Emotional beats motivate the camera to push in for intimacy or pull back for isolation

Prioritize simple, strong compositions over complex maneuvers.

------------------------------------------------------------
SPATIAL BLOCKING AND EYELINE GEOMETRY
------------------------------------------------------------

Before framing any shot, establish the spatial relationship between characters.

Identify where each character is physically located in the scene using the scene_description.
Map character positions to screen directions (screen left, screen right, center).
Establish an imaginary axis of action between interacting characters.
Keep the camera on one consistent side of this axis throughout the conversation.
Maintain consistent screen direction for each character across all moments.

Always direct the speaking character's eyeline toward the listener's established screen position.
Angle the speaker's gaze just past the lens in the listener's direction when the listener is off-screen.
Direct the speaker's gaze to match the spatial relationship described in the scene.
Keep the visual focus tightly on the active character described in the current moment.

------------------------------------------------------------
CHARACTER PRESENCE
------------------------------------------------------------

The character_list is the authoritative source for who is present in this beat.
Limit all character references to those appearing in character_list.
Treat context_notes as continuity of tone, emotion, and physical state only.

------------------------------------------------------------
ACTOR ISOLATION
------------------------------------------------------------

Feature one active character per moment.
Two characters may both be active only when performing one synchronized physical action together.
Apply the single-active-character default when the scene describes no synchronized action.

------------------------------------------------------------
TEMPORAL RULES
------------------------------------------------------------

Plan MOMENTS of 2 seconds each, with a maximum of 3 seconds.
These moments will be merged into shots by the director.
Open with a wide or medium-wide establishing shot.
Reserve slow pan or tilt for the first moment only.

------------------------------------------------------------
DIALOG
------------------------------------------------------------

Extract exact quoted dialog verbatim from DIALOG: "..." lines in scene_description.
Include exact quoted dialog in audio notes when describing speech.
Use either exact dialog text or silent physical behavior for each moment.
Speaking moments require exact quoted dialog to be valid.

------------------------------------------------------------
OBSERVABLE REALITY
------------------------------------------------------------

Describe only what the camera and microphone directly observe.

Visual: visible elements, character actions, environmental details.
Audio: natural diegetic sounds (footsteps, objects, environment) and exact quoted dialog.

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------
moment N | duration_seconds
camera: angle + movement
visual: what is visible + character actions
audio: notable sounds

------------------------------------------------------------
NOW PRODUCE THE CAMERA LOG.
'''

director_prompt = '''
Your job: verify the camera log faithfully represents the beat while maintaining continuity and cinematic grammar.

The beat is the source of truth. The camera log is an interpretation. Your role is to verify and correct.

INPUTS:
- Camera operator log: {camera_log}
- Scene description: {scene_description}
- Characters: {character_list}
- Background: {background_label}
- Context notes: {context_notes}

Your output is the semantic shot plan for the shot planner.

------------------------------------------------------------
BEAT FIDELITY
------------------------------------------------------------

Keep all actions, reactions, and object interactions strictly aligned with scene_description.
Characters remain still unless explicitly described in the scene.
When uncertain, prefer scene_description over camera_log.
Enhance the cinematic framing while preserving the source material.

------------------------------------------------------------
DIALOG EXTRACTION AND VERIFICATION
------------------------------------------------------------

Extract exact quoted dialog verbatim from DIALOG: "..." lines in scene_description.
Include exact quoted dialog in every shot describing speech.
Use exact quoted dialog text for all speaking moments.
Speaking moments require exact quoted dialog to be valid.

------------------------------------------------------------
ACTOR ISOLATION
------------------------------------------------------------

Feature one active character per shot.
Two characters may both be active only when performing one synchronized physical action together.
When a character speaks, they are the sole moving subject.
Other visible characters remain frozen and static during speech.
Passive characters appear with static language only.

------------------------------------------------------------
SPATIAL AND EYELINE VALIDATION
------------------------------------------------------------

Direct the speaker's eyeline toward the listener's established screen position.
Maintain consistent screen direction for each character across all shots.
Keep the camera on one consistent side of the axis of action.
Angle the speaker's gaze just past the lens when the listener is off-screen.
Match the speaker's gaze direction to the spatial relationship in scene_description.

------------------------------------------------------------
SHOT BOUNDARY RULES
------------------------------------------------------------

Start a new shot when:
- Action intent changes
- Gaze target changes
- Speech begins or ends
- Object interaction begins or ends
- Character enters or exits

Merge moments into shots of 2-10 seconds each.
Sum the durations of merged moments.

Merge moments only when:
- Camera angle remains identical
- Motion continues as part of the same phase
- Dialog belongs to the same turn
- No character enters or exits
- Only one active character is present

------------------------------------------------------------
SHOT TYPES
------------------------------------------------------------

- establishing: wide or medium-wide framing
- dialog: speaker isolated in frame
- action: one active performer
- reaction: one active performer

Duration: sum of merged moments, clamped to 2-10 seconds.

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------
shot N
type: establishing / action / dialog / reaction
moments: [list of moment numbers]
duration: estimated duration
purpose: what this shot accomplishes
camera: summary of angles and movement
visual: summary of visible elements
audio: summary of notable sounds
verification: why this shot boundary exists

------------------------------------------------------------
NOW PRODUCE THE DIRECTOR SHOT PLAN.
'''

def extract_dialog(entry):
    """
    Returns the dialog line if any character in the beat speaks.
    If multiple characters speak, return the first one.
    If none speak, return None.
    """
    for char in entry['active_characters']:
        if char.get('dialog'):
            return char['dialog']
    return None

def extract_action(entry):
    """
    Returns the first non-empty action from the characters list.
    If none exist, returns None.
    """
    for char in entry['active_characters']:
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
    background = entry['background']

    dialog = extract_dialog(entry)
    action = extract_action(entry)

    # If there's no action, treat it as a single empty unit so the loop still runs
    if action is None:
        action_units = [None]
    else:
        action_units = split_action_into_units(action)

    director_entries = []
    for idx, unit in enumerate(action_units):
        director_entries.append({
            'location': location,
            'zone': zone,
            'active_characters': entry['active_characters'],
            'passive_characters': entry['passive_characters'],
            'background': background,
            'action': unit,
            'dialog': dialog if idx == 0 else None  # attach dialog to first entry only
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

def build_beat_character_list(beat_entry: dict) -> list:
    """
    Extract only the characters actually present in this beat.
    Returns a list of dicts with minimal info for prompt consumption.
    """
    characters = []
    seen_names = set()
    
    # Active characters (speaking/acting)
    for char in beat_entry.get('active_characters', []):
        name = char['name']
        if name not in seen_names:
            seen_names.add(name)
            characters.append({
                'name': name,
                'role': 'active',
                'delivery': char.get('delivery'),
                'has_dialog': bool(char.get('dialog')),
                'has_action': bool(char.get('action'))
            })
    
    # Passive characters (mentioned but not active)
    for char in beat_entry.get('passive_characters', []):
        name = char['name']
        if name not in seen_names:
            seen_names.add(name)
            characters.append({
                'name': name,
                'role': 'passive',
                'source': char.get('source'),
                'mentioned_by': char.get('mentioned_by')
            })
    
    return characters

def direct(beat_entry: dict, notes=''):
    if notes: 
        notes = summarize_continuity_from_director_shots(notes)

    beat_characters = build_beat_character_list(beat_entry)
    director_entries = build_director_entries(beat_entry)
    scene_parts = []
    
    # 1. Start with summary for overall scene context
    if beat_entry.get('summary'):
        scene_parts.append(beat_entry['summary'])
    
    # 2. Add specific actions and dialog from director entries
    for d in director_entries:
        if d["action"]:
            scene_parts.append(d["action"])

        if d["dialog"]:
            scene_parts.append(
                f'DIALOG: "{d["dialog"]}"'
            )

    scene_description = " ".join(scene_parts)
    with LLMContext() as (p_ctx, m_ctx):
        camera_log = llm(camera_prompt.format(
            scene_description=scene_description,
            character_list=beat_characters,
            background_label=beat_entry['background'],
            context_notes=notes
            ), processor=p_ctx, model=m_ctx
        )  # returns camera log text
        print(f"Camera Done: \n{camera_log}")
        director_shots = llm(
            director_prompt.format(
                camera_log=camera_log,
                scene_description=scene_description,
                character_list=beat_characters,
                background_label=beat_entry["background"],
                context_notes=notes
            ), processor=p_ctx, model=m_ctx
        )
        print(f"Director Done: \n{director_shots}")

        final_shotlist = llm(
            shot_planner_prompt.format(
                director_shot_plan=director_shots
            ), processor=p_ctx, model=m_ctx
        )
        print(f"Shotlist Done: \n{final_shotlist}")
    fixed_shotlist = []
    for line in final_shotlist.split('\n'):
        parts = line.split('|')

        dialog_words = quoted_word_count(line)
        if len(parts) != 3:
            continue
        total_words = len(parts[1].split())

        if dialog_words:
            duration = max(2, min(6, math.ceil(dialog_words / 2.5)))
        else:
            # Action shots should be 2-3 seconds max
            duration = max(2, min(3, math.ceil(total_words / 20)))

        fixed_shotlist.append('|'.join(parts[:-1] + [str(duration)]))

    return '\n'.join(fixed_shotlist), director_shots

