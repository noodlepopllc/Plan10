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

Your job is to convert the approved director shot plan into
final renderer-ready shot lines.

INPUT:
- Director shot plan: {director_shot_plan}

------------------------------------------------------------
SHOT PLANNER PHILOSOPHY
------------------------------------------------------------

The director shot plan is authoritative.

You are NOT:
- a writer
- a director
- a cinematographer
- a continuity supervisor

Your only responsibility is converting the approved director
shot plan into renderer-ready syntax.

Do NOT:
- add actions
- add reactions
- add emotions
- add motivations
- add camera movements
- add camera angles
- add character behaviors
- add sounds
- add dialog
- add characters

Preserve the director shot plan exactly.

------------------------------------------------------------
PRESERVATION RULES
------------------------------------------------------------

1. Action Preservation

Every action in the output must appear in the
director shot plan.

If an action does not appear in the director shot plan,
do not create it.

2. Camera Preservation

Copy the camera description exactly.

Do not:
- reframe
- add camera movement
- change shot size
- change angle
- change composition

3. Dialog Preservation

Dialog Source Rule

If the scene description contains:

DIALOG: "..."

that exact quoted dialog is the only spoken audio.

Do not paraphrase.
Do not summarize.
Do not refer to it as:
- speaking
- mid-line
- continuing speech
- delivering a sentence

Include the exact quoted dialog.

Copy dialog exactly.

Do not rewrite dialog.

If the director shot plan indicates speech,
the final shot MUST contain the exact quoted dialog.

Dialog Formatting Rule

When converting director dialog into renderer syntax,
format all spoken dialog as:

character speaks [English] "dialog text"
They close their mouth and are silent.

[English] is renderer formatting metadata and must be added
to all dialog lines.

This formatting does not constitute modifying the dialog.
Only the quoted dialog text must be preserved exactly.

4. Audio Preservation

Use only ambient audio explicitly present in the
director shot plan.

Do not invent sounds.

5. Duration Preservation

Use the director duration exactly.

Do not modify duration.

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

Each shot MUST be one line:

shot | audio. camera. visual. dialog (if any). | duration

Rules:

- MUST begin with: "shot |"
- Duration MUST be the final pipe-delimited integer.
- One shot per line.
- No additional commentary.
- No explanations.
- No headings.
- No notes.

------------------------------------------------------------
EXAMPLE FORMAT ONLY
------------------------------------------------------------

shot | distant traffic. Medium tracking shot. Carol walks away quickly. | 2

shot | restaurant ambience. Medium-close static. Carol speaks [English] "I'm leaving." They close their mouth and are silent. | 3

-------------------------------------------------
'''

camera_prompt = '''
You are a professional camera operator filming a scene in real time.

CREATIVE CINEMATOGRAPHY
You may enrich framing, composition, camera movement, facial expression, body language, and subject emphasis to make shots visually engaging.

You may NOT introduce new plot events.

CREATIVE ENRICHMENT BOUNDARY
You may enrich observable details that are consistent with the scene:
- Facial expressions that match the character's delivery/emotion
- Body language that reinforces the described action
- Camera angles that emphasize the emotional tone

You may NOT invent:
- New physical actions not described in the scene
- Reactions not implied by the scene description
- Object interactions not mentioned

------------------------------------------------------------
CONTINUOUS MOTION (CRITICAL)
------------------------------------------------------------

The camera moves THROUGH the scene, not between static positions.

1. FLOW PRINCIPLE
   - Each moment connects to the next through motivated camera movement
   - The camera follows action, it does not jump between positions
   - Movement should be continuous: pan, tilt, dolly, track
   - Avoid abrupt cuts unless motivated by action change

2. MOTIVATED MOVEMENT
   Camera moves only when motivated by:
   - Character movement (follow the action)
   - Gaze shift (pan to where character looks)
   - Spatial relationship (reveal the environment)
   - Emotional beat (push in for intimacy, pull back for isolation)

3. TRANSITION RULES
   - Do not cut between static compositions
   - If the camera must change angle, show the movement
   - Example: instead of "medium shot → close-up", use "camera pushes in from medium to close-up"
   - Example: instead of "wide left → wide right", use "camera pans left to right following character"

4. MOMENT CONNECTION
   Each moment should end where the next moment begins:
   - If moment 1 ends with character at frame left, moment 2 should start from that position
   - If moment 1 ends with camera at eye level, moment 2 should continue from eye level
   - Avoid discontinuous jumps in framing, angle, or position

5. ACTION TRACKING
   When a character performs a multi-part action:
   - The camera follows the action continuously
   - Do not cut mid-action unless there's a clear emotional or narrative reason
   - Example: character lifts hand, reaches for object, grasps it → camera tracks the hand through the entire motion

------------------------------------------------------------
CAMERA MOVEMENT & CONTINUITY
------------------------------------------------------------

Connect every moment with smooth, motivated camera movement. 
The camera flows through the scene rather than jumping between static positions.
Frame transitions must describe the camera moving to its new position.
Maintain continuous spatial tracking when a character performs a multi-part action.

------------------------------------------------------------
MOTIVATION REQUIREMENT
------------------------------------------------------------

Every camera movement must serve a specific narrative, emotional, or spatial purpose.
Movement must follow character actions, gaze shifts, or emotional beats.
Select camera movements that enhance the established tone of the scene description.
Prioritize simple, strong compositions over complex, unmotivated maneuvers.

------------------------------------------------------------
CONVERSATIONAL EYELINE GEOMETRY
------------------------------------------------------------

Always direct the speaking character's eyeline toward the listener's established physical position in the scene.
Angle the speaker's gaze just past the lens in the listener's direction when the listener is off-screen.
Maintain consistent screen direction throughout the conversation sequence.
Keep the visual focus tightly on the active character described in the current moment.

------------------------------------------------------------
STRICT ADHERENCE TO SOURCE
------------------------------------------------------------

Limit all described character actions and expressions strictly to those explicitly provided in the scene description.
Enhance the cinematic framing of these specific actions.
Keep the character's physical behavior perfectly aligned with the source text.

INPUTS:
- Scene description: {scene_description}
- Characters: {character_list}
- Background: {background_label}
- Context notes: {context_notes}

Your job: produce a moment-by-moment camera log describing what the camera sees and hears.
This is NOT a shot list — it's the raw temporal plan for the director.

CHARACTER PRESENCE RULE
The character_list is the ONLY authoritative source for who is present in this beat.

Do NOT introduce characters mentioned in context_notes unless they also appear in character_list.
Characters not in character_list are NOT present — do not reference them by name,
do not describe their position, do not describe eyelines toward them.

context_notes provides continuity of tone, emotion, and physical state only.
It does NOT grant character presence.

------------------------------------------------------------
TEMPORAL RULES
------------------------------------------------------------
CAMERA LOG STAGE (camera_prompt):
You are planning MOMENTS (2-3 seconds each).
These moments will be merged into SHOTS by the director.

- Each moment: 2 seconds (max 3 seconds)
- Most moments should be 2 seconds. Maximum 3 seconds.
- First moment: wide/medium-wide establishing shot (only first moment may pan/tilt slowly)
- Speaking without exact quoted dialog is invalid
- Use either exact dialog text OR silent physical behavior
- Never describe speech progression as "opening", "mid-line", "continuing", "final syllables"

------------------------------------------------------------
DIALOG EXTRACTION
------------------------------------------------------------
When scene_description contains DIALOG: "..." lines:
- Extract the exact quoted text verbatim
- Include it in audio notes when describing speech
- Never use placeholders like "[DIALOG TEXT PENDING]"

------------------------------------------------------------
SPATIAL BLOCKING (CRITICAL)
------------------------------------------------------------

Before framing any shot, establish the spatial relationship between characters:

1. CHARACTER POSITIONS
   - Identify where each character is physically located in the scene
   - Use scene_description to determine: "across the table", "at the bar counter", "by the doorway"
   - Map these positions to screen directions: "screen left", "screen right", "center"

2. 180° RULE (AXIS OF ACTION)
   - Establish an imaginary line between the two interacting characters
   - The camera must stay on ONE SIDE of this line for the entire conversation
   - Never cross the line unless there's a clear camera movement that shows the transition
   - This ensures characters maintain consistent screen direction (Character A always screen left, Character B always screen right)

3. CONVERSATION GEOMETRY
   - When Character A speaks to Character B:
     • Character A's eyeline must point toward Character B's screen position
     • If Character B is screen right, Character A looks screen right
     • If Character B is screen left, Character A looks screen left
   - The speaker NEVER looks directly at the camera unless breaking the fourth wall
   - The speaker NEVER looks at someone behind them (180° violation)

4. SHOT TYPES FOR CONVERSATIONS
   - Establishing two-shot: Both characters visible, spatial relationship clear
   - Over-the-shoulder (OTS): Speaker in foreground, listener's shoulder/back in background
   - Close-up: Speaker framed alone, eyeline directed toward listener's off-screen position
   - Reverse shot: Camera flips to other side of conversation (maintains 180° rule)

------------------------------------------------------------
FRAMING & COMPOSITION
------------------------------------------------------------

Dialog Coverage:
- The speaker is the primary subject of the frame
- The speaker's eyeline MUST be directed toward the listener's established screen position
- If both characters are present in character_list:
    • Use the spatial blocking established above
    • The speaker looks TOWARD the listener, NOT at the camera
    • The listener may be at frame edge, over-the-shoulder, or just off-frame
    • The speaker's gaze angle must match the spatial relationship
    • Example: if characters sit across a table, the speaker looks across the table
- If the listener is NOT present in character_list:
    • The speaker's eyeline falls slightly left or right of lens
- Do NOT have characters speak directly into the camera
- Do NOT have characters look behind them (180° violation)

ACTOR ISOLATION
Default: One active character per moment.

Exception: Two characters may both be active ONLY IF they perform 
ONE synchronized physical action together (e.g., both lift an object, 
both turn toward the same point).

If the scene does not describe a synchronized action, apply the default.

------------------------------------------------------------
OBSERVABLE REALITY
------------------------------------------------------------
Describe only what the camera/microphone directly observes.

Visual: what is visible, character actions, environmental details
Audio: natural diegetic sounds only (footsteps, objects, environment), exact quoted dialog when spoken

Do NOT include: invented foley, soundtrack, internal monologue, or actions not in scene_description

The scene_description is authoritative. Do not invent "before" or "after" states that contradict it.

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
VERIFICATION RULES
------------------------------------------------------------

1. BEAT FIDELITY
   - Remove any action, reaction, or object interaction not in scene_description
   - Characters remain still unless explicitly described in scene
   - When uncertain, prefer scene_description over camera_log
   - Do not invent: new actions, motivations, reactions, or story events

2. DIALOG VALIDATION
   - Shots with speech MUST contain exact quoted dialog
   - Invalid without exact text: "speaking", "delivering a line", "mid-line", "continuing", "finishing"
   - Audio must contain exact quoted dialog when speech is present
   - Shots describing speech without quoted text are invalid

------------------------------------------------------------
DIALOG VERIFICATION
------------------------------------------------------------
When scene_description contains DIALOG: "..." lines:
- Extract the exact quoted text from scene_description
- Verify every dialog shot contains the exact quoted text
- Reject shots with "[DIALOG TEXT PENDING]" placeholders

3. ACTOR ISOLATION (CRITICAL)
   - One active character per shot (exception: synchronized shared action)
   - When someone speaks: they are the ONLY moving subject
   - Other visible characters: frozen, static, no reactions, no gaze shifts, no actions
   - Passive characters: described with static language only

4. EYELINE VALIDATION
   - Listener's implied position must be spatially consistent
   - Do not place listener directly behind speaker unless scene requires it
   - Prefer traditional shot/reverse-shot conversational geography
   - Correct invalid eyelines before generating shot plan

   Eyeline & Spatial Validation (CRITICAL):
    - Verify the speaker's eyeline points toward the listener's established screen position
    - Reject any shot where the speaker appears to address the camera directly
    - Reject any shot where the speaker looks behind them (180° violation)
    - If both characters are in character_list, ensure consistent screen direction across all shots
    - Correct any spatial violations before generating the shot plan

   Eyeline Correction:
    - If both speaker and listener are in character_list, verify the speaker's gaze
    is directed toward the listener's position, NOT toward the camera
    - "Looking at camera" or "directed at lens" is INVALID when the listener
    is present in the scene
    - Correct any shot where the speaker appears to address the audience
    instead of the other character

------------------------------------------------------------
SHOT BOUNDARY RULES
------------------------------------------------------------

Start new shot when:
- Action intent changes
- Gaze target changes
- Speech begins or ends
- Object interaction begins or ends
- Character enters or exits

You are merging MOMENTS into SHOTS (2-10 seconds each).
Sum the durations of merged moments.

Merge shots ONLY if:
- Camera angle identical
- Motion part of same phase
- Dialog belongs to same turn
- No character enters/exits
- Only ONE active character present

------------------------------------------------------------
SHOT TYPES
------------------------------------------------------------
- establishing: wide/medium-wide
- dialog: speaker isolated
- action: one active performer
- reaction: one active performer

Duration: sum of merged moments, clamped to 2-10 seconds

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
verification: why this shot boundary exists (speech begins/ends, action changes, gaze changes, entry/exit)

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

