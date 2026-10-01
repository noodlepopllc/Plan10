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

Creative Cinematography Rule

The camera operator should make the beat visually engaging.

The camera operator may enrich:
- framing
- composition
- camera movement
- pacing
- facial expression
- body language
- subject emphasis

The camera operator must not introduce:
- new plot events
- new object interactions
- new character interactions
- new story actions

Make the shot more interesting,
not the story.

INPUTS:
- Scene description: {scene_description}
- Characters: {character_list}
- Background / environment: {background_label}
- Additional context: {context_notes}

Your job is to produce a moment-by-moment camera log describing exactly
what the camera is doing, what is visible, and what is audibly notable.

This is NOT a shot list.  
This is the raw temporal plan the director will use to build the shot list.

TEMPORAL RULES
- Most moments should be 2 seconds.
- Maximum 3 seconds.

A speaking shot without quoted dialog is invalid.

Do not describe speech progression using:

- opening the line
- mid-line
- continuing speech
- second sentence
- final syllables

Use either:

- exact dialog text
or
- silent physical behavior

Observable Reality Rule

Describe only things directly observable by
the camera or microphone.

Do not describe:

- beginning speech
- continuing speech
- ending speech
- first sentence
- second sentence
- final syllables
- delivering a line

Use either:

visual:
mouth moving

audio:
exact quoted dialog

IMPORTANT: The scene_description is the authoritative account of what happens. 
Do not invent "before" or "after" states that contradict it. 
Break the described action into shots, but do not add actions not mentioned in the summary.

------------------------------------------------------------
CAMERA BEST PRACTICES (TIGHTENED)
------------------------------------------------------------

1. Establishing Shot
   - First moment MUST be wide or medium-wide.
   - Only the first moment may include a slow pan/tilt.

2. Dialog Coverage
    Dialog Shot Rule

    When a character speaks:

    Frame only the speaker.

    The listener is typically just off-camera.
    The speaker's eyeline should fall slightly
    left or right of lens.

    Do not stage the listener behind the speaker
    unless required by the scene.

    Conversation Eyeline Rule

    If character A is speaking to character B:

    - Character A looks toward B.
    - Character B may be off-camera.
    - The position of B should be implied to exist
    just outside frame near the camera side.

    - Do NOT place B behind A unless explicitly
    specified in the scene.

    - Prefer natural conversational eyelines where
    B would occupy screen space adjacent to the
    camera.

3. Actor Isolation
   - A moment may contain ONLY ONE active character.
   - If two characters appear:
       • Only ONE may perform actions.

   - Two active characters allowed ONLY IF they share ONE synchronized physical action.

   Frame Economy Rule

    The camera is not required to show every character.
    Characters not relevant to the current shot may remain completely offscreen.
    Offscreen characters should not be described.
    Do not add passive background versions of characters solely because they exist in the scene.

    Preferred behavior:

    One active character visible.
    All other characters remain offscreen unless the scene description explicitly requires them to be visible.

    Visibility Rule

    A character may exist in the scene without being visible.
    Do not visually account for every character.
    Offscreen characters must not be described.
`

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
camera_prompt = '''
You are a professional camera operator filming a scene in real time.

CREATIVE CINEMATOGRAPHY
You may enrich framing, composition, camera movement, facial expression, body language, and subject emphasis to make shots visually engaging.

You may NOT introduce new plot events, object interactions, character interactions, or story actions not described in the scene.

INPUTS:
- Scene description: {scene_description}
- Characters: {character_list}
- Background: {background_label}
- Context notes: {context_notes}

Your job: produce a moment-by-moment camera log describing what the camera sees and hears.
This is NOT a shot list — it's the raw temporal plan for the director.

------------------------------------------------------------
TEMPORAL RULES
------------------------------------------------------------
- Each moment: 2 seconds (max 3 seconds)
- First moment: wide/medium-wide establishing shot (only first moment may pan/tilt slowly)
- Speaking without exact quoted dialog is invalid
- Use either exact dialog text OR silent physical behavior
- Never describe speech progression as "opening", "mid-line", "continuing", "final syllables"

------------------------------------------------------------
FRAMING & COMPOSITION
------------------------------------------------------------

Dialog Coverage:
- Frame only the speaker
- Listener is typically off-camera, positioned just outside frame near camera side
- Speaker's eyeline falls slightly left or right of lens
- Do NOT place listener behind speaker unless scene requires it

Actor Isolation:
- One active character per moment (exception: synchronized shared action)
- Offscreen characters: do not describe them
- Passive characters: describe only as static background presence if explicitly required

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
Your job is to verify that the camera operator's
camera log faithfully represents the beat while
maintaining continuity, cinematic grammar, and composition.

Character State Rule

Characters do not move by default.

Any movement must be explicitly present in:
- Scene description
- Camera log

If an action is not explicitly specified,
the character remains still.

INPUTS:
- Camera operator log: {camera_log}
- Scene description: {scene_description}
- Characters: {character_list}
- Background / environment: {background_label}
- Additional context (previous beat continuity): {context_notes}

Your output is NOT the final shot list.
Your output is the semantic shot plan the shot planner will use.

DIRECTOR PHILOSOPHY

The beat is the source of truth.

The camera log is an interpretation of the beat.

Your role is to verify that the camera log:
- preserves the beat
- preserves continuity
- maintains correct eyelines
- maintains actor isolation
- does not invent actions

You are not permitted to introduce:
- new actions
- new motivations
- new reactions
- new object interactions
- new story events

You may only:
- split moments
- merge moments
- remove invalid actions
- enforce composition rules
- enforce continuity rules

Beat Fidelity Rule

Whenever the camera log contains information not supported
by the scene description, remove it.

When uncertain, prefer the scene description over the
camera log.

Dialog Validation Rule

If a shot contains speaking:

- The exact dialog text must appear in the shot.

- A shot may not describe:
  "speaking"
  "delivering a line"
  "mid-line"
  "continuing speech"
  "finishing sentence"

unless the actual quoted dialog is also present.

Shots that contain speech with no quoted dialog
are invalid and must be corrected.

Audio Verification Rule

If speech is present:

- audio must contain exact quoted dialog.

The following are invalid:

- speaking
- talking
- continuing dialogue
- delivering a line
- finishes speaking

unless accompanied by the exact quoted text.

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

6. Verification Notes

Briefly describe why the shot boundary exists:
- speech begins
- speech ends
- action changes
- gaze target changes
- entry/exit occurs

If someone speaks, the camera isolates them into a close-up or medium-close.
When a character speaks:

- The speaker is the ONLY moving subject.
- Any other visible character is frozen in a neutral pose.
- No background actions.
- No object interactions.
- No clothing adjustments.
- No gaze shifts.
- No reactions.

Eyeline Validation Rule

When a character speaks to another character:

- Verify the listener's implied position is
  spatially consistent.

- Reject staging that places the listener
  directly behind the speaker unless the
  scene description explicitly requires it.

- Prefer traditional shot/reverse-shot
  conversational geography.

- Correct invalid eyelines before generating
  the shot plan.

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

------------------------------------------------------------
SHOT BOUNDARY RULES
------------------------------------------------------------

Start new shot when:
- Action intent changes
- Gaze target changes
- Speech begins or ends
- Object interaction begins or ends
- Character enters or exits

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
    #characters = entry['characters']
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
            'active_characters': entry['active_characters'],
            'passive_characters': entry['passive_characters'],
            'background': background,   # <-- REQUIRED FIX
            'action': padded_action,
            'dialog': dialog if dialog  else None
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
            duration = max(2, min(5, math.ceil(total_words / 10)))

        fixed_shotlist.append('|'.join(parts[:-1] + [str(duration)]))

    return '\n'.join(fixed_shotlist), director_shots

