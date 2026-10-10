# story_to_script.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

from plan10.lib.config import load_config
load_config()
from plan10.lib.qwen_llm import llm_analyze_media

import re
import json
from pathlib import Path
import sys

WORLD = '''
⭐ PASS 0A — SCHEMATIC WORLD EXTRACTOR (FINAL FIXED VERSION)
ROLE — SCHEMATIC WORLD EXTRACTOR

Your job is to read both the SEED FILE and the STORY FILE and extract a low‑resolution, story‑aware world model.
INPUTS

You will receive two inputs:

SEED FILE  
Authoritative identity + world definitions.

STORY FILE  
Narrative events generated from the seed.
MERGE PROCEDURE (MANDATORY)

If the STORY omits body‑covering details, you MUST preserve the full‑body covering defined in the SEED. STORY omissions do not imply absence.

You MUST follow this exact merge procedure:
1. CHARACTERS

For each character:

    Read the SEED FILE FIRST and extract all identity details.
    Then read the STORY FILE and add ONLY new details.
    If the STORY contradicts the SEED, SEED WINS.
    Never drop seed‑level identity details.
    Never invent new details.

2. LOCATIONS

    Extract world constraints from the SEED FILE FIRST.
    Then add environmental cues from the STORY FILE.
    If STORY contradicts SEED, SEED WINS.

3. STAGING ZONES

    Use locations from the merged SEED+STORY.
    Zones must be schematic and filmable.
    No foreground blockers.

OUTPUT FORMAT

You MUST output the following sections in this exact Markdown format. Do not skip any section.
SECTION 1 — WORLD SUMMARY  
2–3 sentences max.
MUST explicitly state: Time Period, Technology Level.
Include: Tone, atmosphere, sensory palette.
❌ FORBIDDEN: Story events, character actions, detailed visuals.
SECTION 2 — CHARACTER SUMMARIES
For each character, provide a bulleted list:

    Name, Age, Gender, Race (default "humanoid"), Ethnicity/Species.
    Face shape and distinctive facial features (e.g., "sharp jawline, prominent nose, high cheekbones, scar above left eye").
    Build and posture (e.g., "tall and lean, slightly hunched").
    External body covering (specific garments, materials, colors, condition).
    Hair (exact style, color, length, how it's worn).
    Movement style (broad strokes: e.g., "heavy steps", "quick movements").
    DISTINCTIVE VISUAL MARKERS: 2-3 highly specific, unique visual traits that make this character instantly recognizable (e.g., "tattoo of serpent on neck", "missing left ear", "always wears red scarf").

❌ FORBIDDEN: Backstory, lore, emotional analysis, story events, internal thoughts.
✅ REQUIRED: Specific visual details that distinguish this character from others.
SECTION 3 — LOCATIONS
For each distinct location in the story, provide:
[Location Name]

    2–3 sentences describing what this location is, what it's used for, and its sensory elements.
    MUST include time-period-appropriate environmental context.
    ❌ FORBIDDEN: Interior sub-zones, specific props, architectural specifics beyond broad strokes, story events.

SECTION 4 — STAGING ZONES
For each location identified in Section 3, list 2–4 schematic, filmable zones within that location.
[Location Name] Zones
[Zone Name]

    Functional Purpose: [What this zone is used for]
    Schematic Objects: [2–5 non-descriptive, functional objects. CRITICAL: Objects MUST be small, peripheral, or background items (e.g., side tables, wall sconces, small stools, chairs, shelves). DO NOT list large, central, or foreground-blocking objects (e.g., large dining tables, kitchen islands, massive counters) that would dominate the frame and block actors from standing in the center/foreground.]

GLOBAL RULES  

    Do NOT write the story.  
    Do NOT write detailed locations or biographies.  
    Do NOT invent new characters or locations.  
    Keep everything schematic, non-visual, and non-descriptive.
    NO FOREGROUND BLOCKERS: Never list large, space-dominating furniture (like large tables or islands) in the Schematic Objects. The center and foreground of every zone must remain physically clear for actors to stand and interact.
'''

BIOGRAPHY = '''
⭐ PASS 0B — FILMABLE ASSET JSON COMPILER (SYSTEM PROMPT)
ROLE — FILMABLE ASSET JSON COMPILER
Your job is to take the schematic PASS 0A output and expand it into a strict, deterministic, filmable JSON object optimized for Text-to-Video (T2V) and Image-to-Video (I2V) pipelines.
INPUT EXPECTATION
You will receive the PASS 0A Markdown output. You MUST use ONLY the information provided in that input.
CORE RULES

    NO HALLUCINATION: Do not invent new characters, objects, furniture, rooms, or technology. If it is not in PASS 0A, it does not exist.
    FILMABLE & PHYSICAL: All descriptions must be physical, deterministic, and renderable by a video model. No abstract emotions, no cinematic camera language (e.g., no "dramatic lighting", use "light originates from north window").
    SPATIAL FREEDOM: The room MUST feel open, spacious, and two-orientation-compatible unless PASS 0A explicitly forbids it.
    NO FOREGROUND BLOCKERS: Large furniture (e.g., large tables, islands, massive counters) MUST NOT dominate the foreground or block the central acting area. Actors must have clear, unobstructed floor space to stand and interact.
    NATIVE JSON ONLY: Output ONLY the raw JSON object. Do NOT wrap it in markdown code blocks (no json or ). Do NOT add any text before or after the JSON. The output must start with { and end with }.

⭐ TWO-TIER ARCHITECTURE
The world model has two hierarchical levels:

    LOCATIONS: Top-level physical spaces (e.g., "The Gilded Tankard", "Parking Lot", "Back Alley")
    ZONES: Physical sub-areas within a location (e.g., "Corner Table", "Bar Counter", "Kitchen")

RULES:

    Each location contains 2-4 zones (physical sub-areas)
    Each zone generates ONE wide background image
    The wide image is automatically cropped by the renderer for different character views (left/right)
    Character positions are determined by biography order: biographies[0] = left, biographies[1] = right
    No character names needed in zone descriptions - just describe left/right environmental elements

⭐ ZONE DEFINITION (PHYSICAL SPACE + LEFT/RIGHT ELEMENTS)
A zone is a physical sub-area within a location. It describes WHAT the environment looks like on each side.
Zone naming: Use functional area names (e.g., "Corner Table", "Bar Counter", "Kitchen Pass-Through")
Zone definition must describe:

    What part of the location this area occupies
    Fixed physical features (furniture, structures)
    Spatial relationship to other zones
    Functional purpose (eating, serving, waiting)
    What environmental elements are on the LEFT side of the zone
    What environmental elements are on the RIGHT side of the zone
    CLEAR FLOOR SPACE: Explicitly describe the open floor space in the center/foreground where actors will stand unobstructed.

The zone_definition describes the FULL environment as one continuous space. The renderer will crop this into left/right portions. Character positions are implicit: first character in biographies goes left, second character goes right.
✅ CORRECT zone_definition:
"Corner Table occupies the central seating area of the tavern. The center and foreground feature wide, unobstructed wooden floorboards for standing. A small bistro table sits against the left wall, leaving the center clear. Stone walls with mounted torches surround the space. The left side of the zone shows the fireplace and stone wall. The right side shows the bar counter and doorway. Warm torchlight illuminates the entire space."
❌ WRONG zone_definition:
"Corner Table is where Maya and Elias sit. A massive oak dining table spans the entire foreground, blocking the center. Maya stands on the left near the fireplace. The lighting is warm."
The wrong version mentions character names, and places a massive table in the foreground that blocks the actors' standing space. Just describe the environment, left/right elements, and ensure the center is clear.
⭐ ZONE ENVIRONMENT DESCRIPTION (ENVIRONMENT ONLY)
The zone_definition describes ONLY the physical environment. It MUST NOT describe:

    Camera positioning or angles
    What the camera sees
    Any subject being filmed
    Character names, appearance, clothing, or actions
    Character positions (these are implicit from biography order)

Think of it as: "What does this physical space look like, and what's on the left vs right side?"
The description should cover:

    Full environment (walls, floor, ceiling, furniture, structures)
    Left side elements (what's visible on the left portion)
    Right side elements (what's visible on the right portion)
    Lighting and atmosphere (consistent across entire zone)
    Open, clear floor space in the center/foreground for actors.

⭐ ANCHORED ELEMENTS (ZONE-LEVEL OWNERSHIP)
Each zone MUST define its own anchored_elements array. These are the fixed physical objects in that specific zone.

    Each anchored element MUST have: name, material, position (left side, right side, background, etc.), orientation
    position MUST NOT be "foreground" or "center" if it is a large object that blocks the actors' standing area. Large objects must be placed in the "background" or "far left/right".
    Do NOT duplicate objects across zones unless they physically exist in multiple zones
    The zone is the SOURCE OF TRUTH for what objects exist in that space
    There is NO global environment_objects list - each zone owns its objects

⭐ VISIBLE_BACKGROUND_ELEMENTS (MANDATORY FIELD)
Every zone MUST include a "visible_background_elements" array listing 5-8 specific environmental elements visible in the full zone.
These elements should cover both left and right sides of the zone.
Examples:

    Beach zone: ["turquoise ocean", "white sand", "clear blue sky", "distant horizon", "sandy dunes on left", "beach chair on right", "cooler in background", "driftwood on left edge"]
    Tavern zone: ["stone walls", "small bistro table on left", "mounted torches", "fireplace on left", "bar counter on right", "doorway on right", "open wooden floor in center", "ceiling beams"]

The visible_background_elements help the renderer understand what props and environmental features exist in the zone.
⭐ STAGING & FOREGROUND CLEARANCE (CRITICAL)

    NO FOREGROUND BLOCKERS: Large, wide furniture (e.g., large dining tables, kitchen islands, massive counters) MUST NOT be placed in the immediate foreground or spanning the center of the zone. These objects dominate the frame and leave no physical space for actors to stand.
    CLEAR ACTING AREA: The center and foreground of every zone MUST be open, unobstructed floor space. Actors require clear physical room to stand, face each other, and interact without being blocked by furniture.
    FURNITURE PLACEMENT: Place large objects against walls, in the deep background, or pushed to the far left/right edges. If a table is necessary, it must be small (e.g., a small side table, bistro table) or positioned so actors can stand around it, not behind a massive foreground barrier.
    FORBIDDEN ORIENTATIONS: Do not orient the zone so that a large object spans the entire width of the frame in the foreground, cutting off the actors' lower bodies or blocking them entirely.

⭐ VALIDATION CHECK
Before outputting, verify:

    Does each zone_definition mention ZERO camera references (no "camera", no "angle", no "frame")?
    Does each zone_definition mention ZERO character names (no "Maya", no "Elias", no character references)?
    Does each zone_definition describe both left and right side environmental elements?
    Does each zone have a "visible_background_elements" array with 5-8 items?
    Does each zone have an "anchored_elements" array with 2-5 items?
    Are there ZERO duplicate objects across zones (unless they physically exist in multiple zones)?
    If any zone_definition contains camera references or character names, you have failed.
    If any zone is missing visible_background_elements or anchored_elements, you have failed.
    Does any zone have a large foreground object blocking the central standing area? If yes, you have failed.
    Is the center/foreground of the zone explicitly clear for actors to stand? If no, you have failed.

⭐ GLOBAL LIGHTING STATE (CRITICAL FOR CONTINUITY)
The setting MUST establish a single, fixed time of day and lighting condition that applies to ALL zones in ALL locations.
In the setting.room_form field, you MUST specify:

    Exact time of day (e.g., "mid-morning", "late afternoon", "night")
    Sky condition (e.g., "clear blue sky", "overcast grey", "stormy dark clouds")
    Sun position (e.g., "sun high overhead", "low golden sun from west", "no sun visible")
    External light color (e.g., "warm golden daylight", "cool blue twilight", "dark night with no natural light")

ALL zones MUST reference this same lighting state. The lighting MUST be consistent across all zones.
Example:
If setting says "late afternoon with low golden sun from west", then:

    Every zone shows "low golden sunlight from west"
    Every zone shows "warm orange sky" (if windows visible)
    NO zone can show "bright midday sun" or "dark night sky"

⭐ WINDOW VIEW CONSISTENCY
For any zone that includes windows or exterior views:

    visible_background_elements MUST include sky condition matching global lighting
    zone_definition MUST describe exterior lighting consistent with time of day
    If time is "night", windows show dark sky, no sunlight
    If time is "day", windows show sky color matching global state

❌ FORBIDDEN: Inconsistent sky/lighting across zones
✅ REQUIRED: All windows show same time of day and sky condition
REQUIRED JSON SCHEMA
{
  "setting": {
    "room_form": "3-5 sentences describing overall form, major fixed structures, openings, ground material, lighting sources, architectural style, and spatial scale. MUST include exact time of day, sky condition, sun position, and external light color.",
    "time_of_day": "string (e.g., 'late afternoon', 'night', 'early morning')",
    "sky_condition": "string (e.g., 'clear blue', 'overcast grey', 'dark with stars')",
    "external_lighting": "string (e.g., 'warm golden sunlight from west', 'no natural light', 'cool blue twilight')"
  },
  "biographies": [
    {
      "name": "string",
      "age": "string",
      "gender": "string",
      "race": "string",
      "ethnicity_species": "string",
      "appearance": "Combined physical description (build, face shape, skin, facial features) - ethnicity-appropriate",
      "clothing": "Silhouette, material, and color",
      "hair": "Silhouette, color, and style",
      "distinctive_visual_markers": [
        "Unique visual trait 1",
        "Unique visual trait 2"
      ],
      "movement_style": "Broad, observable physical traits",
      "personality_traits": "1-2 filmable physical traits"
    }
  ],
  "locations": [
    {
      "name": "string",
      "architectural_shell": "3-5 sentences describing shape, fixed structures, openings, materials, lighting, and scale.",
      "zones": [
        {
          "zone_name": "string (physical area name, e.g., 'Corner Table', 'Bar Counter')",
          "zone_definition": "3-5 sentences describing physical space: what part of location it occupies, fixed features, furniture, spatial relationship to other zones, what environmental elements are on left side, what environmental elements are on right side, and clear open floor space in the center/foreground. NO camera references, NO character names, NO character appearance, NO character actions, NO large foreground-blocking objects.",
          "purpose": "1-2 sentences describing functional purpose. NO story events.",
          "anchored_elements": [
            {
              "name": "string",
              "material": "string",
              "position": "string (physical position in zone: left side, right side, background, far left, far right. MUST NOT be foreground/center if large)",
              "orientation": "string"
            }
          ],
          "visible_background_elements": [
            "string (list of 5-8 background elements visible in the full zone)"
          ]
        }
      ]
    }
  ]
}
⭐ CHARACTER DISTINCTIVENESS RULES

    Each character MUST have unique facial features (different face shape, different specific features).
    Each character MUST have at least 2 distinctive_visual_markers that are highly specific and unique.
    If two characters share the same facial features or lack distinctive markers, you have failed.
    These details are CRITICAL for T2V rendering - they prevent character blending.

⭐ JSON SYNTAX VALIDATION (CRITICAL)
Before outputting, verify:

    Every array [...] closes with ] only, NOT ]}
    Every object {...} closes with } only
    The visible_background_elements array MUST close with ] followed by either , (if more fields) or } (if last field in object)
    Do NOT add extra closing braces after arrays

❌ WRONG: "visible_background_elements": ["item1", "item2"]}
✅ CORRECT: "visible_background_elements": ["item1", "item2"]
BEGIN OUTPUT NOW
Output only the valid JSON object.
'''

# ═══════════════════════════════════════════════════════════════
# PROMPT 1: ANALYZER (Semantic Extraction)
# ═══════════════════════════════════════════════════════════════
ANALYZER_PROMPT = """Extract structured data from this story beat.

WORLD CONTEXT:
{world_text}

PREVIOUS CONTEXT (for continuity):
{history}

CURRENT BEAT (raw story text):
{beat_text}

OUTPUT FORMAT (JSON ONLY):
{{
  "zone": "Exact zone name from WORLD CONTEXT",
  "summary": "One-sentence visual description of the moment.",
  "characters": [
    {{
      "name": "CHARACTER NAME",
      "physical_state": "Current physical position. MUST inherit from PREVIOUS CONTEXT unless beat explicitly describes a change.",
      "delivery": "ONE WORD describing how dialog is spoken (e.g., hoarse, tense, dangerous). If no dialog, use 'silent'.",
      "dialog": "spoken words in quotes, or null if completely silent",
      "action": "physical action description, or null"
    }}
  ]
}}

PHYSICAL STATE CONTINUITY RULES (CRITICAL):
1. Check PREVIOUS CONTEXT for each character's last known physical_state.
2. If the CURRENT BEAT explicitly describes a state change (e.g., "stands up", "kneels down", "crouches"), output the NEW state.
3. If the CURRENT BEAT does NOT describe a state change, you MUST INHERIT the previous physical_state from PREVIOUS CONTEXT.
4. NEVER output null for physical_state. If this is the character's first appearance, infer their initial state from the beat text (default to "standing" if unclear).
5. Examples:
   - Beat 1: "Sora kneels by the wall" → physical_state: "kneeling"
   - Beat 2: "Sora searches the panel" (no state change mentioned) → physical_state: "kneeling" (inherited from Beat 1)
   - Beat 3: "Sora stands up" → physical_state: "standing" (explicit change)

OTHER RULES:
1. Use PREVIOUS CONTEXT to maintain continuity across beats.
2. If the beat does not explicitly change zone, inherit the previous zone.
3. DIALOG: Extract ONLY text inside quotes. If a character makes vocal sounds but no exact quote exists, output a generic vocalization like "[mumbles]" or "[sighs]". Only use null if completely silent.
4. ACTION: Extract physical actions. If dialog is null, the action MUST NOT contain verbs of speech. Rewrite to be strictly physical (e.g., "gestures animatedly", "mouth moves silently").
5. CHARACTERS: Include ALL characters present in the beat, even if silent.
6. Output ONLY raw JSON.
"""


# ═══════════════════════════════════════════════════════════════
# PROMPT 2: FORMATTER (Strict Templating)
# ═══════════════════════════════════════════════════════════════
FORMATTER_PROMPT = """Format this beat as a script line using BEAT DATA.

BEAT DATA:
{beat_data_json}

OUTPUT FORMAT:
[ZONE: <zone>]
>> <summary>

<CHARACTER> (<delivery>) [STATE: <physical_state>]
"<dialog>"
<action>

STRICT FORMATTING RULES:
1. CHARACTER: MUST be the EXACT full name from BEAT DATA, in ALL CAPS.
2. DELIVERY: If 'delivery' exists in BEAT DATA and is not null, wrap it in parentheses: (delivery). If null or missing, omit the parentheses entirely.
3. STATE: ALWAYS include [STATE: <physical_state>] for every character. This is a continuity marker showing their current physical position.
4. DIALOG: If 'dialog' exists and is not null, wrap it in double QUOTES on its own line: "dialog text here". If null, omit this line entirely.
5. ACTION: If 'action' exists and is not null, output it on its own line. If null, omit this line entirely.
6. Output ONLY the formatted text. No markdown, no explanations.

EXAMPLE 1 (Full data with dialog):
[ZONE: Escape Pod Interior]
>> Sora pushes herself up from a slumped position.
SORA (hoarse) [STATE: kneeling]
"Comms... where are the comms?"
turns her head toward the jagged tear, shifts her weight, and reaches for the control panel

EXAMPLE 2 (No dialog, still output state):
[ZONE: Pod Exterior Debris Field]
>> Lindsy's trembling hand releases a glowing rectangular comms device.
LINDSY (stammering) [STATE: standing]
hand trembles and the metallic object slips from her fingers
"""

def build_history_context(state, max_beats=5):
    history = {
        "last_zone": state.get("zone", None),
        "last_summary": state.get("last_summary", None),
        "recent_beats": state.get("history_beats", [])[-max_beats:],
        "characters": {}
    }

    for char in state.get("active_characters", []):
        history["characters"][char] = {
            "delivery": state.get("character_delivery", {}).get(char),
            "last_dialog": state.get("character_dialog", {}).get(char),
            "last_action": state.get("character_action", {}).get(char),
            "physical_state": state.get("character_physical_state", {}).get(char)  # NEW
        }

    return history



# ═══════════════════════════════════════════════════════════════
# PYTHON STATE TRACKER (Deterministic Logic)
# ═══════════════════════════════════════════════════════════════
def update_state(state, analyzed_beat):
    new_state = state.copy()

    new_state.setdefault("active_characters", [])
    new_state.setdefault("character_delivery", {})
    new_state.setdefault("character_dialog", {})
    new_state.setdefault("character_action", {})
    new_state.setdefault("character_physical_state", {})
    new_state.setdefault("history_beats", [])

    zone = analyzed_beat.get("zone")
    if zone and zone != "Unknown":
        new_state["zone"] = zone

    summary = analyzed_beat.get("summary")
    if summary:
        new_state["last_summary"] = summary

    beat_chars = []
    for char_data in analyzed_beat.get("characters", []):
        name = char_data.get("name")
        if not name:
            continue

        if name not in new_state["active_characters"]:
            new_state["active_characters"].append(name)

        delivery = char_data.get("delivery")
        dialog = char_data.get("dialog")
        action = char_data.get("action")
        physical_state = char_data.get("physical_state")

        if delivery:
            new_state["character_delivery"][name] = delivery
        if dialog:
            new_state["character_dialog"][name] = dialog
        if action:
            new_state["character_action"][name] = action
        if physical_state:  # NEW
            new_state["character_physical_state"][name] = physical_state

        beat_chars.append({
            "name": name,
            "delivery": delivery,
            "dialog": dialog,
            "action": action,
            "physical_state": physical_state 
        })

    # append compact beat snapshot
    new_state["history_beats"].append({
        "zone": new_state.get("zone"),
        "summary": new_state.get("last_summary"),
        "characters": beat_chars
    })

    return new_state



# ═══════════════════════════════════════════════════════════════
# MAIN PROCESSING FUNCTION
# ═══════════════════════════════════════════════════════════════
def story_to_script(story_path, world_text, output_path, llm_call_func):
    from plan10.lib.qwen_llm import LLMContext
    # Load story
    story_text = Path(story_path).read_text(encoding='utf-8')

    # Split into beats
    beats = split_into_beats(story_text)

    # Prepare output file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    # Initial continuity state
    state = {
        "zone": "Unknown",
        "last_summary": None,
        "active_characters": [],
        "character_delivery": {},
        "character_dialog": {},
        "character_action": {},
        "history_beats": []  # NEW
    }

    with LLMContext() as (p_ctx, m_ctx):
        # Process each beat
        for i, beat in enumerate(beats):

            # Build history context
            history_context = build_history_context(state)

            # 1. Analyze beat with continuity
            analyzer_prompt = ANALYZER_PROMPT.format(
                world_text=world_text,
                beat_text=beat,
                history=json.dumps(history_context, indent=2)
            )

            analyzed_text = llm_call_func(analyzer_prompt, temperature=0.1, processor=p_ctx, model=m_ctx)
            analyzed_beat = safe_json_load(analyzed_text)

            if not analyzed_beat:
                print(f"WARNING: Beat {i+1} failed analysis, skipping.")
                continue

            # 2. Update continuity state
            state = update_state(state, analyzed_beat)

            # 3. Format beat
            formatter_prompt = FORMATTER_PROMPT.format(
                beat_data_json=json.dumps(analyzed_beat, indent=2)
            )

            script_line = llm_call_func(formatter_prompt, temperature=0.1, processor=p_ctx, model=m_ctx)

            # 4. Write to script file
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(script_line.strip() + '\n\n')

            print(f"Processed beat {i+1}/{len(beats)} | Zone: {state.get('zone', 'Unknown')}")


def safe_json_load(text):
    """Safely extract JSON from LLM output."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def split_into_beats(story_text):
    """Split story into beats, handling both paragraph and line-by-line formats."""
    
    # Remove everything before and including COLD OPEN END
    if 'COLD OPEN END' in story_text:
        story_text = story_text.split('COLD OPEN END')[-1]

        # Remove continuity log and everything after it
    if 'CONTINUITY LOG FOR NEXT SCENE:' in story_text:
        story_text = story_text.split('CONTINUITY LOG FOR NEXT SCENE:')[0]
    
    # Remove star markers
    story_text = re.sub(r'\*+', '', story_text)
    
    # Try paragraph breaks
    paragraphs = [b.strip() for b in re.split(r'\n\s*\n', story_text) if b.strip()]
    
    # Count total non-empty lines
    total_lines = len([l for l in story_text.split('\n') if l.strip()])
    
    # Check if any single paragraph contains many lines (line-by-line block)
    has_line_block = any(len(p.split('\n')) > 5 for p in paragraphs)
    
    if len(paragraphs) <= 5 and total_lines > 10 and has_line_block:
        # Fall back to single-line beats
        beats = [l.strip() for l in story_text.split('\n') if l.strip()]
        # Strip leading numbers like "1. " or "1) "
        beats = [re.sub(r'^\d+[\.\)]\s*', '', b) for b in beats]
        return beats
    
    return paragraphs

def run_prompt(prompt, system, pth):
    if not Path(pth).exists():
      result = llm_analyze_media(
          media="", 
          prompt=prompt,
          system=system,
          max_tokens=8192,
          temperature=0.2)['analysis']
      with open(pth, 'w', encoding='utf-8') as out_f:
        out_f.write(result)
      print(f'Wrote {pth}')
      return result
    else:
      print(f'{pth} Exists')
      return Path(pth).read_text(encoding='utf-8')

def extract_zone_differentiators(world_json):
    """Programmatically find shared vs unique elements across zones."""
    from collections import defaultdict
    
    # Collect all elements from all zones
    element_zones = defaultdict(set)  # element_name -> set of zone_names
    
    for loc in world_json.get('locations', []):
        for zone in loc.get('zones', []):
            zone_name = zone.get('zone_name', '')
            
            # From anchored_elements
            for elem in zone.get('anchored_elements', []):
                elem_name = elem.get('name', '').lower()
                if elem_name:
                    element_zones[elem_name].add(zone_name)
            
            # From visible_background_elements
            for bg_elem in zone.get('visible_background_elements', []):
                elem_name = bg_elem.lower().strip()
                if elem_name:
                    element_zones[elem_name].add(zone_name)
    
    # Classify: shared (appears in 2+ zones) vs unique (appears in 1 zone)
    shared_elements = []
    zone_unique = defaultdict(list)
    
    for elem, zones in element_zones.items():
        if len(zones) >= 2:
            shared_elements.append(elem)
        else:
            # This element is unique to the single zone it appears in
            zone_name = list(zones)[0]
            zone_unique[zone_name].append(elem)
    
    return shared_elements, zone_unique

def format_compact_world(world_json):
    """Extract names and programmatically-computed zone differentiators."""
    chars = [c['name'] for c in world_json.get('biographies', [])]
    
    # Compute shared vs unique programmatically
    shared_elements, zone_unique = extract_zone_differentiators(world_json)
    
    # Build zone descriptions with unique elements only
    zones = []
    for loc in world_json.get('locations', []):
        for zone in loc.get('zones', []):
            zone_name = zone.get('zone_name', '')
            purpose = zone.get('purpose', 'N/A')
            
            # Get ONLY the unique elements for this zone
            unique_props = zone_unique.get(zone_name, [])
            
            if zone_name:
                zone_info = f"- {zone_name}: {purpose}"
                if unique_props:
                    zone_info += f" | UNIQUE: {', '.join(unique_props[:5])}"  # Limit to top 5
                zones.append(zone_info)
    
    # Build final output
    lines = [
        "VALID CHARACTERS: " + ", ".join(chars),
        "",
        "SHARED ACROSS ALL ZONES (ignore these for zone selection):",
        f"- {', '.join(shared_elements[:10])}",  # Limit to top 10
        "",
        "VALID ZONES (choose based on UNIQUE elements only):",
        "\n".join(zones)
    ]
    
    return "\n".join(lines)

def main():
    import json
    from plan10.lib.qwen_llm import llm_analyze_media
    
    def my_llm_call(prompt, temperature=0.1, processor=None, model=None):
        result = llm_analyze_media('', prompt=prompt, system=None, max_tokens=8192, temperature=temperature, processor=processor, model=model)
        return result['analysis'] 

    import argparse

    # ═══════════════════════════════════════════════════════════════
    # ARGUMENT PARSING
    # ═══════════════════════════════════════════════════════════════
    parser = argparse.ArgumentParser(description='Convert story prose to script format')
    parser.add_argument('paths', nargs='+', help='[seed_path] dir_path')
    parser.add_argument('--story', default=None, help='Path to story file (default: <dir_path>/story.txt)')
    args = parser.parse_args()

    # Parse positional arguments (backward compatible)
    if len(args.paths) == 1:
        dir_path = args.paths[0]
        seed = None
    elif len(args.paths) == 2:
        seed_path = args.paths[0]
        dir_path = args.paths[1]
        seed = Path(seed_path).read_text(encoding='utf-8')
    else:
        parser.error("Expected 1 or 2 positional arguments: [seed_path] dir_path")

    # Determine story input
    if args.story:
        story_input = Path(args.story).read_text(encoding='utf-8')
    else:
        story_input = Path(f'{dir_path}/story.txt').read_text(encoding='utf-8')
    
    PLANNING_DIR = Path(__file__).resolve().parent.parent
    prompt_path = str(PLANNING_DIR / "prompts")
    world = run_prompt(f'SEED FILE: \n{seed}\n STORY FILE: \n{story_input}', WORLD, f'{dir_path}/world.txt')
    biography_text = run_prompt(world, BIOGRAPHY, f'{dir_path}/registry.json')
    world_text = format_compact_world(json.loads(biography_text))
    
    story_to_script(
        story_path=args.story if args.story else f'{dir_path}/story.txt',
        world_text=world_text,
        output_path=f'{dir_path}/script.txt',
        llm_call_func=my_llm_call
    )

# ═══════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    main()
