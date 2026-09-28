# story_to_script.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import json
from pathlib import Path
import sys

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
      "delivery": "ONE WORD describing how the dialog is spoken (e.g., suspicious, weary, hopeful, neutral)",
      "dialog": "spoken words or null",
      "action": "action description or null"
    }}
  ]
}}

RULES:
1. Use PREVIOUS CONTEXT to maintain continuity across beats.
2. If the beat does not explicitly change zone, inherit the previous zone.
3. If the beat does not explicitly change spatial layout, inherit the previous summary.
4. DELIVERY:
   - If dialog exists, infer delivery from tone.
   - If unclear, inherit delivery from PREVIOUS CONTEXT.
5. DIALOG:
   - Extract ONLY text inside quotes.
   - Strip quotes.
   - If none, set to null.
6. ACTION:
   - Extract physical actions performed by the character.
   - If none, set to null.
7. CHARACTERS:
   - Include ALL characters present in the beat, even if silent.
8. SUMMARY:
   - Must describe the visual moment.
   - Should be consistent with PREVIOUS CONTEXT unless the beat explicitly changes the scene.
9. Output ONLY raw JSON."""


# ═══════════════════════════════════════════════════════════════
# PROMPT 2: FORMATTER (Strict Templating)
# ═══════════════════════════════════════════════════════════════
FORMATTER_PROMPT = """Format this beat as a script line using BEAT DATA.

BEAT DATA:
{beat_data_json}

OUTPUT FORMAT:
[ZONE: <zone>]
>> <summary>

<CHARACTER> (<delivery>)
"<dialog>"
<action>

RULES:
1. If dialog exists, wrap it in QUOTES: "dialog text here".
2. If dialog is null/empty, omit the dialog line entirely.
3. If action is null/empty, omit the action line entirely.
4. Character name MUST be the EXACT full name from BEAT DATA, in ALL CAPS.
5. Output ONLY the formatted text."""

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
            "last_action": state.get("character_action", {}).get(char)
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

        if delivery:
            new_state["character_delivery"][name] = delivery
        if dialog:
            new_state["character_dialog"][name] = dialog
        if action:
            new_state["character_action"][name] = action

        beat_chars.append({
            "name": name,
            "delivery": delivery,
            "dialog": dialog,
            "action": action
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

        analyzed_text = llm_call_func(analyzer_prompt, temperature=0.1)
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
        script_line = llm_call_func(formatter_prompt, temperature=0.1)

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

# ═══════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import json
    from plan10.lib.qwen_llm import llm_analyze_media
    
    def my_llm_call(prompt, temperature=0.1):
        result = llm_analyze_media('', prompt=prompt, system=None, max_tokens=2048, temperature=temperature)
        return result['analysis'] 
    
    if len(sys.argv) < 2:
        print("Usage: python story_to_script.py <directory_path>")
        sys.exit(1)
    if len(sys.argv) == 3:
        seed = Path(sys.argv[1]).read_text(encoding='utf-8')
        dir_path = sys.argv[2]
    else:
        dir_path = sys.argv[1]
    prompt_path = './Planning/prompts'
    WORLD = Path(f'{prompt_path}/scriptwriter/world.txt').read_text(encoding='utf-8')
    BIOGRAPHY = Path(f'{prompt_path}/scriptwriter/biography.txt').read_text(encoding='utf-8')
    story_input = Path(f'{dir_path}/story.txt').read_text(encoding='utf-8')
    world = run_prompt(f'SEED FILE: \n{seed}\n STORY FILE: \n{story_input}', WORLD, f'{dir_path}/world.txt')
    biography_text = run_prompt(world, BIOGRAPHY, f'{dir_path}/registry.json')
    world_text = format_compact_world(json.loads(biography_text))
    
    story_to_script(
        story_path=f'{dir_path}/story.txt',
        world_text=world_text,
        output_path=f'{dir_path}/script.txt',
        llm_call_func=my_llm_call
    )