# story_to_script.py
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

BEAT:
{beat_text}

OUTPUT FORMAT (JSON ONLY):
{{
  "zone": "Exact zone name from WORLD CONTEXT",
  "summary": "One-sentence visual description of the moment (from the beat).",
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
1. ZONE SELECTION: Choose the zone that best matches environmental cues, props, terrain, or background elements mentioned in the beat.
2. SUMMARY: Extract the beat’s visual moment description. If unclear, infer from actions and environmental cues.
3. CHARACTERS: Include ALL characters present in the beat, even if they have no dialog or action.
4. DELIVERY: ONE WORD describing how the dialog is spoken. If no dialog, infer tone from context or set to "neutral".
5. DIALOG: Extract ONLY text inside quotes. Strip quotes. If none, set to null.
6. ACTION: Extract physical actions performed by the character. If none, set to null.
7. Output ONLY raw JSON."""


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


# ═══════════════════════════════════════════════════════════════
# PYTHON STATE TRACKER (Deterministic Logic)
# ═══════════════════════════════════════════════════════════════
def update_state(state, analyzed_beat):
    """Track only zone, active characters, last speaker, last actor."""
    new_state = state.copy()

    # Track active characters
    if 'active_characters' not in new_state:
        new_state['active_characters'] = []

    for char_data in analyzed_beat.get('characters', []):
        char = char_data.get('name')
        if not char:
            continue

        # Add to active characters
        if char not in new_state['active_characters']:
            new_state['active_characters'].append(char)

        # Track last speaker
        if char_data.get('dialog'):
            new_state['last_speaker'] = char

        # Track last actor
        if char_data.get('action'):
            new_state['last_actor'] = char

    # Update zone if valid
    zone = analyzed_beat.get('zone')
    if zone and zone != "Unknown":
        new_state['zone'] = zone

    return new_state

# ═══════════════════════════════════════════════════════════════
# MAIN PROCESSING FUNCTION
# ═══════════════════════════════════════════════════════════════
def story_to_script(story_path, world_text, output_path, llm_call_func):
    # Load files
    story_text = Path(story_path).read_text()
    
    # Split into beats (paragraphs)
    #beats = [b.strip() for b in re.split(r'\n\s*\n', story_text) if b.strip() and 'COLD OPEN END' not in b]
    beats = split_into_beats(story_text)
    
    # Initialize output file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()
    
    # Initial state
    state = {
        'zone': 'Unknown',
        'active_characters': [],
        'last_speaker': None,
        'last_actor': None,
        'character_postures': {}
    }
    
    # Process each beat
    for i, beat in enumerate(beats):
        # 1. Analyze (LLM does semantic extraction)
        analyzer_prompt = ANALYZER_PROMPT.format(world_text=world_text, beat_text=beat)
        analyzed_text = llm_call_func(analyzer_prompt, temperature=0.1)
        analyzed_beat = safe_json_load(analyzed_text)
        
        if not analyzed_beat:
            print(f"WARNING: Beat {i+1} failed analysis, skipping.")
            continue
            
        # 2. Track State (Python does deterministic tracking)
        state = update_state(state, analyzed_beat)
        
        # 3. Format (LLM does strict templating)
        formatter_prompt = FORMATTER_PROMPT.format(
            beat_data_json=json.dumps(analyzed_beat, indent=2)
        )
        script_line = llm_call_func(formatter_prompt, temperature=0.1)
        
        # Append to file
        with open(output_file, 'a') as f:
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
      with open(pth, 'w') as out_f:
        out_f.write(result)
      print(f'Wrote {pth}')
      return result
    else:
      print(f'{pth} Exists')
      return Path(pth).read_text()

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
        seed = Path(sys.argv[1]).read_text()
        dir_path = sys.argv[2]
    else:
        dir_path = sys.argv[1]
    prompt_path = './Planning/prompts'
    WORLD = Path(f'{prompt_path}/scriptwriter/world.txt').read_text()
    BIOGRAPHY = Path(f'{prompt_path}/scriptwriter/biography.txt').read_text()
    story_input = Path(f'{dir_path}/story.txt').read_text()
    world = run_prompt(f'SEED FILE: \n{seed}\n STORY FILE: \n{story_input}', WORLD, f'{dir_path}/world.txt')
    biography_text = run_prompt(world, BIOGRAPHY, f'{dir_path}/registry.json')
    world_text = format_compact_world(json.loads(biography_text))
    
    story_to_script(
        story_path=f'{dir_path}/story.txt',
        world_text=world_text,
        output_path=f'{dir_path}/script.txt',
        llm_call_func=my_llm_call
    )