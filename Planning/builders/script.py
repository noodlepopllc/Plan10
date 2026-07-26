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
  "characters": [
    {{
      "name": "CHARACTER NAME",
      "posture": "ONE WORD (sitting/standing/walking)",
      "posture_changed": true or false,
      "emotion": "ONE WORD",
      "dialog": "spoken words or null",
      "action": "action description or null"
    }}
  ],
  "zone": "Exact zone name from WORLD CONTEXT",
  "shot_setup": "Brief visual description of ALL characters and props (max 20 words)."
}}

CRITICAL RULES:
1. CHARACTER NAMES: Use EXACT full names from WORLD CONTEXT. NEVER abbreviate (e.g., use "RIVKA" not "RIV", "KAELEN" not "KAEL").
2. CHARACTERS: Extract ALL characters present in the beat, not just the speaker.
3. For each character, extract their individual posture, emotion, dialog, and action.
4. POSTURE_CHANGED: Set to true ONLY if that specific character's posture explicitly changes.
5. DIALOG: Extract ONLY words inside quotation marks. Strip quotes. Preserve punctuation.
6. SHOT_SETUP: Describe the visual setup including ALL characters present.
7. Output ONLY the raw JSON."""

# ═══════════════════════════════════════════════════════════════
# PROMPT 2: FORMATTER (Strict Templating)
# ═══════════════════════════════════════════════════════════════
FORMATTER_PROMPT = """Format this beat as a script line using data from BEAT DATA.

BEAT DATA:
{beat_data_json}

OUTPUT FORMAT:
[ZONE: <zone>]
>> <shot_setup>

<CHARACTER> (<posture>, <emotion>)
"<dialog>"
<action>

RULES:
1. If dialog exists, wrap it in QUOTES: "dialog text here"
2. If dialog is null/empty, omit the dialog line entirely.
3. Action line has NO quotes.
4. If action is null/empty, omit the action line entirely.
5. Character name MUST be the EXACT full name from BEAT DATA. NEVER abbreviate or shorten names.
6. Character name in ALL CAPS.
7. Output ONLY the formatted text."""

# ═══════════════════════════════════════════════════════════════
# PYTHON STATE TRACKER (Deterministic Logic)
# ═══════════════════════════════════════════════════════════════
def update_state(state, analyzed_beat):
    """Handle multiple characters per beat."""
    new_state = state.copy()
    
    if 'character_postures' not in new_state:
        new_state['character_postures'] = {}
    
    # Process each character in the beat
    for char_data in analyzed_beat.get('characters', []):
        char = char_data.get('name')
        if not char:
            continue
            
        # Add to active characters
        if char not in new_state['active_characters']:
            new_state['active_characters'].append(char)
        
        # Check if posture changed
        posture_changed = char_data.get('posture_changed', False)
        
        if posture_changed:
            # Analyzer detected a change, use the new posture
            new_state['character_postures'][char] = char_data['posture']
        elif char in state['character_postures']:
            # No change detected, enforce continuity with previous posture
            char_data['posture'] = state['character_postures'][char]
        else:
            # First time seeing this character, use analyzer's posture
            new_state['character_postures'][char] = char_data['posture']
        
        # Track last speaker/actor
        if char_data.get('dialog'):
            new_state['last_speaker'] = char
        if char_data.get('action'):
            new_state['last_actor'] = char
    
    # Update zone
    if analyzed_beat.get('zone') and analyzed_beat['zone'] != 'Unknown':
        new_state['zone'] = analyzed_beat['zone']
    
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

def format_compact_world(world_json):
    """Extract names and zone purposes for Analyzer."""
    chars = [c['name'] for c in world_json.get('biographies', [])]
    
    zones = []
    for loc in world_json.get('locations', []):
        for zone in loc.get('zones', []):
            zone_name = zone.get('zone_name', '')
            purpose = zone.get('purpose', 'N/A')
            if zone_name:
                zones.append(f"{zone_name}: {purpose}")
    
    return "\n".join([
        "VALID CHARACTERS: " + ", ".join(chars),
        "",
        "VALID ZONES (with purposes):",
        "\n".join(f"- {z}" for z in zones)
    ])

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
        
    dir_path = sys.argv[1]
    prompt_path = './Planning/prompts'
    WORLD = Path(f'{prompt_path}/scriptwriter/world.txt').read_text()
    BIOGRAPHY = Path(f'{prompt_path}/scriptwriter/biography.txt').read_text()
    story_input = Path(f'{dir_path}/story.txt').read_text()
    world = run_prompt(story_input, WORLD, f'{dir_path}/world.txt')
    biography_text = run_prompt(world, BIOGRAPHY, f'{dir_path}/registry.json')
    world_text = format_compact_world(json.loads(biography_text))
    
    story_to_script(
        story_path=f'{dir_path}/story.txt',
        world_path=world_text,
        output_path=f'{dir_path}/script.txt',
        llm_call_func=my_llm_call
    )