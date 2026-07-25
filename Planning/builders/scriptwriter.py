import json, sys
import re

from plan10.lib.qwen_llm import llm_analyze_media
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# 1. LOAD PROMPTS
# ═══════════════════════════════════════════════════════════════
prompt_path = './Planning/prompts'
WORLD = Path(f'{prompt_path}/scriptwriter/world.txt').read_text()
BIOGRAPHY = Path(f'{prompt_path}/scriptwriter/biography.txt').read_text()

# NEW: Load the two iterative screenplay prompts
SCREENPLAY_FIRST = Path(f'{prompt_path}/scriptwriter/screenplay_first.txt').read_text()
SCREENPLAY_CONTINUE = Path(f'{prompt_path}/scriptwriter/screenplay_continue.txt').read_text()

REQUIRED_FIELDS = ['actor', 'speaker', 'action', 'dialog', 'location', 'zone', 'posture', 'facial']

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

# ═══════════════════════════════════════════════════════════════
# 2. ITERATIVE SCREENPLAY HELPERS
# ═══════════════════════════════════════════════════════════════

def split_prose_into_paragraphs(prose):
    """Split prose into chunks by double newlines."""
    paragraphs = re.split(r'\n\s*\n', prose.strip())
    return [p.strip() for p in paragraphs if p.strip()]

def parse_iterative_output(llm_output):
    """Extract the clean screenplay text and the new state from the LLM output."""
    parts = re.split(r'<STATE_UPDATE>\s*(.*?)\s*</STATE_UPDATE>', llm_output, flags=re.DOTALL)
    
    screenplay_text = parts[0].strip()
    
    new_state = {
        "previous_zone": "Unknown",
        "active_characters": "",
        "last_known_action": ""
    }
    
    if len(parts) > 1:
        state_block = parts[1].strip()
        zone_match = re.search(r'CURRENT_ZONE:\s*(.+)', state_block)
        chars_match = re.search(r'ACTIVE_CHARACTERS:\s*(.+)', state_block)
        action_match = re.search(r'LAST_KNOWN_ACTION:\s*(.+)', state_block)
        
        if zone_match: new_state['previous_zone'] = zone_match.group(1).strip()
        if chars_match: new_state['active_characters'] = chars_match.group(1).strip()
        if action_match: new_state['last_known_action'] = action_match.group(1).strip()
        
    return screenplay_text, new_state

# ═══════════════════════════════════════════════════════════════
# 3. YOUR EXISTING PARSING LOGIC (Unchanged)
# ═══════════════════════════════════════════════════════════════

def extract_metadata_from_screenplay(screenplay_text):
    """Extract tag line, location, and character intros from screenplay header."""
    lines = screenplay_text.strip().split('\n')
    
    tag_line = ""
    location = "Unknown"
    characters = {}
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('>>'):
            tag_line = line[2:].strip()
        
        if line.startswith('INT.') or line.startswith('EXT.'):
            match = re.match(r'(INT\.|EXT\.)\s+(.+?)\s*-\s*(.+)', line)
            if match:
                location = match.group(2).strip()
        
        char_match = re.match(r'^([A-Z][A-Z\s]+)\s*\(([^)]+)\)\s*(.+)$', line)
        if char_match:
            name = char_match.group(1).strip()
            traits = char_match.group(2).strip()
            characters[name] = traits
    
    return tag_line, location, characters

def parse_screenplay_to_jsonl(screenplay_text, biography_data):
    """Parse screenplay format directly into JSONL beats."""
    # Safety tweak: strip any leftover STATE_UPDATE tags just in case
    screenplay_text = re.sub(r'<STATE_UPDATE>.*?</STATE_UPDATE>', '', screenplay_text, flags=re.DOTALL).strip()
    
    beats = []
    tag_line, location, characters = extract_metadata_from_screenplay(screenplay_text)
    
    valid_zones = set()
    if 'locations' in biography_data:
        for loc in biography_data['locations']:
            if 'zones' in loc:
                for zone in loc['zones']:
                    valid_zones.add(zone.get('zone_name', ''))
    
    valid_characters = set()
    if 'biographies' in biography_data:
        for char in biography_data['biographies']:
            valid_characters.add(char.get('name', '').upper())
    
    zone_pattern = r'\[ZONE:\s*([^\]]+)\](.*?)(?=\[ZONE:|$)'
    blocks = re.findall(zone_pattern, screenplay_text, re.DOTALL)
    
    current_state = {
        'posture': 'standing',
        'facial': 'neutral',
        'location': location,
        'zone': 'Unknown',
        'last_actor': ''
    }
    
    for zone_name, block_content in blocks:
        zone_name = zone_name.strip()
        
        if zone_name not in valid_zones:
            print(f"WARNING: Invalid zone '{zone_name}', using first valid zone")
            zone_name = list(valid_zones)[0] if valid_zones else 'Unknown'
        
        current_state['zone'] = zone_name
        
        sub_beats = [b.strip() for b in block_content.split('\n\n') if b.strip()]
        
        for sub in sub_beats:
            match_combined = re.match(r'^([A-Z][A-Z\s]+)\s*\(([^)]+)\)\s*\n(.+)$', sub, re.DOTALL)
            match_dialog = re.match(r'^([A-Z][A-Z\s]+)\s*\n(.+)$', sub, re.DOTALL)
            match_action = re.match(r'^([A-Z][A-Z\s]+)\s+(.+)$', sub, re.DOTALL)
            
            if match_combined:
                speaker = match_combined.group(1).strip()
                action = match_combined.group(2).strip()
                dialog = match_combined.group(3).strip().replace('"', '').replace('"', '').replace('"', '').strip()
                
                beat = build_beat(speaker, action, dialog, current_state, valid_characters)
                beats.append(beat)
                
                current_state['posture'] = infer_posture(action)
                current_state['facial'] = infer_facial(action)
                current_state['last_actor'] = speaker
                
            elif match_dialog:
                speaker = match_dialog.group(1).strip()
                dialog = match_dialog.group(2).strip().replace('"', '').replace('"', '').replace('"', '').strip()
                
                beat = build_beat(speaker, "", dialog, current_state, valid_characters)
                beats.append(beat)
                current_state['last_actor'] = speaker
                
            elif match_action:
                speaker = match_action.group(1).strip()
                action = match_action.group(2).strip()
                
                beat = build_beat(speaker, action, "", current_state, valid_characters)
                beats.append(beat)
                
                current_state['posture'] = infer_posture(action)
                current_state['facial'] = infer_facial(action)
                current_state['last_actor'] = speaker
    
    return beats

def build_beat(speaker, action, dialog, current_state, valid_characters):
    if speaker not in valid_characters:
        print(f"WARNING: Unknown character '{speaker}'")
        speaker = list(valid_characters)[0] if valid_characters else "UNKNOWN"
    
    actor = speaker if action else ""
    speaker_field = speaker if dialog else ""
    
    return {
        "actor": actor,
        "speaker": speaker_field,
        "action": action,
        "dialog": dialog,
        "location": current_state['location'],
        "zone": current_state['zone'],
        "posture": current_state['posture'],
        "facial": current_state['facial']
    }

def infer_posture(action_text):
    a = action_text.lower()
    if any(w in a for w in ['sits', 'sitting', 'seated', 'on bench', 'on chair']): return 'sitting'
    if any(w in a for w in ['stands', 'standing', 'rose', 'stands up']): return 'standing'
    if any(w in a for w in ['kneels', 'kneeling']): return 'kneeling'
    if any(w in a for w in ['crouches', 'crouching']): return 'crouching'
    if any(w in a for w in ['lies', 'laying', 'reclines', 'lays down']): return 'laying'
    return 'standing'

def infer_facial(action_text):
    a = action_text.lower()
    if 'eyes widen' in a or 'eyes widened' in a: return 'eyes widened'
    if 'frown' in a or 'frustrated' in a: return 'frowning'
    if 'smile' in a or 'beams' in a or 'grin' in a: return 'smiling'
    if 'rolls' in a and 'eye' in a: return 'eyes narrowed'
    if 'angry' in a or 'glare' in a: return 'angry'
    if 'laugh' in a: return 'laughing'
    if 'scowl' in a: return 'scowling'
    if 'look' in a and 'down' in a: return 'looking down'
    if 'look' in a and 'away' in a: return 'looking away'
    return 'neutral'

# ═══════════════════════════════════════════════════════════════
# 4. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def build_script(user_input_path, outpath):
    """Main pipeline: Generate world → iterative screenplay → biography → narrative JSONL."""
    
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)
    
    user_input = Path(user_input_path).read_text()
    
    # Step 1: Generate world model
    world = run_prompt(user_input, WORLD, f'{outpath}/world.txt')
    
    # Step 2: Generate screenplay PER PARAGRAPH (The Fix)
    screenplay_path = f'{outpath}/screenplay.txt'
    
    if not Path(screenplay_path).exists():
        print("Generating screenplay iteratively per paragraph...")
        paragraphs = split_prose_into_paragraphs(user_input)
        
        # Initial state for the first paragraph
        state = {
            "previous_zone": "Unknown",
            "active_characters": "",
            "last_known_action": "Scene begins."
        }
        
        full_screenplay_parts = []
        
        for i, para in enumerate(paragraphs):
            is_first = (i == 0)
            
            # Choose the correct prompt template
            template = SCREENPLAY_FIRST if is_first else SCREENPLAY_CONTINUE
            
            # CORRECTLY INJECT THE ACTUAL VARIABLES
            prompt = template.format(
                world_model=world,
                previous_zone=state['previous_zone'],
                active_characters=state['active_characters'],
                last_known_action=state['last_known_action'],
                prose_story=para
            )
            
            # Call LLM (system prompt is empty because ROLE is in the template)
            result = llm_analyze_media(
                media="", 
                prompt=prompt,
                system="", 
                max_tokens=8192,
                temperature=0.2
            )['analysis']
            
            # Parse output into clean text and new state
            screenplay_text, new_state = parse_iterative_output(result)
            full_screenplay_parts.append(screenplay_text)
            
            # Update state for the next paragraph
            state = new_state
            print(f"  -> Processed paragraph {i+1}/{len(paragraphs)} | Zone: {state['previous_zone']}")
            
        full_screenplay = "\n\n".join(full_screenplay_parts)
        with open(screenplay_path, 'w') as out_f:
            out_f.write(full_screenplay)
        print(f'Wrote {screenplay_path}')
    else:
        print(f'{screenplay_path} Exists')
    
    # Step 3: Generate biography/registry
    biography_text = run_prompt(world, BIOGRAPHY, f'{outpath}/registry.json')
    
    try:
        biography_data = json.loads(biography_text)
    except json.JSONDecodeError:
        print("ERROR: Could not parse biography JSON")
        biography_data = {}
    
    # Step 4: Parse screenplay directly to JSONL
    narrative_path = f'{outpath}/narrative.json'
    
    if not Path(narrative_path).exists():
        print("Parsing screenplay to JSONL...")
        screenplay_text = Path(screenplay_path).read_text()
        
        beats = parse_screenplay_to_jsonl(screenplay_text, biography_data)
        
        with open(narrative_path, 'w') as out_f:
            for beat in beats:
                out_f.write(json.dumps(beat) + '\n')
        
        print(f'Wrote {narrative_path} with {len(beats)} beats')
    else:
        print(f'{narrative_path} Exists')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <user_input.txt> <output_directory>")
        sys.exit(1)
    
    user_input = sys.argv[1]
    outpath = sys.argv[2]
    build_script(user_input, outpath)