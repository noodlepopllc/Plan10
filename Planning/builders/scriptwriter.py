import json, sys
import re

from plan10.lib.qwen_llm import llm_analyze_media
from pathlib import Path

prompt_path = './Planning/prompts'
WORLD = Path(f'{prompt_path}/scriptwriter/world.txt').read_text()
BIOGRAPHY = Path(f'{prompt_path}/scriptwriter/biography.txt').read_text()
SCREENPLAY = Path(f'{prompt_path}/screenplay.txt').read_text()

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

def extract_metadata_from_screenplay(screenplay_text):
    """Extract tag line, location, and character intros from screenplay header."""
    lines = screenplay_text.strip().split('\n')
    
    tag_line = ""
    location = "Unknown"
    characters = {}
    
    for line in lines:
        line = line.strip()
        
        # Tag line
        if line.startswith('>>'):
            tag_line = line[2:].strip()
        
        # Scene heading: INT. LOCATION - TIME
        if line.startswith('INT.') or line.startswith('EXT.'):
            match = re.match(r'(INT\.|EXT\.)\s+(.+?)\s*-\s*(.+)', line)
            if match:
                location = match.group(2).strip()
        
        # Character intro: NAME (traits) action.
        char_match = re.match(r'^([A-Z][A-Z\s]+)\s*\(([^)]+)\)\s*(.+)$', line)
        if char_match:
            name = char_match.group(1).strip()
            traits = char_match.group(2).strip()
            characters[name] = traits
    
    return tag_line, location, characters

def parse_screenplay_to_jsonl(screenplay_text, biography_data):
    """Parse screenplay format directly into JSONL beats."""
    beats = []
    
    # Extract metadata
    tag_line, location, characters = extract_metadata_from_screenplay(screenplay_text)
    
    # Get valid zones from biography
    valid_zones = set()
    if 'locations' in biography_data:
        for loc in biography_data['locations']:
            if 'zones' in loc:
                for zone in loc['zones']:
                    valid_zones.add(zone.get('zone_name', ''))
    
    # Get valid character names
    valid_characters = set()
    if 'biographies' in biography_data:
        for char in biography_data['biographies']:
            valid_characters.add(char.get('name', '').upper())
    
    # Split by [ZONE: ...] markers
    zone_pattern = r'\[ZONE:\s*([^\]]+)\](.*?)(?=\[ZONE:|$)'
    blocks = re.findall(zone_pattern, screenplay_text, re.DOTALL)
    
    # Track state
    current_state = {
        'posture': 'standing',
        'facial': 'neutral',
        'location': location,
        'zone': 'Unknown',
        'last_actor': ''
    }
    
    for zone_name, block_content in blocks:
        zone_name = zone_name.strip()
        
        # Validate zone
        if zone_name not in valid_zones:
            print(f"WARNING: Invalid zone '{zone_name}', using first valid zone")
            zone_name = list(valid_zones)[0] if valid_zones else 'Unknown'
        
        current_state['zone'] = zone_name
        
        # Split block into sub-beats by double newlines
        sub_beats = [b.strip() for b in block_content.split('\n\n') if b.strip()]
        
        for sub in sub_beats:
            lines = sub.split('\n')
            
            # Pattern 1: CHARACTER (action)\nDialog → action+dialog
            match_combined = re.match(r'^([A-Z][A-Z\s]+)\s*\(([^)]+)\)\s*\n(.+)$', sub, re.DOTALL)
            
            # Pattern 2: CHARACTER\nDialog → dialog-only
            match_dialog = re.match(r'^([A-Z][A-Z\s]+)\s*\n(.+)$', sub, re.DOTALL)
            
            # Pattern 3: CHARACTER action. → action-only
            match_action = re.match(r'^([A-Z][A-Z\s]+)\s+(.+\.?)$', sub, re.DOTALL)
            
            if match_combined:
                speaker = match_combined.group(1).strip()
                action = match_combined.group(2).strip()
                dialog = match_combined.group(3).strip()
                
                # Clean dialog
                dialog = dialog.replace('"', '').replace('"', '').replace('"', '').strip()
                
                beat = build_beat(speaker, action, dialog, current_state, valid_characters)
                beats.append(beat)
                
                # Update state
                current_state['posture'] = infer_posture(action)
                current_state['facial'] = infer_facial(action)
                current_state['last_actor'] = speaker
                
            elif match_dialog:
                speaker = match_dialog.group(1).strip()
                dialog = match_dialog.group(2).strip()
                
                # Clean dialog
                dialog = dialog.replace('"', '').replace('"', '').replace('"', '').strip()
                
                beat = build_beat(speaker, "", dialog, current_state, valid_characters)
                beats.append(beat)
                
                current_state['last_actor'] = speaker
                
            elif match_action:
                speaker = match_action.group(1).strip()
                action = match_action.group(2).strip()
                
                beat = build_beat(speaker, action, "", current_state, valid_characters)
                beats.append(beat)
                
                # Update state
                current_state['posture'] = infer_posture(action)
                current_state['facial'] = infer_facial(action)
                current_state['last_actor'] = speaker
    
    return beats

def build_beat(speaker, action, dialog, current_state, valid_characters):
    """Build a single JSONL beat with proper field mapping."""
    
    # Validate speaker
    if speaker not in valid_characters:
        print(f"WARNING: Unknown character '{speaker}'")
        speaker = list(valid_characters)[0] if valid_characters else "UNKNOWN"
    
    # Determine actor/speaker fields
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
    """Infer posture from action description."""
    a = action_text.lower()
    if any(w in a for w in ['sits', 'sitting', 'seated', 'on bench', 'on chair']): 
        return 'sitting'
    if any(w in a for w in ['stands', 'standing', 'rose', 'stands up']): 
        return 'standing'
    if any(w in a for w in ['kneels', 'kneeling']): 
        return 'kneeling'
    if any(w in a for w in ['crouches', 'crouching']): 
        return 'crouching'
    if any(w in a for w in ['lies', 'laying', 'reclines', 'lays down']): 
        return 'laying'
    return 'standing'  # Default

def infer_facial(action_text):
    """Infer facial expression from action description."""
    a = action_text.lower()
    if 'eyes widen' in a or 'eyes widened' in a: 
        return 'eyes widened'
    if 'frown' in a or 'frustrated' in a: 
        return 'frowning'
    if 'smile' in a or 'beams' in a or 'grin' in a: 
        return 'smiling'
    if 'rolls' in a and 'eye' in a: 
        return 'eyes narrowed'
    if 'angry' in a or 'glare' in a: 
        return 'angry'
    if 'laugh' in a: 
        return 'laughing'
    if 'scowl' in a: 
        return 'scowling'
    if 'look' in a and 'down' in a: 
        return 'looking down'
    if 'look' in a and 'away' in a: 
        return 'looking away'
    return 'neutral'  # Default

def build_script(user_input_path, outpath):
    """Main pipeline: Generate world → screenplay → biography → narrative JSONL."""
    
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)
    
    # Read user input
    user_input = Path(user_input_path).read_text()
    
    # Step 1: Generate world model
    #world = run_prompt(user_input, WORLD, f'{outpath}/world.txt')
    world = Path(f'{outpath}/world.txt.txt').read_text()
    
    # Step 2: Generate screenplay from prose
    screenplay_prompt = f"""World Model:
{world}

Prose Story:
{user_input}"""
    
    #screenplay = run_prompt(screenplay_prompt, SCREENPLAY, f'{outpath}/screenplay.txt')
    screenplay = Path(f'{outpath}/screenplay.txt').read_text()
    
    # Step 3: Generate biography/registry
    biography_text = run_prompt(world, BIOGRAPHY, f'{outpath}/registry.json')
    
    # Parse biography JSON
    try:
        biography_data = json.loads(biography_text)
    except json.JSONDecodeError:
        print("ERROR: Could not parse biography JSON")
        biography_data = {}
    
    # Step 4: Parse screenplay directly to JSONL (NEW APPROACH)
    narrative_path = f'{outpath}/narrative.json'
    
    if not Path(narrative_path).exists():
        print("Parsing screenplay to JSONL...")
        
        beats = parse_screenplay_to_jsonl(screenplay, biography_data)
        
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