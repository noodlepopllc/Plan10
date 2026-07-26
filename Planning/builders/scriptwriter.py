import json, sys
import re

from plan10.lib.qwen_llm import llm_analyze_media
from pathlib import Path

prompt_path = './Planning/prompts'
WORLD = Path(f'{prompt_path}/scriptwriter/world.txt').read_text()
BIOGRAPHY = Path(f'{prompt_path}/scriptwriter/biography.txt').read_text()
SCREENPLAY = Path(f'{prompt_path}/screenplay.txt').read_text()

REQUIRED_FIELDS = ['actor', 'speaker', 'action', 'dialog', 'location', 'zone', 'posture', 'facial']

def find_closest_character(name, valid_characters):
    """Find closest matching character name using multiple strategies."""
    name_upper = name.upper().strip()
    
    # Strategy 1: Exact match
    if name_upper in valid_characters:
        return name_upper
    
    # Strategy 2: Input is a prefix of a valid character (KAEL -> KAELEN, RIV -> RIVKA)
    prefix_matches = [v for v in valid_characters if v.startswith(name_upper)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    
    # Strategy 3: Valid character is a prefix of input
    prefix_matches = [v for v in valid_characters if name_upper.startswith(v)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    
    # Strategy 4: Substring containment (KAEL in KAELEN)
    contains_matches = [v for v in valid_characters if name_upper in v or v in name_upper]
    if len(contains_matches) == 1:
        return contains_matches[0]
    
    # Strategy 5: Simple edit distance (Levenshtein-like)
    best_match = None
    best_distance = float('inf')
    for v in valid_characters:
        distance = simple_edit_distance(name_upper, v)
        if distance < best_distance:
            best_distance = distance
            best_match = v
    
    # Only accept if reasonably close (within 30% of name length)
    if best_distance <= max(len(name_upper), len(best_match)) * 0.3:
        return best_match
    
    return None

def simple_edit_distance(s1, s2):
    """Simple character-by-character distance (no external deps)."""
    # Quick check: if one is prefix of other, distance is length difference
    if s1.startswith(s2) or s2.startswith(s1):
        return abs(len(s1) - len(s2))
    
    # Otherwise use basic Levenshtein
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

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

def find_closest_zone(zone_name, valid_zones):
    """Find closest matching zone, handling bracketed suffixes."""
    zone_clean = zone_name.strip()
    
    # Strip bracketed suffixes: "Neon Bazaar Stall Zones [Greasy Steel Table Zone]" -> "Greasy Steel Table Zone"
    bracket_match = re.search(r'\[([^\]]+)\]', zone_clean)
    if bracket_match:
        zone_clean = bracket_match.group(1).strip()
    
    # Exact match
    for v in valid_zones:
        if zone_clean.lower() == v.lower():
            return v
    
    # Substring containment
    for v in valid_zones:
        if v.lower() in zone_clean.lower() or zone_clean.lower() in v.lower():
            return v
    
    # Edit distance
    best_match = None
    best_distance = float('inf')
    for v in valid_zones:
        distance = simple_edit_distance(zone_clean.lower(), v.lower())
        if distance < best_distance:
            best_distance = distance
            best_match = v
    
    if best_distance <= max(len(zone_clean), len(best_match)) * 0.4:
        return best_match
    
    return None

def parse_screenplay_to_jsonl(screenplay_text, biography_data):
    """Parse screenplay format directly into JSONL beats."""
    beats = []
    
    # Extract metadata
    tag_line, location, characters = extract_metadata_from_screenplay(screenplay_text)
    
    # Get valid zones from biography (these are the clean names)
    valid_zones = set()
    if 'locations' in biography_data:
        for loc in biography_data['locations']:
            if 'zones' in loc:
                for zone in loc['zones']:
                    zone_name = zone.get('zone_name', '').strip()
                    if zone_name:
                        valid_zones.add(zone_name)
    
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
        
        # Find closest zone match
        matched_zone = find_closest_zone(zone_name, valid_zones)
        if matched_zone:
            if matched_zone != zone_name:
                print(f"INFO: Normalized zone '{zone_name}' -> '{matched_zone}'")
            zone_name = matched_zone
        else:
            print(f"WARNING: Invalid zone '{zone_name}', using first valid zone")
            zone_name = list(valid_zones)[0] if valid_zones else 'Unknown'
        
        current_state['zone'] = zone_name
        
        # Rest of parsing logic...
        sub_beats = [b.strip() for b in block_content.split('\n\n') if b.strip()]
        
        for sub in sub_beats:
            lines = [l.strip() for l in sub.split('\n') if l.strip()]
            
            if not lines:
                continue
            
            # First line should be: CHARACTER (posture, emotion)
            char_match = re.match(r'^([A-Z][A-Z\s]+)\s*\(([^)]+)\)', lines[0])
            
            if not char_match:
                continue
            
            speaker = char_match.group(1).strip()
            posture_emotion = char_match.group(2).strip()
            
            # Extract posture and emotion from parentheses
            posture, emotion = parse_posture_emotion(posture_emotion)
            
            # Update state with posture
            current_state['posture'] = posture
            current_state['facial'] = emotion
            
            # Process remaining lines
            dialog = ""
            action = ""
            
            for line in lines[1:]:
                # Check if line starts with quote (dialog)
                if line.startswith('"') or line.startswith('"'):
                    # Extract dialog, strip quotes
                    dialog = line.strip('"').strip('"').strip('"').strip('"').strip()
                else:
                    # It's an action
                    action = line
            
            # Build beat
            beat = build_beat(speaker, action, dialog, current_state, valid_characters)
            beats.append(beat)
            
            # Update state
            if action:
                current_state['posture'] = infer_posture(action)
                current_state['facial'] = infer_facial(action)
            current_state['last_actor'] = speaker
    
    return beats

def parse_posture_emotion(text):
    """Parse 'posture, emotion' from parentheses."""
    parts = [p.strip() for p in text.split(',')]
    posture = parts[0] if len(parts) > 0 else 'standing'
    emotion = parts[1] if len(parts) > 1 else 'neutral'
    return posture, emotion

def build_beat(speaker, action, dialog, current_state, valid_characters):
    """Build a single JSONL beat with proper field mapping."""
    
    # Fuzzy match speaker to valid characters
    matched_speaker = find_closest_character(speaker, valid_characters)
    
    if matched_speaker:
        if matched_speaker != speaker.upper():
            print(f"INFO: Normalized character '{speaker}' -> '{matched_speaker}'")
        speaker = matched_speaker
    else:
        print(f"WARNING: Unknown character '{speaker}', using first valid character")
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
    world = Path(f'{outpath}/world.txt').read_text()
    
    # Step 2: Generate screenplay from prose
    screenplay_prompt = f"""World Model:
{world}

Prose Story:
{user_input}"""
    
    #screenplay = run_prompt(screenplay_prompt, SCREENPLAY, f'{outpath}/screenplay.txt')
    screenplay = Path(f'{outpath}/script.txt').read_text()
    
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