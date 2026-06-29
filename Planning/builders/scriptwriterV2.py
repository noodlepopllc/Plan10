import json, sys
sys.path.append('./lib')
from qwen_llm import llm_analyze_media
from pathlib import Path

prompt_path = './Planning/prompts'
WORLD = Path(f'{prompt_path}/scriptwriter/world.txt').read_text()
BIOGRAPHY = Path(f'{prompt_path}/scriptwriter/biography.txt').read_text()
NARRATOR = Path(f'{prompt_path}/scriptwriter/narratorV2.txt').read_text()

# Updated to include backdrop
REQUIRED_FIELDS = ['actor', 'speaker', 'action', 'dialog', 'location', 'zone', 'backdrop', 'posture', 'facial']
COLD_OPEN_MARKER = '******* COLD OPEN END ****'

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

def split_story_at_cold_open(prose):
    """
    Split story into cold open and story beats.
    Returns (cold_open_text, story_text).
    If marker not found, returns ('', prose).
    """
    if COLD_OPEN_MARKER in prose:
        parts = prose.split(COLD_OPEN_MARKER, 1)
        cold_open = parts[0].strip()
        story = parts[1].strip() if len(parts) > 1 else ''
        return cold_open, story
    else:
        print(f'Warning: Cold open marker not found, processing entire story')
        return '', prose

def split_into_paragraphs(prose):
    """Split prose into paragraphs."""
    paragraphs = prose.split('\n\n')
    if len(paragraphs) == 1:
        paragraphs = prose.split('\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

def fix_truncated_json(line):
    """Try to fix truncated JSON by closing unclosed strings and braces."""
    quote_count = line.count('"')
    
    if quote_count % 2 == 1:
        line = line.rstrip() + '"'
    
    open_braces = line.count('{')
    close_braces = line.count('}')
    
    if open_braces > close_braces:
        missing = open_braces - close_braces
        line = line.rstrip() + '}' * missing
    
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None

def validate_beat(beat, current_state):
    """Ensure beat has all required fields."""
    validated = {}
    
    for field in REQUIRED_FIELDS:
        if field in beat and beat[field]:
            validated[field] = beat[field]
        else:
            if field in current_state:
                validated[field] = current_state[field]
            elif field in ['actor', 'speaker', 'action', 'dialog']:
                validated[field] = ""
            else:
                validated[field] = "unknown"
    
    return validated

def parse_jsonl(text, current_state):
    """Parse JSONL text into list of dicts with validation."""
    beats = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        try:
            beat = json.loads(line)
            validated_beat = validate_beat(beat, current_state)
            beats.append(validated_beat)
            continue
        except json.JSONDecodeError:
            pass
        
        fixed = fix_truncated_json(line)
        if fixed:
            validated_beat = validate_beat(fixed, current_state)
            beats.append(validated_beat)
            print(f"Fixed truncated JSON line")
            continue
        
        print(f"Warning: Could not parse line: {line[:100]}...")
    
    return beats

def extract_initial_backdrop(biography_data):
    """Extract the first available backdrop from biography."""
    try:
        locations = biography_data.get('locations', [])
        if locations:
            zones = locations[0].get('zones', [])
            if zones:
                backdrops = zones[0].get('backdrops', [])
                if backdrops:
                    return backdrops[0].get('backdrop_name', 'Unknown')
    except:
        pass
    return 'Unknown'

def extract_initial_state(biography):
    """Extract initial state from biography with three-tier location/zone/backdrop."""
    try:
        bio_data = json.loads(biography)
        if 'biographies' in bio_data and len(bio_data['biographies']) > 0:
            first_char = bio_data['biographies'][0]['name']
            
            location = "Unknown"
            zone = "Unknown"
            backdrop = "Unknown"
            
            if 'locations' in bio_data and len(bio_data['locations']) > 0:
                first_loc = bio_data['locations'][0]
                location = first_loc.get('name', 'Unknown')
                
                if 'zones' in first_loc and len(first_loc['zones']) > 0:
                    first_zone = first_loc['zones'][0]
                    zone = first_zone.get('zone_name', 'Unknown')
                    
                    # Extract first backdrop from this zone
                    backdrop = extract_initial_backdrop(bio_data)
            
            return {
                "posture": "standing",
                "facial": "neutral",
                "location": location,
                "zone": zone,
                "backdrop": backdrop,
                "last_actor": first_char
            }
    except Exception as e:
        print(f"Warning: Could not extract initial state: {e}")
        pass
    
    return {
        "posture": "standing",
        "facial": "neutral",
        "location": "Unknown",
        "zone": "Unknown",
        "backdrop": "Unknown",
        "last_actor": ""
    }

def build_script(story, outpath):
    expanded = Path(f'{outpath}/story.txt').read_text()
    world = run_prompt(f'{expanded}', WORLD, f'{outpath}/world.txt')
    biography = run_prompt(world, BIOGRAPHY, f'{outpath}/registry.json')
    
    # Paragraph-level extraction
    narrative_path = f'{outpath}/narrative.json'
    if not Path(narrative_path).exists():
        # Split story at cold open marker
        cold_open, story_text = split_story_at_cold_open(expanded)
        
        if cold_open:
            print(f'Found cold open ({len(cold_open)} chars), skipping it')
        else:
            print('No cold open marker found, processing entire story')
        
        paragraphs = split_into_paragraphs(story_text)
        print(f'Split story into {len(paragraphs)} paragraphs')
        
        current_state = extract_initial_state(biography)
        all_beats = []
        
        for idx, paragraph in enumerate(paragraphs):
            print(f'Processing paragraph {idx+1}/{len(paragraphs)}')
            
            paragraph_prompt = f"""Biography: {biography}

Current State: {json.dumps(current_state)}

Paragraph: {paragraph}"""
            
            result = llm_analyze_media(
                media="",
                prompt=paragraph_prompt,
                system=NARRATOR,
                max_tokens=4096,
                temperature=0.2
            )['analysis']
            
            beats = parse_jsonl(result, current_state)
            all_beats.extend(beats)
            
            if beats:
                last_beat = beats[-1]
                # Update state with all three tiers
                current_state = {
                    "posture": last_beat.get("posture", current_state["posture"]),
                    "facial": last_beat.get("facial", current_state["facial"]),
                    "location": last_beat.get("location", current_state["location"]),
                    "zone": last_beat.get("zone", current_state["zone"]),
                    "backdrop": last_beat.get("backdrop", current_state["backdrop"]),
                    "last_actor": last_beat.get("actor", current_state["last_actor"])
                }
        
        with open(narrative_path, 'w') as out_f:
            for beat in all_beats:
                out_f.write(json.dumps(beat) + '\n')
        
        print(f'Wrote {narrative_path} with {len(all_beats)} beats')
    else:
        print(f'{narrative_path} Exists')

if __name__ == '__main__':
    user_input = Path(sys.argv[1]).read_text()
    outpath = sys.argv[2]
    build_script(user_input, outpath)