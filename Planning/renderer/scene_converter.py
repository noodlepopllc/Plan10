import sys, json
from pathlib import Path
from plan10.lib.image_analysis import AnalyzeImage
from plan10.emergent.video_runner import h3_ref

def get_visual_id(ref_path):
        prompt = """Analyze this image and extract a complete profile for character ONLY.


For EACH prominent character, provide:
1. 15-25 word description including ethnicity, exact age range, hair color and style (length, texture), skin tone, face shape, distinctive facial features, and main clothing items with specific colors

"""
        
        result = AnalyzeImage(ref_path, prompt)['analysis'].strip()
        return result

def filter_empty(entry):
    vals = set([])
    for x in ("actor", "speaker", "action", "dialog"):
        vals.add(entry[x])
    return not len(vals) <= 1

def fix_locations(base, lines, registry, context):
    locations = {}
    for location in registry['locations']:
        name = location['name']
        locations[name] = []
        for zone in location['zones']:
            locations[name].append(zone['zone_name'])
    #print(locations)

    location_info = {}
    for asset in context['assets']:
        if 'BACKGROUND' in asset:
            print(asset)
            for location in locations:
                if asset.startswith(location.replace(' ','_').upper()):
                    if location not in location_info:
                        location_info[location] = {}
                    for zone in locations[location]:
                        if zone.replace(' ','_').upper() in asset:
                            location_info[location][zone] = str((base / Path(context['assets'][asset]['path'])).resolve())
    for line in lines:
        for k, v in locations.items():
            if line['zone'] in v:
                line['location'] = k
                line['background'] = location_info[k][line['zone']]
    return lines



def get_characters(base, registry, context):
    characters = {}
    for key in context['assets']:
        if key.startswith('CHAR') and 'VOICE' not in key:
            asset_path = str((base / Path(context['assets'][key]['path'])).resolve())
            char_key = key.split('_')[1].upper()
            if not char_key in characters:
                characters[char_key] = {}
            characters[char_key]['reference_path'] = asset_path
            characters[char_key]['Visual_Id'] = get_visual_id(asset_path)
        if key.startswith('CHAR') and 'VOICE' in key:
            asset_path = str((base / Path(context['assets'][key]['path'])).resolve())
            char_key = key.split('_')[1].upper()
            if not char_key in characters:
                characters[char_key] = {}
            characters[char_key]['Voice'] = asset_path
    return characters

def to_h3_prompt(entry, characters):
    # 1. Location + zone header
    header = f"{entry['location']}, {entry['zone']}"

    # 2. Character roster (from your character dict)
    roster_parts = []
    for name, info in characters.items():
        # Visual_Id is already a clean description
        desc = info['Visual_Id']
        roster_parts.append(f"{name} ({desc})")
    roster_line = ", ".join(roster_parts)

    # 3. Action line (required by H3)
    action_line = f"Action: {entry['action']}"

    # 4. Dialog line (optional)
    dialog_line = f"Dialog: {entry['dialog']}" if entry['dialog'] else ""

    # 5. Camera default (your runner expects this)
    camera_line = "Camera: static shot, medium framing"

    # 6. Combine into final string
    return "\n\n".join(x for x in [
        header,
        roster_line,
        action_line,
        dialog_line,
        camera_line
    ] if x)

def main():
    scene_base = sys.argv[1]
    context = json.loads((Path(scene_base) / 'scene/context.json').read_text())
    base = Path(scene_base).parent
    registry = json.loads((Path(scene_base) / 'output/registry.json').read_text())
    lines = [json.loads(x) for x in (Path(scene_base) / 'output/narrative.json').read_text().split('\n') if x]
    lines = [line for line in lines if filter_empty(line)]
    characters = get_characters(base, registry, context)
    lines = fix_locations(base, lines, registry, context)
    character_refs =  [characters[x]['reference_path'] for x in characters]
    #print(character_refs)
    visual_ids = [characters[x]['Visual_Id'] for x in characters]
    #print(visual_ids)
    character_names=[x for x in characters]
    #print(character_names)
    for beat, line in enumerate(lines, start=1):
        script = h3_ref(line['background'], None, character_refs, None, to_h3_prompt(line, characters), duration=10.0, visual_ids=visual_ids, char_names=character_names)
        (Path(scene_base) / f'beat_{beat:03d}.txt').write_text(script)
        print(script)
    #for line in lines:
    #    print(json.dumps(line, indent=4))

if __name__ == '__main__':
    main()