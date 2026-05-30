import os, sys, json
sys.path.append('./lib')
from config import load_environ
from qwen_llm import llm_analyze_media

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "123456"))

def normalize(name):
    name = name.replace(' ','_').replace('/','_')
    name = ''.join([x for x in name.upper() if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789' ])
    return name

def get_identity(registry):
    for char in assets['characters']:
        bio = char['biography']
        description = (
            bio['name'], 
            f"{bio['gender']}, Age: {bio['age']}, "
            f"{bio['race']}/{bio['ethnicity_species']}, "
            f"{bio['appearance']},{ bio['hair']}, {bio['clothing']}"
        )
        yield  (
            f'>> ALIAS: {description[0].upper()}\n' 
            f'create a character sheet of {description[1]}, Seed: {SEED}\n\n'
            f'>> ALIAS: {description[0].upper()}_VOICE\n'
            f'design a voice for {','.join(description[1].split(',')[:3])}\n'
        )

def create_zone_mapping(registry, story):
    mappings = {}
    for location in registry['locations']:
        for zone in location['zones']:
            zn = zone['zone_name']
            for beat in story:
                if beat['zone'].upper() in zn.upper():
                    mappings[zn] = beat['zone']
                    break
    return mappings

CAMERA_SYSTEM_PROMPT = '''
Use the provided zone description to generate two text-to-image prompts capturing the exact same environment from two completely different perspectives. 
For the first prompt, select one specific object as the primary focal point. For the second prompt, select a completely different object as the focal point. 
Both prompts must strictly preserve the original room layout, scale, and architectural geometry. 
Crucially, both scenes must be strictly background-only: absolutely no characters, people, or creatures should appear. 
Intentionally leave ample, unobstructed empty space in each composition to allow characters to be positioned and interact in later stages. 
Do not use the word camera anywhere in the prompts; always use view instead. Output only raw, valid JSON. 
Do not include markdown formatting, code fences, or backticks. Follow this exact structure:
{
    "cameraA": "<text to image prompt>",
    "cameraB": "<text to image prompt>"
}

'''
def compile_cinematic_zone(architecture: str, description: str, anchored_objects: str, view_angle: str = "center_wide") -> str:
    # 1. Extract core atmosphere & layout from Description
    desc_core = description.split('.')[0].strip()
    
    # 2. Parse anchored objects (first = primary, rest = secondary)
    objs = [o.strip() for o in anchored_objects.split('\n') if o.strip()]
    primary = objs[0] if objs else "central focal element"
    secondary = "; ".join(objs[1:]) if len(objs) > 1 else ""
    
    # 3. View-specific framing & spatial directives
    frame_presets = {
        "center_wide": {
            "camera": "wide straight-ahead view",
            "focus": "centered composition",
            "bleed": "left wall receding out of frame suggesting spatial continuity beyond the visible area",
            "empty": "vast empty foreground floor area completely unobstructed for character placement"
        },
        "left_diagonal": {
            "camera": "35mm view from left perimeter looking diagonally right",
            "focus": "vanishing point shifted right",
            "bleed": "right side of frame opening into implied adjacent space",
            "empty": "left midground adjacent to wall completely clear for character placement"
        },
        "right_diagonal": {
            "camera": "35mm view from right perimeter looking diagonally left",
            "focus": "vanishing point shifted left",
            "bleed": "left side of frame opening into implied adjacent space",
            "empty": "right midground adjacent to primary feature completely clear for character placement"
        }
    }
    framing = frame_presets.get(view_angle, frame_presets["center_wide"])
    
    # 4. Compose cinematic prompt
    prompt = (
        f"cinematic {framing['camera']} of {desc_core}, "
        f"anchored by {primary} positioned centrally, "
        f"{secondary} arranged to maintain direct sightlines to primary focal planes, "
        f"environmental foundation: {architecture.split('.')[0]}, "
        f"lighting: cool artificial illumination blending with shifting ambient fixtures, "
        f"composition: {framing['focus']}, {framing['bleed']}, {framing['empty']}, "
        f"smooth neutral flooring extends across open floor plan creating depth and panning potential, "
        f"no people or creatures present, sterile yet vibrant atmosphere, "
        f"cinematic open-frame composition, compositing-ready background"
    )
    return prompt

def get_backgrounds(registry, mappings, prompt_path=''):
    views = {}
    for location in registry['locations']:
        architecture = location['architectural_shell']
        for zone in location['zones']:
            if zone['zone_name'] in mappings:
                prompt = f'Architecture: {architecture}, Description: {zone["definition"]}, Anchored objects: {zone["anchored_elements"]}'
                prompt2 = compile_cinematic_zone(architecture, zone["definition"], zone["anchored_elements"])
                views[mappings[zone['zone_name']]] = {"background": prompt}
                
                if prompt_path:
                    with open(prompt_path, 'a') as pp:
                        pp.write(prompt)
                result = llm_analyze_media(
                    media="", 
                    prompt=prompt,
                    system=CAMERA_SYSTEM_PROMPT,
                    max_tokens=8192)['analysis']
                views[mappings[zone['zone_name']]].update(json.loads(result))

    for view in views:
        for camera in views[view]:
            yield f'''
>> ALIAS: {normalize(view)}_{camera.upper()}
create_background cinematic widescreen composition with generous negative space at left and right frame edges, 
primary focal objects positioned safely within center 60% of frame, smooth flooring extends toward edges to provide 
clean tracking margins for camera movement, {views[view][camera]}, Seed: {SEED}'''
            if camera == 'background':
                yield f'''
>> ALIAS: {normalize(view)}_BACKGROUND_RESIZED
edit_image {normalize(view)}_BACKGROUND asset resize image to the new resolution, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}'''         
                for position in ["front_right", "right", "back_right", "back", "back_left", "left", "front_left"]:
                    yield f'''
>> ALIAS: {normalize(view)}_{normalize(position)}
apply_gimbal_shot {normalize(view)}_BACKGROUND_RESIZED asset angle="{position}", Seed: {SEED}'''

if __name__ == '__main__':
    import sys, json
    basepath = sys.argv[1]
    with open(f'{basepath}/output/registry.json') as ass:
        assets = json.load(ass)
    with open(f'{basepath}/output/complete.json') as act:
        actions = json.load(act)

for x in get_identity(assets):
    print(x)

mappings = create_zone_mapping(assets,actions)

for x in get_backgrounds(assets, mappings, f'{basepath}/dump.txt'):
    print(x)

