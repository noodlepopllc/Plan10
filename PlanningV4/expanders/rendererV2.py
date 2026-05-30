import os, sys, json
sys.path.append('./lib')
from config import load_environ
from qwen_llm import llm_analyze_media

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "123456"))


GLOBAL_STATE = {}

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
            f'>> ALIAS: {normalize(description[0])}\n' 
            f'create a character sheet of {description[1]}, Seed: {SEED}\n\n'
            f'>> ALIAS: {normalize(description[0])}_VOICE\n'
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


def get_backgrounds(registry, mappings, prompt_path=''):
    views = {}
    for location in registry['locations']:
        architecture = location['architectural_shell']
        for zone in location['zones']:
            if zone['zone_name'] in mappings:
                prompt = f'Architecture: {architecture}, Description: {zone["definition"]}, Anchored objects: {zone["anchored_elements"]}'
                views[mappings[zone['zone_name']]] = {"background": prompt}
                

    for view in views:
        for camera in views[view]:
            yield f'''
>> ALIAS: {normalize(view)}_{camera.upper()}
create_background cinematic widescreen composition with generous negative space at left and right frame edges, 
primary focal objects positioned safely within center 60% of frame, smooth flooring extends toward edges to provide 
clean tracking margins for camera movement, {views[view][camera]}, Seed: {SEED}'''

'''
            if camera == 'background':
                yield f'''
>> ALIAS: {normalize(view)}_BACKGROUND_RESIZED
edit_image {normalize(view)}_BACKGROUND asset resize image to the new resolution, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}'''         
                for position in ["front_right", "front_left"]:
                    yield f'''
>> ALIAS: {normalize(view)}_{normalize(position)}
apply_gimbal_shot {normalize(view)}_BACKGROUND_RESIZED asset angle="{position}", Seed: {SEED}'''
'''

if __name__ == '__main__':
    import sys, json
    basepath = sys.argv[1]
    with open(f'{basepath}/output/registry.json') as ass:
        assets = json.load(ass)
    with open(f'{basepath}/output/complete.json') as act:
        actions = json.load(act)

def render_beats(assets, actions):
    names = {}
    for char in assets['characters']:
        bio = char['biography']
        names[char['name']] = f'{char['name']} ({bio['gender']}, {bio['clothing']})'
    for beat in actions:
        if beat['actions']:
            for action in beat['actions']:
                for char in names:
                    if char in action:
                        print(f'''
>> ALIAS: BEAT_{beat["beat"]}_{normalize(char)}_ACTION
composite_scene {normalize(beat['zone'])}_BACKGROUND asset and {normalize(char)} asset, {action.replace(char, names[char])}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}\n''')
                        print(f'''
>> ALIAS: BEAT_{beat["beat"]}_{normalize(char)}_ACTION_VIDEO
image_to_video BEAT_{beat["beat"]}_{normalize(char)}_ACTION asset, {action.replace(char, names[char])}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}\n''')

for x in get_identity(assets):
    print(x)

mappings = create_zone_mapping(assets,actions)

for x in get_backgrounds(assets, mappings, f'{basepath}/dump.txt'):
    print(x)

render_beats(assets, actions)

