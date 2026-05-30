import os, sys, json
sys.path.append('./lib')
from config import load_environ
from qwen_llm import llm_analyze_media  # kept for consistency, unused here

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "123456"))

GLOBAL_STATE = {}

def normalize(name: str) -> str:
    name = name.replace(' ', '_').replace('/', '_')
    name = ''.join([x for x in name.upper() if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'])
    return name

def get_identity(assets):
    for char in assets['characters']:
        bio = char['biography']
        description = (
            bio['name'],
            f"{bio['gender']}, Age: {bio['age']}, "
            f"{bio['race']}/{bio['ethnicity_species']}, "
            f"{bio['appearance']},{bio['hair']}, {bio['clothing']}"
        )
        name = description[0]
        yield (
            f">> ALIAS: {normalize(name)}\n"
            f"create a character sheet of {description[1]}, Seed: {SEED}\n\n"
            f">> ALIAS: {normalize(name)}_VOICE\n"
            f"design a voice for {','.join(description[1].split(',')[:3])}\n"
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

def get_backgrounds(registry, mappings):
    views = {}
    for location in registry['locations']:
        architecture = location['architectural_shell']
        for zone in location['zones']:
            if zone['zone_name'] in mappings:
                prompt = (
                    f"Architecture: {architecture}, "
                    f"Description: {zone['definition']}, "
                    f"Anchored objects: {zone['anchored_elements']}"
                )
                views[mappings[zone['zone_name']]] = {"background": prompt}

    for view in views:
        for camera in views[view]:
            yield f"""
>> ALIAS: {normalize(view)}_{camera.upper()}
create_background cinematic widescreen composition with generous negative space at left and right frame edges,
primary focal objects positioned safely within center 60% of frame, smooth flooring extends toward edges to provide
clean tracking margins for camera movement, {views[view][camera]}, Seed: {SEED}"""

def build_action_prompt_for_char(action: str, char_name: str) -> str:
    """
    Build a prompt for a single character's action.
    Strips the character name from the action string.
    """
    return action.replace(char_name, '').strip()

def build_wide_action_prompt(beat, char_names):
    """
    Build a prompt for the wide/composite action shot.
    Describes all characters' actions in the beat.
    """
    return "; ".join(beat['actions'])

def _get_per_speaker_value(beat, key: str, speaker: str, default: str):
    """
    Helper: supports both scalar and per-speaker dict formats.
    """
    val = beat.get(key, default)
    if isinstance(val, dict):
        return val.get(speaker, default)
    return val

def build_dialog_closeup_prompt(beat, speaker: str) -> str:
    """
    Build a closeup performance prompt for dialog.
    Dialog closeups MUST NOT include actions.
    They include ONLY:
      - facial expression
      - head gesture
      - vocal tone
    """
    facial = _get_per_speaker_value(beat, 'facial_state', speaker, 'neutral')
    head   = _get_per_speaker_value(beat, 'head_gesture', speaker, 'none')
    tone   = _get_per_speaker_value(beat, 'tone', speaker, 'neutral')

    if head == 'none':
        head_desc = "no notable head movement"
    else:
        head_desc = f"head gesture {head}"

    return (
        f"closeup shot of {speaker} performing: "
        f"facial expression {facial}, {head_desc}, vocal tone {tone}"
    )

def render_beats_actions(assets, actions):
    """
    Wide + medium action rendering.

    For each beat:
      - ONE wide shot with all characters that have actions in that beat.
      - ONE medium shot per character, performing only their own action.
    """
    names = {}
    for char in assets['characters']:
        bio = char['biography']
        names[char['name']] = f"{char['name']} ({bio['gender']}, {bio['clothing']})"

    for beat in actions:
        if not beat.get('actions'):
            continue

        zone_alias = f"{normalize(beat['zone'])}_BACKGROUND"

        # Collect all characters that appear in any action in this beat
        chars_in_beat = []
        for action in beat['actions']:
            for char_name in names:
                if char_name in action and char_name not in chars_in_beat:
                    chars_in_beat.append(char_name)

        if not chars_in_beat:
            continue

        # WIDE SHOT: one image/video per beat with ALL characters
        char_assets_clause = " and ".join(f"{normalize(c)} asset" for c in chars_in_beat)
        wide_prompt = build_wide_action_prompt(beat, chars_in_beat)

        print(f"""
>> ALIAS: BEAT_{beat["beat"]}_WIDE_ACTION
composite_scene {zone_alias} asset and {char_assets_clause}, {wide_prompt}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

        print(f"""
>> ALIAS: BEAT_{beat["beat"]}_WIDE_ACTION_VIDEO
image_to_video BEAT_{beat["beat"]}_WIDE_ACTION asset, {wide_prompt}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

        # MEDIUM SHOTS: one per character, using only that character's action
        for char_name in chars_in_beat:
            char_action = None
            for action in beat['actions']:
                if char_name in action:
                    char_action = action
                    break
            if not char_action:
                continue

            char_alias = normalize(char_name)
            medium_prompt = build_action_prompt_for_char(char_action, char_name)

            print(f"""
>> ALIAS: BEAT_{beat["beat"]}_{char_alias}_ACTION
composite_scene {zone_alias} asset and {char_alias} asset, {medium_prompt}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

            print(f"""
>> ALIAS: BEAT_{beat["beat"]}_{char_alias}_ACTION_VIDEO
image_to_video BEAT_{beat["beat"]}_{char_alias}_ACTION asset, {medium_prompt}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

def render_beats_dialog(assets, actions):
    """
    Dialog renderer.

    For each beat with dialog:
      - pick background by zone
      - pick character by speaker
      - create closeup composite with facial_state/head_gesture/tone (NO actions)
      - pass to speech_to_video with VOICE alias and dialog line
    """
    char_aliases = {c['name']: normalize(c['name']) for c in assets['characters']}

    for beat in actions:
        dialog_list = beat.get('dialog') or []
        if not dialog_list:
            continue

        zone_alias = f"{normalize(beat['zone'])}_BACKGROUND"

        for dlg in dialog_list:
            speaker = dlg['speaker']
            line = dlg['line']
            if speaker not in char_aliases:
                continue

            speaker_alias = char_aliases[speaker]
            closeup_prompt = build_dialog_closeup_prompt(beat, speaker)

            print(f"""
>> ALIAS: BEAT_{beat["beat"]}_{speaker_alias}_DIALOG_FRAME
composite_scene {zone_alias} asset and {speaker_alias} asset,
{closeup_prompt}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

            print(f"""
>> ALIAS: BEAT_{beat["beat"]}_{speaker_alias}_DIALOG_VIDEO
speech_to_video using=BEAT_{beat["beat"]}_{speaker_alias}_DIALOG_FRAME
audio={speaker_alias}_VOICE
text="{line}"
Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

def main():
    basepath = sys.argv[1]
    with open(f"{basepath}/output/registry.json") as ass:
        assets = json.load(ass)
    with open(f"{basepath}/output/complete.json") as act:
        actions = json.load(act)

    for x in get_identity(assets):
        print(x)

    mappings = create_zone_mapping(assets, actions)
    for x in get_backgrounds(assets, mappings):
        print(x)

    render_beats_actions(assets, actions)
    render_beats_dialog(assets, actions)

if __name__ == "__main__":
    main()
