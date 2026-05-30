import os, sys, json
sys.path.append('./lib')
from config import load_environ
from qwen_llm import llm_analyze_media  # unused but kept for consistency

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "123456"))

def normalize(name: str) -> str:
    name = name.replace(' ', '_').replace('/', '_')
    return ''.join([x for x in name.upper() if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'])

# ---------------------------------------------------------
# CHARACTER IDENTITY BINDING
# ---------------------------------------------------------

def build_identity_map(assets):
    """
    Returns:
        { "George": "George (Male, flannel pajamas)", ... }
    """
    names = {}
    for char in assets['characters']:
        bio = char['biography']
        desc = f"{char['name']} ({bio['gender']}, {bio['clothing']})"
        names[char['name']] = desc
    return names

def bind_identity(action: str, names: dict) -> str:
    """
    Replace character name with full identity descriptor.
    Ensures the model never invents a second subject.
    """
    for char, ident in names.items():
        if char in action:
            return action.replace(char, ident)
    return action  # fallback, should not happen

# ---------------------------------------------------------
# BACKGROUNDS
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# ACTION RENDERING (WIDE + MEDIUM)
# ---------------------------------------------------------

def render_beats_actions(assets, actions):
    names = build_identity_map(assets)

    for beat in actions:
        beat_actions = beat.get('actions') or []
        if not beat_actions:
            continue

        zone_alias = f"{normalize(beat['zone'])}_BACKGROUND"

        # Identify characters in this beat
        chars_in_beat = []
        for action in beat_actions:
            for char in names:
                if char in action and char not in chars_in_beat:
                    chars_in_beat.append(char)

        if not chars_in_beat:
            continue

        # -----------------------------
        # WIDE SHOT
        # -----------------------------
        char_assets = " and ".join(f"{normalize(c)} asset" for c in chars_in_beat)
        wide_prompt = "; ".join(bind_identity(a, names) for a in beat_actions)

        print(f"""
>> ALIAS: BEAT_{beat["beat"]}_WIDE_ACTION
composite_scene {zone_alias} asset and {char_assets}, {wide_prompt}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

        print(f"""
>> ALIAS: BEAT_{beat["beat"]}_WIDE_ACTION_VIDEO
image_to_video BEAT_{beat["beat"]}_WIDE_ACTION asset, {wide_prompt}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

        # -----------------------------
        # MEDIUM SHOTS (one per character)
        # -----------------------------
        for char in chars_in_beat:
            # find this character's action
            char_action = next((a for a in beat_actions if char in a), None)
            if not char_action:
                continue

            char_alias = normalize(char)
            medium_prompt = bind_identity(char_action, names)

            print(f"""
>> ALIAS: BEAT_{beat["beat"]}_{char_alias}_ACTION
composite_scene {zone_alias} asset and {char_alias} asset, {medium_prompt}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

            print(f"""
>> ALIAS: BEAT_{beat["beat"]}_{char_alias}_ACTION_VIDEO
image_to_video BEAT_{beat["beat"]}_{char_alias}_ACTION asset, {medium_prompt}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

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

# ---------------------------------------------------------
# DIALOG CLOSEUPS (NO ACTIONS)
# ---------------------------------------------------------

def _get_per_speaker_value(beat, key, speaker, default):
    val = beat.get(key, default)
    if isinstance(val, dict):
        return val.get(speaker, default)
    return val

def build_dialog_closeup_prompt(beat, speaker, names):
    facial = _get_per_speaker_value(beat, 'facial_state', speaker, 'neutral')
    head   = _get_per_speaker_value(beat, 'head_gesture', speaker, 'none')
    tone   = _get_per_speaker_value(beat, 'tone', speaker, 'neutral')

    head_desc = "no notable head movement" if head == "none" else f"head gesture {head}"

    return (
        f"closeup shot of {names[speaker]} performing: "
        f"facial expression {facial}, {head_desc}, vocal tone {tone}"
    )

def render_beats_dialog(assets, actions):
    names = build_identity_map(assets)
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
            closeup_prompt = build_dialog_closeup_prompt(beat, speaker, names)

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

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    basepath = sys.argv[1]
    with open(f"{basepath}/output/registry.json") as ass:
        assets = json.load(ass)
    with open(f"{basepath}/output/complete.json") as act:
        actions = json.load(act)

    # Character sheets + voices
    for x in get_identity(assets):
        print(x)

    # Backgrounds
    mappings = create_zone_mapping(assets, actions)
    for x in get_backgrounds(assets, mappings):
        print(x)

    # Action shots
    render_beats_actions(assets, actions)

    # Dialog closeups
    render_beats_dialog(assets, actions)

if __name__ == "__main__":
    main()
