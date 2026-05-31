import os, sys, json
sys.path.append('./lib')
from config import load_environ
from qwen_llm import llm_analyze_media  # kept for consistency, unused here

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "123456"))

def normalize(name: str) -> str:
    name = name.replace(' ', '_').replace('/', '_')
    return ''.join([x for x in name.upper() if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'])

# ---------------------------------------------------------
# CHARACTER SHEETS + VOICES (CHAR_ ALIASES)
# ---------------------------------------------------------

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
        char_alias = f"CHAR_{normalize(name)}"
        yield (
            f">> ALIAS: {char_alias}\n"
            f"create a character sheet of {description[1]}, Seed: {SEED}\n\n"
            f">> ALIAS: {char_alias}_VOICE\n"
            f"design a voice for {','.join(description[1].split(',')[:3])}\n"
        )

# ---------------------------------------------------------
# IDENTITY BINDING (TEXT SIDE)
# ---------------------------------------------------------

def build_identity_map(assets):
    """
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
    Replace character name with full identity descriptor in the text.
    """
    for char, ident in names.items():
        if char in action:
            return action.replace(char, ident)
    return action

# ---------------------------------------------------------
# ZONE MAPPING + BACKGROUNDS (ZONES = CAMERA ANGLES)
# ---------------------------------------------------------

def create_zone_mapping(registry, story):
    """
    Map registry zone_name -> story beat['zone'] label.
    Zones are camera angles inside a location.
    """
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
    """
    ONE background per zone (zone = camera angle).
    All characters that use that zone share this plate.
    """
    for location in registry['locations']:
        architecture = location['architectural_shell']

        for zone in location['zones']:
            if zone['zone_name'] not in mappings:
                continue

            zone_key = mappings[zone['zone_name']]
            zone_alias = normalize(zone_key)

            prompt = (
                f"Architecture: {architecture}, "
                f"Description: {zone['definition']}, "
                f"Anchored objects: {zone['anchored_elements']}"
            )

            yield f"""
>> ALIAS: {zone_alias}_BACKGROUND
create_background cinematic widescreen composition with generous negative space at left and right frame edges,
primary focal objects positioned safely within center 60% of frame, smooth flooring extends toward edges to provide
clean tracking margins for camera movement, {prompt}, Seed: {SEED}"""

# ---------------------------------------------------------
# ACTION RENDERING (WIDE + MEDIUM, SAME ZONE BACKGROUND)
# ---------------------------------------------------------

def render_beats_actions(assets, actions):
    """
    For each beat:
      - ONE wide shot with all characters in that beat (zone background)
      - ONE medium shot per character (same zone background)
    Zones are already camera angles; we do NOT change backgrounds per character.
    """
    names = build_identity_map(assets)

    # map plain names -> CHAR_ aliases
    char_aliases = {c['name']: f"CHAR_{normalize(c['name'])}" for c in assets['characters']}

    for beat in actions:
        beat_actions = beat.get('actions') or []
        if not beat_actions:
            continue

        zone_base = normalize(beat['zone'])
        zone_alias = f"{zone_base}_BACKGROUND"

        # Identify characters in this beat
        chars_in_beat = []
        for action in beat_actions:
            for char in names:
                if char in action and char not in chars_in_beat:
                    chars_in_beat.append(char)

        if not chars_in_beat:
            continue

        # -----------------------------
        # WIDE SHOT (zone background)
        # -----------------------------
        char_assets = " and ".join(f"{char_aliases[c]} asset" for c in chars_in_beat)
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
        # MEDIUM SHOTS (per character, same zone background)
        # -----------------------------
        for char in chars_in_beat:
            char_action = next((a for a in beat_actions if char in a), None)
            if not char_action:
                continue

            char_alias = char_aliases[char]
            medium_prompt = bind_identity(char_action, names)

            print(f"""
>> ALIAS: BEAT_{beat["beat"]}_{normalize(char)}_ACTION
composite_scene {zone_alias} asset and {char_alias} asset, {medium_prompt}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

            print(f"""
>> ALIAS: BEAT_{beat["beat"]}_{normalize(char)}_ACTION_VIDEO
image_to_video BEAT_{beat["beat"]}_{normalize(char)}_ACTION asset, {medium_prompt}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

# ---------------------------------------------------------
# DIALOG CLOSEUPS (SAME ZONE BACKGROUND, CHAR_ ALIASES)
# ---------------------------------------------------------

def _get_per_speaker_value(beat, key, speaker, default):
    val = beat.get(key, default)
    if isinstance(val, dict):
        return val.get(speaker, default)
    return val

def build_dialog_closeup_prompt(beat, speaker, names):
    """
    Dialog closeup:
      - NO body actions
      - ONLY facial_state, head_gesture, tone
      - Identity-bound description
    """
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
    char_aliases = {c['name']: f"CHAR_{normalize(c['name'])}" for c in assets['characters']}

    for beat in actions:
        dialog_list = beat.get('dialog') or []
        if not dialog_list:
            continue

        zone_base = normalize(beat['zone'])
        zone_alias = f"{zone_base}_BACKGROUND"

        for dlg in dialog_list:
            speaker = dlg['speaker']
            line = dlg['line']
            if speaker not in char_aliases:
                continue

            speaker_alias = char_aliases[speaker]
            closeup_prompt = build_dialog_closeup_prompt(beat, speaker, names)

            base_alias = f"BEAT_{beat['beat']}_WIDE_ACTION"
            speaker_alias = char_aliases[speaker]

            facial = _get_per_speaker_value(beat, 'facial_state', speaker, 'neutral')
            head   = _get_per_speaker_value(beat, 'head_gesture', speaker, 'none')
            tone   = _get_per_speaker_value(beat, 'tone', speaker, 'neutral')

            print(f"""
>> ALIAS: BEAT_{beat["beat"]}_{normalize(speaker)}_DIALOG_FRAME
edit_image {base_alias} asset,
reference_face: {speaker_alias} asset,
apply: facial expression {facial}, head gesture {head},
crop: closeup_of_face,
Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")




            print(f"""
>> ALIAS: BEAT_{beat["beat"]}_{normalize(speaker)}_DIALOG_VIDEO
speech_to_video using=BEAT_{beat["beat"]}_{normalize(speaker)}_DIALOG_FRAME
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

    # Character sheets + voices (CHAR_ aliases)
    for x in get_identity(assets):
        print(x)

    # Backgrounds (one per zone; zones are camera angles)
    mappings = create_zone_mapping(assets, actions)
    for x in get_backgrounds(assets, mappings):
        print(x)

    # Action shots
    render_beats_actions(assets, actions)

    # Dialog closeups
    render_beats_dialog(assets, actions)

if __name__ == "__main__":
    main()
