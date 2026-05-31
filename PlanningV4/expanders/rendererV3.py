import os, sys, json, re
sys.path.append('./lib')
from config import load_environ

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "123456"))

def normalize(name: str) -> str:
    name = name.replace(' ', '_').replace('/', '_')
    return ''.join([x for x in name.upper() if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'])

# ---------------------------------------------------------
# ZONE LABEL NORMALIZATION + MAPPING
# ---------------------------------------------------------

def _clean_zone_label(label: str) -> str:
    if not label:
        return ""
    label = label.lower().strip()
    label = re.sub(r"^\d+\.\s*", "", label)
    label = re.sub(r"[^a-z0-9\s]", " ", label)
    label = re.sub(r"\s+", " ", label)
    return label

def create_zone_mapping(registry, story):
    """
    Map registry zone_name -> story beat['zone'] label.
    Matching is done on normalized labels, not raw strings.
    """
    mappings = {}

    # Pre-normalize beat zones
    beat_zones = {}
    for beat in story:
        bz_raw = beat.get('zone', '')
        bz_clean = _clean_zone_label(bz_raw)
        if bz_clean:
            beat_zones[bz_clean] = beat['zone']

    for location in registry['locations']:
        for zone in location['zones']:
            zn_raw = zone['zone_name']
            zn_clean = _clean_zone_label(zn_raw)

            if zn_clean in beat_zones:
                mappings[zone['zone_name']] = beat_zones[zn_clean]

    return mappings

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
    { "Sally": "Sally (Female, A flowing, ...)", ... }
    """
    names = {}
    for char in assets['characters']:
        bio = char['biography']
        desc = f"{char['name']} ({bio['gender']}, {bio['clothing']})"
        names[char['name']] = desc
    return names

def bind_identity(action: str, names: dict) -> str:
    """
    Simple identity replacement for single actions (used for medium pose).
    """
    for char, ident in names.items():
        if action.startswith(char):
            parts = action.split(" ", 1)
            if len(parts) > 1:
                return f"{ident} {parts[1]}"
            return ident
    return action

def bind_identity_first_only(actions, names):
    """
    Multi-character safe version.

    For each character:
      - FIRST action for that character: replace FIRST WORD with identity
      - Continuation actions: remove FIRST WORD only

    Returns:
      pose_action  -> first rewritten action (for still)
      motion       -> full comma-separated motion chain
    """
    if not actions:
        return "", ""

    # Guard: if a single string sneaks in, wrap it
    if isinstance(actions, str):
        actions = [actions]

    used_identity = set()
    pose_action = None
    motion_parts = []

    for action in actions:
        action = action.strip()
        if not action:
            continue

        parts = action.split(" ", 1)
        first_word = parts[0]

        if first_word in names:
            identity = names[first_word]

            if first_word not in used_identity:
                used_identity.add(first_word)
                if len(parts) > 1:
                    rewritten = identity + " " + parts[1]
                else:
                    rewritten = identity

                if pose_action is None:
                    pose_action = rewritten

                motion_parts.append(rewritten)
            else:
                if len(parts) > 1:
                    rewritten = parts[1]
                else:
                    rewritten = ""
                motion_parts.append(rewritten)
        else:
            if pose_action is None:
                pose_action = action
            motion_parts.append(action)

    if not motion_parts:
        motion = pose_action or ""
    elif len(motion_parts) == 1:
        motion = motion_parts[0]
    else:
        motion = motion_parts[0] + ", " + ", ".join(motion_parts[1:])

    return pose_action, motion

# ---------------------------------------------------------
# ZONE BACKGROUNDS
# ---------------------------------------------------------

def get_backgrounds(registry, mappings):
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
# ACTION RENDERING (SHOT SELECTION)
# ---------------------------------------------------------

def _get_chars_in_beat(beat_actions, names):
    chars = []
    for a in beat_actions:
        first = a.split(" ", 1)[0]
        if first in names and first not in chars:
            chars.append(first)
    return chars

def render_beats_actions(assets, actions):
    """
    Shot selection rule:
      - If a beat has ONE character with actions -> MEDIUM only
      - If a beat has TWO OR MORE characters with actions -> WIDE only
    """
    names = build_identity_map(assets)
    char_aliases = {c['name']: f"CHAR_{normalize(c['name'])}" for c in assets['characters']}

    for beat in actions:
        beat_actions = beat.get('actions') or []
        if not beat_actions:
            continue

        chars_in_beat = _get_chars_in_beat(beat_actions, names)
        if not chars_in_beat:
            continue

        zone_base = normalize(beat['zone'])
        zone_alias = f"{zone_base}_BACKGROUND"

        # Single-character beat → MEDIUM only
        if len(chars_in_beat) == 1:
            char = chars_in_beat[0]
            char_alias = char_aliases[char]

            # Pose = identity-bound first action for this character
            char_action = next((a for a in beat_actions if a.startswith(char)), beat_actions[0])
            char_pose = bind_identity(char_action, names)

            # Build full motion chain ONCE
            pose_action, motion_actions = bind_identity_first_only(beat_actions, names)

            # Medium still
            print(f"""
        >> ALIAS: BEAT_{beat["beat"]}_{normalize(char)}_ACTION
        composite_scene {zone_alias} asset and {char_alias} asset, {char_pose}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
        """)

            # Medium video uses FULL motion chain
            print(f"""
        >> ALIAS: BEAT_{beat["beat"]}_{normalize(char)}_ACTION_VIDEO
        image_to_video BEAT_{beat["beat"]}_{normalize(char)}_ACTION asset, {motion_actions}, Width: {WIDTH}, Height: {HEIGHT}, Duration: 5, Seed: {SEED}
        """)
            continue


        # Multi-character beat -> WIDE only
        pose_action, motion_actions = bind_identity_first_only(beat_actions, names)
        char_assets = " and ".join(f"{char_aliases[c]} asset" for c in chars_in_beat)

        print(f"""
>> ALIAS: BEAT_{beat["beat"]}_WIDE_ACTION
composite_scene {zone_alias} asset and {char_assets}, {pose_action}, Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
""")

        print(f"""
>> ALIAS: BEAT_{beat["beat"]}_WIDE_ACTION_VIDEO
image_to_video BEAT_{beat["beat"]}_WIDE_ACTION asset, {motion_actions}, Width: {WIDTH}, Height: {HEIGHT}, Duration: 5, Seed: {SEED}
""")

# ---------------------------------------------------------
# DIALOG CLOSEUPS
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
    char_aliases = {c['name']: f"CHAR_{normalize(c['name'])}" for c in assets['characters']}

    for beat in actions:
        raw_dialog = beat.get('dialog') or []

        dialog_list = [
            d for d in raw_dialog
            if d.get("line") and d.get("line").strip().lower() not in ("", "none")
        ]
        if not dialog_list:
            continue

        zone_base = normalize(beat['zone'])
        zone_alias = f"{zone_base}_BACKGROUND"

        base_alias = f"BEAT_{beat['beat']}_WIDE_ACTION"

        for dlg in dialog_list:
            speaker = dlg['speaker']
            line = dlg['line']
            if speaker not in char_aliases:
                continue

            speaker_alias = char_aliases[speaker]
            facial = _get_per_speaker_value(beat, 'facial_state', speaker, 'neutral')
            head   = _get_per_speaker_value(beat, 'head_gesture', speaker, 'none')

            print(f"""
>> ALIAS: BEAT_{beat["beat"]}_{normalize(speaker)}_DIALOG_FRAME
edit_image {base_alias} asset,
reference_face: {speaker_alias} asset,
subject: the person matching the description of {speaker_alias} asset,
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

    for x in get_identity(assets):
        print(x)

    mappings = create_zone_mapping(assets, actions)
    for x in get_backgrounds(assets, mappings):
        print(x)

    render_beats_actions(assets, actions)
    render_beats_dialog(assets, actions)

if __name__ == "__main__":
    main()
