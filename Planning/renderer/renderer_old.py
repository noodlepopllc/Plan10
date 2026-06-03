#!/usr/bin/env python3
import os, sys, json, re, unicodedata
sys.path.append('./lib')
from config import load_environ

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "123456"))

POSTURE = {}

# ---------------------------------------------------------
# CANONICAL CHARACTER NAME NORMALIZATION (A2)
# ---------------------------------------------------------

def canonical(name: str) -> str:
    """
    Convert any character name into a stable ASCII-only identity key.
    Removes all symbols, punctuation, unicode variants.
    Keeps only A-Z and 0-9.
    Uppercase for stability.
    """
    if not name:
        return ""

    # Normalize unicode to NFKD and strip accents
    name = unicodedata.normalize("NFKD", name)

    # Keep only ASCII letters and digits
    name = "".join(ch for ch in name if ch.isascii() and ch.isalnum())

    return name.upper()


# ---------------------------------------------------------
# COMMAND BUFFER
# ---------------------------------------------------------

class CommandBuffer:
    def __init__(self):
        self.identity = []
        self.images = []
        self.videos = []

    def add_identity(self, cmd: str):
        self.identity.append(cmd)

    def add_image(self, cmd: str):
        self.images.append(cmd)

    def add_video(self, cmd: str):
        self.videos.append(cmd)

    def dump(self, mode: str = "all"):
        mode = mode.lower()
        if mode in ("all", "identity", "images", "videos"):
            for c in self.identity:
                print(c)
        if mode in ("all", "images", "videos"):
            for c in self.images:
                print(c)
        if mode in ("all", "videos"):
            for c in self.videos:
                print(c)


# ---------------------------------------------------------
# NORMALIZATION / MATCHING
# ---------------------------------------------------------

def normalize(name: str) -> str:
    name = name.replace(' ', '_').replace('/', '_')
    return ''.join([x for x in name.upper() if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'])

def soft_normalize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).lower()

def make_fuzzy_pattern(name: str) -> str:
    return rf"\b{re.escape(name)}\b(?:['’]s)?"


# ---------------------------------------------------------
# SINGLE IDENTITY BINDER
# ---------------------------------------------------------

def resolve_character_mentions(text: str, names: dict) -> str:
    rewritten = text
    for char, ident in names.items():
        pattern = make_fuzzy_pattern(char)
        rewritten = re.sub(
            pattern,
            lambda m: f"{m.group(0)} ({ident})",
            rewritten,
            flags=re.IGNORECASE
        )
    return rewritten


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
    mappings = {}

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

            for bz_clean, bz_raw in beat_zones.items():
                if bz_clean in zn_clean or zn_clean in bz_clean:
                    mappings[zone['zone_name']] = bz_raw
                    break

    return mappings


# ---------------------------------------------------------
# RESOLVE BACKGROUND ALIAS
# ---------------------------------------------------------

def resolve_zone_alias(beat_zone: str, mappings: dict) -> str:
    for reg_zone_name, mapped_beat_zone in mappings.items():
        if mapped_beat_zone == beat_zone:
            return f"{normalize(mapped_beat_zone)}_BACKGROUND"
    return f"{normalize(beat_zone)}_BACKGROUND"


# ---------------------------------------------------------
# CHARACTER SHEETS + VOICES
# ---------------------------------------------------------

def get_identity(assets, commands: CommandBuffer):
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
        cmd = (
            f">> ALIAS: {char_alias}\n"
            f"create a character sheet of {description[1]}, Seed: {SEED}\n\n"
            f">> ALIAS: {char_alias}_VOICE\n"
            f"design a voice for {','.join(description[1].split(',')[:3])}\n"
        )
        commands.add_identity(cmd)


# ---------------------------------------------------------
# IDENTITY MAP (CANONICAL KEYS)
# ---------------------------------------------------------

def build_identity_map(assets, beat=None):
    names = {}

    if beat:
        for k, v in beat['posture'].items():
            POSTURE[canonical(k)] = v

    for char in assets['characters']:
        bio = char['biography']
        ckey = canonical(char['name'])

        posture = POSTURE.get(ckey, 'neutral')
        gender = bio['gender']
        clothing = bio['clothing'].replace('.', '')
        hair = bio.get('hair', '').strip()

        if hair:
            desc = f"{posture}, {gender}, {clothing}, {hair}"
        else:
            desc = f"{posture}, {gender}, {clothing}"

        names[ckey] = desc

    return names


# ---------------------------------------------------------
# PRONOUN RESOLUTION (REFLEXIVE-SAFE)
# ---------------------------------------------------------

def resolve_pronouns(text: str, names: dict):
    if len(names) != 2:
        return text

    chars = list(names.keys())
    c1, c2 = chars[0], chars[1]

    def get_gender(desc):
        parts = [p.strip().lower() for p in desc.split(",")]
        if len(parts) >= 2:
            return parts[1]
        return ""

    g1 = get_gender(names[c1])
    g2 = get_gender(names[c2])

    pronoun_map = {}

    # Female
    if g1 == "female":
        pronoun_map["her"] = c1
        pronoun_map["hers"] = c1
    if g2 == "female":
        pronoun_map["her"] = c2
        pronoun_map["hers"] = c2

    # Male (NO "his" to avoid reflexive corruption)
    if g1 == "male":
        pronoun_map["him"] = c1
    if g2 == "male":
        pronoun_map["him"] = c2

    # Neutral plural
    pronoun_map["them"] = c2
    pronoun_map["their"] = c2

    for p, target in pronoun_map.items():
        text = re.sub(rf"\b{p}\b", target, text, flags=re.IGNORECASE)

    return text


# ---------------------------------------------------------
# BEAT-SCOPED CHARACTER DETECTION
# ---------------------------------------------------------

def get_beat_characters(beat, all_names):
    chars = set()

    # posture
    for c in beat.get("posture", {}).keys():
        c_norm = canonical(c)
        if c_norm in all_names:
            chars.add(c_norm)

    # actions
    for a in beat.get("actions", []):
        ca = soft_normalize(a)
        for c in all_names:
            if soft_normalize(c) in ca:
                chars.add(c)

    # dialog
    for d in beat.get("dialog", []):
        speaker = canonical(d.get("speaker"))
        if speaker in all_names:
            chars.add(speaker)

    return list(chars)


# ---------------------------------------------------------
# POSE EXTRACTOR
# ---------------------------------------------------------

def bind_identity_first_only(actions, names):
    if not actions:
        return {}, ""

    if isinstance(actions, str):
        actions = [actions]

    pose_actions = {}
    motion_parts = []

    for action in actions:
        action = action.strip()
        if not action:
            continue

        parts = action.split(" ", 1)
        first_word_raw = canonical(parts[0])

        for char in names:
            if soft_normalize(char) == soft_normalize(first_word_raw):
                if char not in pose_actions:
                    pose_actions[char] = action
                break

        resolved = resolve_character_mentions(action, names)
        motion_parts.append(resolved)

    motion = ", ".join(motion_parts)
    return pose_actions, motion


# ---------------------------------------------------------
# ZONE BACKGROUNDS
# ---------------------------------------------------------

def get_backgrounds(registry, mappings, commands: CommandBuffer):
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

            cmd = f"""
>> ALIAS: {zone_alias}_BACKGROUND
create_background cinematic composition with tighter framing focused on the primary functional area,
minimize negative space at the frame edges,
center the back wall as the dominant architectural surface,
include only the objects positioned against or near the back wall,
preserve natural perspective and room geometry,
{prompt}, Seed: {SEED}"""
            commands.add_image(cmd)


# ---------------------------------------------------------
# ACTION RENDERING
# ---------------------------------------------------------

def _get_chars_in_actions_only(beat_actions, names):
    chars = []
    pronouns = {"her", "him", "them", "their", "hers", "the other", "each other", "one another"}

    for a in beat_actions:
        clean_a = soft_normalize(a)

        for char in names:
            if soft_normalize(char) in clean_a and char not in chars:
                chars.append(char)

        lower = a.lower()
        if any(p in lower for p in pronouns):
            if len(names) >= 2:
                return list(names.keys())

    return chars


def format_pose_block(pose_actions):
    blocks = []
    for char, action in pose_actions.items():
        parts = action.split(" ", 1)
        pose = parts[1] if len(parts) > 1 else action
        blocks.append(f"{char} asset ({pose})")
    return " and ".join(blocks)


def render_beats_actions(assets, actions, mappings, commands: CommandBuffer):
    char_aliases = {canonical(c['name']): f"CHAR_{normalize(c['name'])}" for c in assets['characters']}

    for beat in actions:
        all_names = build_identity_map(assets, beat)
        beat_chars = get_beat_characters(beat, all_names)
        if not beat_chars:
            continue

        names = {c: all_names[c] for c in beat_chars}

        pose_action = None
        if continuity := beat.get('continuity', None):
            pose_action = continuity.get('object_introductions', None)[0]["action"] if continuity.get('object_introductions', None) else None
        if not pose_action:
            pose_action = beat['actions'][0]

        beat_actions = [pose_action] + [a for a in beat['actions'] if a != pose_action] or []
        beat_actions = [a for a in beat_actions if 'speak' not in a and 'voice' not in a]
        if not beat_actions:
            continue

        resolved_actions = [resolve_pronouns(a, names) for a in beat_actions]

        chars_in_actions = _get_chars_in_actions_only(resolved_actions, names)
        if not chars_in_actions:
            continue

        zone_alias = resolve_zone_alias(beat['zone'], mappings)

        if len(chars_in_actions) == 1:
            char = chars_in_actions[0]
            char_alias = char_aliases[char]

            char_action = next(
                (a for a in resolved_actions if soft_normalize(canonical(a.split(" ", 1)[0])) == soft_normalize(char)),
                resolved_actions[0]
            )
            char_pose = resolve_character_mentions(char_action, names)
            _, motion_actions = bind_identity_first_only(resolved_actions, names)

            img_cmd = f"""
>> ALIAS: BEAT_{beat["beat"]}_{normalize(char)}_ACTION
composite_scene {zone_alias} asset and {char_alias} asset, shot_type: "medium", prompt: "{char_pose}", Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
"""
            vid_cmd = f"""
>> ALIAS: BEAT_{beat["beat"]}_{normalize(char)}_ACTION_VIDEO
image_to_video BEAT_{beat["beat"]}_{normalize(char)}_ACTION asset, {motion_actions}, Width: {WIDTH}, Height: {HEIGHT}, Duration: 5, Seed: {SEED}
"""
            commands.add_image(img_cmd)
            commands.add_video(vid_cmd)
            continue

        pose_action_map, motion_actions = bind_identity_first_only(resolved_actions, names)
        char_assets = " and ".join(f"{char_aliases[c]} asset" for c in chars_in_actions)

        pose_block = format_pose_block(pose_action_map)

        img_cmd = f"""
>> ALIAS: BEAT_{beat["beat"]}_WIDE_ACTION
composite_scene {zone_alias} asset and {char_assets}, shot_type: "two_shot", prompt: "{pose_block}", Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
"""
        vid_cmd = f"""
>> ALIAS: BEAT_{beat["beat"]}_WIDE_ACTION_VIDEO
image_to_video BEAT_{beat["beat"]}_WIDE_ACTION asset, {motion_actions}, Width: {WIDTH}, Height: {HEIGHT}, Duration: 5, Seed: {SEED}
"""
        commands.add_image(img_cmd)
        commands.add_video(vid_cmd)


# ---------------------------------------------------------
# DIALOG RENDERING
# ---------------------------------------------------------

def _get_per_speaker_value(beat, key, speaker, default):
    val = beat.get(key, default)
    if isinstance(val, dict):
        return val.get(speaker, default)
    return val

def split_dialog_into_sentences(line):
    import re
    text = " ".join(line.split()).strip()
    if not text:
        return []
    quoted = re.findall(r'"(.*?)"', text)
    if not quoted:
        return [text]
    if len(quoted) == 1:
        return [quoted[0].strip()]
    first = quoted[0].strip()
    if len(first.split()) <= 1:
        merged = f"{first} {quoted[1].strip()}"
        return [merged]
    return [q.strip() for q in quoted]


def render_beats_dialog(assets, actions, mappings, commands: CommandBuffer):
    char_aliases = {canonical(c['name']): f"CHAR_{normalize(c['name'])}" for c in assets['characters']}

    for beat in actions:
        names = build_identity_map(assets, beat)
        dialog_list = [
            d for d in beat.get('dialog') or []
            if d.get("line") and d.get("line").strip().lower() not in ("", "none")
        ]
        if not dialog_list:
            continue

        zone_alias = resolve_zone_alias(beat['zone'], mappings)

        for idx, dlg in enumerate(dialog_list, start=1):
            speaker = canonical(dlg['speaker'])
            line = dlg['line']

            if speaker not in char_aliases:
                continue

            speaker_alias = char_aliases[speaker]
            facial = _get_per_speaker_value(beat, 'facial_state', speaker, 'neutral')
            head   = _get_per_speaker_value(beat, 'head_gesture', speaker, 'none')
            tone   = _get_per_speaker_value(beat, 'tone', speaker, 'neutral')

            sentences = split_dialog_into_sentences(line)

            for s_idx, sentence in enumerate(sentences, start=1):

                base_alias   = f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_BASE_{idx:02d}_{s_idx:02d}"
                motion_alias = f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_MOTION_{idx:02d}_{s_idx:02d}"
                final_alias  = f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_VIDEO_{idx:02d}_{s_idx:02d}"

                dialog_pose_prompt_close = f"{dlg['speaker']} ({tone} tone, facial expression {facial})"
                dialog_pose_prompt = f"{dlg['speaker']} ({tone} tone, facial expression {facial}, head gesture {head})"

                img_close = f"""
>> ALIAS: {base_alias}
composite_scene {zone_alias} asset and {speaker_alias} asset,
shot_type: "closeup",
prompt: "{dialog_pose_prompt_close}",
Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
"""
                img_medium = f"""
>> ALIAS: {base_alias}_medium
composite_scene {zone_alias} asset and {speaker_alias} asset,
shot_type: "medium",
prompt: "{dialog_pose_prompt}",
Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
"""
                commands.add_image(img_close)
                commands.add_image(img_medium)

                motion_prompt = (
                    f"{dlg['speaker']}, calm and still, "
                    f"lips gently closed, jaw unmoving, "
                    f"eyes with tiny natural micro‑saccades only, "
                    f"stable head position, minimal idle motion, "
                    f"maintain facial expression {facial}, head gesture {head}, "
                    f"no large body motion"
                )

                motion_cmd = f"""
>> ALIAS: {motion_alias}
image_to_video {base_alias}_medium asset, "{motion_prompt}", Width: {WIDTH}, Height: {HEIGHT}, Duration: 2, Seed: {SEED}
"""
                commands.add_video(motion_cmd)

                final_cmd = f"""
>> ALIAS: {final_alias}
dialog_to_video media={base_alias} asset
audio={speaker_alias}_VOICE
text="{sentence}"
Width: {WIDTH}, Height: {HEIGHT}, Seed: {SEED}
"""
                commands.add_video(final_cmd)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    basepath = sys.argv[1]
    with open(f"{basepath}/output/registry.json") as ass:
        assets = json.load(ass)
    with open(f"{basepath}/output/complete.json") as act:
        actions = json.load(act)

    commands = CommandBuffer()

    get_identity(assets, commands)

    mappings = create_zone_mapping(assets, actions)
    get_backgrounds(assets, mappings, commands)

    render_beats_actions(assets, actions, mappings, commands)
    render_beats_dialog(assets, actions, mappings, commands)

    if len(sys.argv) > 2 and sys.argv[2] in ("images", "all", "videos", "identity"):
        mode = sys.argv[2]
    else:
        mode = os.environ.get("MODE", "all")
    commands.dump(mode)


if __name__ == "__main__":
    main()
