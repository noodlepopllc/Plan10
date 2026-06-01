#!/usr/bin/env python3
import os, sys, json, re
sys.path.append('./lib')
from config import load_environ

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED", "123456"))

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
# NORMALIZATION / FUZZY MATCHING
# ---------------------------------------------------------

def normalize(name: str) -> str:
    name = name.replace(' ', '_').replace('/', '_')
    return ''.join([x for x in name.upper() if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'])

def soft_normalize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).lower()

def make_fuzzy_pattern(name: str) -> str:
    parts = []
    for ch in name:
        if ch.isalnum():
            parts.append(re.escape(ch) + r"[^A-Za-z0-9]*")
        else:
            parts.append(re.escape(ch) + r"*")
    pattern = "".join(parts)
    pattern += r"(?:['’]s)?"
    return pattern

def resolve_character_mentions(text: str, names: dict) -> str:
    rewritten = text
    for char, ident in names.items():
        if ident in rewritten:
            continue
        pattern = make_fuzzy_pattern(char)
        rewritten = re.sub(pattern, ident, rewritten, flags=re.IGNORECASE)
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
# IDENTITY BINDING
# ---------------------------------------------------------

def build_identity_map(assets):
    names = {}
    for char in assets['characters']:
        bio = char['biography']
        desc = f"{char['name']} ({bio['gender']}, {bio['clothing']})"
        names[char['name']] = desc
    return names

def bind_identity(action: str, names: dict) -> str:
    if not action:
        return action
    first_token = action.split(" ", 1)[0]
    first_clean = soft_normalize(first_token)
    for char, ident in names.items():
        if soft_normalize(char) == first_clean:
            parts = action.split(" ", 1)
            if len(parts) > 1:
                return f"{ident} {parts[1]}"
            return ident
    return action

def bind_identity_first_only(actions, names):
    if not actions:
        return {}, ""

    if isinstance(actions, str):
        actions = [actions]

    used_identity = {char: False for char in names}
    pose_actions = {}
    motion_parts = []

    for action in actions:
        action = action.strip()
        if not action:
            continue

        parts = action.split(" ", 1)
        first_word_raw = parts[0]
        first_char = None

        for char in names:
            if soft_normalize(char) == soft_normalize(first_word_raw):
                first_char = char
                break

        if first_char is not None:
            identity = names[first_char]

            if not used_identity[first_char]:
                used_identity[first_char] = True
                rewritten = identity + " " + parts[1] if len(parts) > 1 else identity

                if first_char not in pose_actions:
                    pose_actions[first_char] = rewritten
            else:
                rewritten = parts[1] if len(parts) > 1 else ""
        else:
            rewritten = action

        rewritten = resolve_character_mentions(rewritten, names)
        motion_parts.append(rewritten)

    motion = motion_parts[0] if len(motion_parts) == 1 else motion_parts[0] + ", " + ", ".join(motion_parts[1:])
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
# ACTION RENDERING (PASS A)
# ---------------------------------------------------------

def _get_chars_in_actions_only(beat_actions, names):
    chars = []
    for a in beat_actions:
        clean_a = soft_normalize(a)
        for char in names:
            if soft_normalize(char) in clean_a and char not in chars:
                chars.append(char)
    return chars

def format_pose_block(pose_actions):
    blocks = []
    for char, action in pose_actions.items():
        parts = action.split(" ", 1)
        pose = parts[1] if len(parts) > 1 else action
        blocks.append(f"{char} asset ({pose})")
    return " and ".join(blocks)

def render_beats_actions(assets, actions, mappings, commands: CommandBuffer):
    names = build_identity_map(assets)
    char_aliases = {c['name']: f"CHAR_{normalize(c['name'])}" for c in assets['characters']}

    for beat in actions:
        pose_action = None
        if continuity := beat.get('continuity', None):
            pose_action = continuity.get('object_introductions', None)[0]["action"] if continuity.get('object_introductions', None) else None
        if not pose_action:
            pose_action = beat['actions'][0]

        beat_actions = [pose_action] + [a for a in beat['actions'] if a != pose_action] or []
        if not beat_actions:
            continue

        chars_in_actions = _get_chars_in_actions_only(beat_actions, names)
        if not chars_in_actions:
            continue

        zone_alias = resolve_zone_alias(beat['zone'], mappings)

        if len(chars_in_actions) == 1:
            char = chars_in_actions[0]
            char_alias = char_aliases[char]

            char_action = next(
                (a for a in beat_actions if soft_normalize(a.split(" ", 1)[0]) == soft_normalize(char)),
                beat_actions[0]
            )
            char_pose = bind_identity(char_action, names)
            _, motion_actions = bind_identity_first_only(beat_actions, names)

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

        pose_action_map, motion_actions = bind_identity_first_only(beat_actions, names)
        char_assets = " and ".join(f"{char_aliases[c]} asset" for c in chars_in_actions)

        for c in chars_in_actions:
            if c not in pose_action_map:
                identity = names[c]
                pose_action_map[c] = f"{identity} stands neutrally"

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
# DIALOG CLOSEUPS (PASS B, DIALOG‑FORWARD)
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

    major_parts = re.split(r'([.!?…;])', text)
    major_units = []
    for i in range(0, len(major_parts) - 1, 2):
        unit = (major_parts[i].strip() + major_parts[i+1]).strip()
        if unit:
            major_units.append(unit)

    if not major_units:
        major_units = [text]

    clause_regex = r",|;| but | and | so | because | although | though | however "
    natural_units = []
    for unit in major_units:
        words = unit.split()
        if len(words) <= 12:
            natural_units.append(unit)
            continue

        sub = re.split(clause_regex, unit)
        sub = [s.strip() for s in sub if s.strip()]
        natural_units.extend(sub)

    merged = []
    i = 0
    while i < len(natural_units):
        cur = natural_units[i].strip()
        wc = len(cur.split())

        if wc >= 5 or cur.endswith(('.', '?', '!')):
            merged.append(cur)
            i += 1
            continue

        if i + 1 < len(natural_units):
            merged.append(cur + " " + natural_units[i+1].strip())
            i += 2
        else:
            if merged:
                merged[-1] = merged[-1] + " " + cur
            else:
                merged.append(cur)
            i += 1

    final_units = []
    for unit in merged:
        words = unit.split()
        if len(words) <= 12:
            final_units.append(unit)
            continue

        start = 0
        while start < len(words):
            end = min(start + 10, len(words))
            chunk = " ".join(words[start:end])
            final_units.append(chunk)
            start = end

    final_units = [u.strip() for u in final_units if u.strip()]

    bad_endings = {"but", "and", "or", "so", "but by", "and by"}
    cleaned = []
    skip_next = False

    for i, unit in enumerate(final_units):
        if skip_next:
            skip_next = False
            continue

        words = unit.split()
        if not words:
            continue

        last_one = words[-1]
        last_two = " ".join(words[-2:]) if len(words) >= 2 else last_one

        if last_one in bad_endings or last_two in bad_endings:
            if i + 1 < len(final_units):
                merged_unit = unit + " " + final_units[i+1]
                cleaned.append(merged_unit.strip())
                skip_next = True
            else:
                if cleaned:
                    cleaned[-1] = cleaned[-1] + " " + unit
                else:
                    cleaned.append(unit)
        else:
            cleaned.append(unit)

    return cleaned

def render_beats_dialog(assets, actions, mappings, commands: CommandBuffer):
    names = build_identity_map(assets)
    char_aliases = {c['name']: f"CHAR_{normalize(c['name'])}" for c in assets['characters']}

    for beat in actions:
        dialog_list = [
            d for d in beat.get('dialog') or []
            if d.get("line") and d.get("line").strip().lower() not in ("", "none")
        ]
        if not dialog_list:
            continue

        zone_alias = resolve_zone_alias(beat['zone'], mappings)

        for idx, dlg in enumerate(dialog_list, start=1):
            speaker = dlg['speaker']
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

                dialog_pose_prompt_close = f"{speaker} ({tone} tone, facial expression {facial})"

                dialog_pose_prompt = f"{speaker} ({tone} tone, facial expression {facial}, head gesture {head})"

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
                    f"{speaker}, calm and still, "
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
    
    if len(sys.argv) > 2 and sys.argv[2] in ("images","all","videos", "identity"):
        mode = sys.argv[2]
    else:
        mode = os.environ.get("MODE", "all")
    commands.dump(mode)

if __name__ == "__main__":
    main()
