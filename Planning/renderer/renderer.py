#!/usr/bin/env python3
import os, sys, json, re

# ONE import for all template output
from templates import Templates

# ONE import for all shared helpers
from utils import (
    canonical, normalize, soft_normalize,
    resolve_character_mentions, resolve_pronouns,
    create_zone_mapping, resolve_zone_alias,
    get_beat_characters, bind_identity_first_only
)

POSTURE = {}


# ---------------------------------------------------------
# CHARACTER SHEETS + VOICES
# ---------------------------------------------------------

def get_identity(assets, T):
    for char in assets['characters']:
        bio = char['biography']
        name = bio['name']
        alias = f"CHAR_{normalize(name)}"

        description = (
            f"{bio['gender']}, Age: {bio['age']}, "
            f"{bio['race']}/{bio['ethnicity_species']}, "
            f"{bio['appearance']},{bio['hair']}, {bio['clothing']}"
        )

        T.character_sheet(alias, description)
        T.voice_design(alias, ",".join(description.split(",")[:3]))


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
# BACKGROUNDS
# ---------------------------------------------------------

def get_backgrounds(assets, mappings, T):
    for location in assets['locations']:
        architecture = location['architectural_shell']
        for zone in location['zones']:
            if zone['zone_name'] not in mappings:
                continue

            zone_key = mappings[zone['zone_name']]
            alias = normalize(zone_key)

            T.background(
                alias,
                architecture,
                zone['definition'],
                zone['anchored_elements']
            )


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


def render_beats_actions(assets, actions, mappings, T):
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

        # SINGLE CHARACTER
        if len(chars_in_actions) == 1:
            char = chars_in_actions[0]
            char_alias = char_aliases[char]

            char_action = next(
                (a for a in resolved_actions if soft_normalize(canonical(a.split(" ", 1)[0])) == soft_normalize(char)),
                resolved_actions[0]
            )
            char_pose = resolve_character_mentions(char_action, names)
            _, motion_actions = bind_identity_first_only(resolved_actions, names)

            alias = f"BEAT_{beat['beat']}_{normalize(char)}_ACTION"

            T.action_medium(alias, zone_alias, char_alias, char_pose)
            T.action_video(
                f"{alias}_VIDEO",
                alias,
                motion_actions,
                duration=5
            )
            continue

        # MULTI-CHARACTER
        pose_action_map, motion_actions = bind_identity_first_only(resolved_actions, names)
        char_assets = " and ".join(f"{char_aliases[c]} asset" for c in chars_in_actions)
        pose_block = format_pose_block(pose_action_map)

        alias = f"BEAT_{beat['beat']}_WIDE_ACTION"

        T.action_wide(alias, zone_alias, char_assets, pose_block)
        T.action_video(
            f"{alias}_VIDEO",
            alias,
            motion_actions,
            duration=5
        )


# ---------------------------------------------------------
# DIALOG RENDERING
# ---------------------------------------------------------

def _get_per_speaker_value(beat, key, speaker, default):
    val = beat.get(key, default)
    if isinstance(val, dict):
        return val.get(speaker, default)
    return val

def split_dialog_into_sentences(line):
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


def render_beats_dialog(assets, actions, mappings, T):
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

                T.dialog_closeup(base_alias, zone_alias, speaker_alias, dialog_pose_prompt_close)
                T.dialog_medium(f"{base_alias}_medium", zone_alias, speaker_alias, dialog_pose_prompt)

                motion_prompt = T.dialog_motion_prompt(dlg['speaker'], facial, head)

                T.dialog_motion(motion_alias, base_alias, motion_prompt, duration=2)
                T.dialog_final(final_alias, base_alias, f"{speaker_alias}_VOICE", sentence)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    basepath = sys.argv[1]

    with open(f"{basepath}/output/registry.json") as ass:
        assets = json.load(ass)
    with open(f"{basepath}/output/complete.json") as act:
        actions = json.load(act)

    T = Templates()   # ← ONE OBJECT

    get_identity(assets, T)
    mappings = create_zone_mapping(assets, actions)
    get_backgrounds(assets, mappings, T)

    render_beats_actions(assets, actions, mappings, T)
    render_beats_dialog(assets, actions, mappings, T)

    if len(sys.argv) > 2 and sys.argv[2] in ("images", "all", "videos", "identity"):
        mode = sys.argv[2]
    else:
        mode = os.environ.get("MODE", "all")

    T.buffer.dump(mode)


if __name__ == "__main__":
    main()
