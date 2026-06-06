#!/usr/bin/env python3
import os, sys, json, re

# ONE import for all template output
from templates import Templates

# ONE import for all shared helpers
from utils import (
    canonical, normalize, soft_normalize,
    resolve_character_mentions, resolve_pronouns,
    create_zone_mapping, resolve_zone_alias,
    get_beat_characters, bind_identity_first_only,
    split_dialog_sentences
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
            desc = f"{gender}, {clothing}, {hair}"
        else:
            desc = f"{gender}, {clothing}"

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
        # pose is something like "leans in closer" or "wipes counter"
        pose = pose.strip()

        # Normalize pose into a sentence
        if pose.startswith("is ") or pose.startswith("sits") or pose.startswith("stands"):
            blocks.append(f"{char} {pose}.")
        else:
            blocks.append(f"{char} {pose}.")

    return " and ".join(blocks)

def posture_sentence(char, posture):
    if posture == "sitting":
        return f"{char} is sitting."
    if posture == "standing":
        return f"{char} is standing."
    return ""

def strip_leading_name(action, char):
    # Remove leading character name (any case)
    pattern = re.compile(rf"^{char}\s+", re.IGNORECASE)
    return pattern.sub("", action).strip()

'''

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

            pose = POSTURE.get(char, None)
            if pose and pose != "neutral":
                pose_sentence = f"{char} is {pose}."
            else:
                pose_sentence = ""

            char_action = next(
                (a for a in resolved_actions if soft_normalize(canonical(a.split(" ", 1)[0])) == soft_normalize(char)),
                resolved_actions[0]
            )

            char_pose = resolve_character_mentions(char_action, names)
            _, motion_actions = bind_identity_first_only(resolved_actions, names)

            alias = f"BEAT_{beat['beat']}_{normalize(char)}_ACTION"

            T.action_medium(alias, zone_alias, char_alias, f"{pose_sentence} {char_pose}")
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

        sentences = []

        for char in chars_in_actions:
            posture = POSTURE.get(char, None)
            if posture:
                sentences.append(posture_sentence(char, posture))

        for char, action in pose_action_map.items():
            clean_action = strip_leading_name(action, char)
            sentences.append(f"{char} {clean_action}.")


        sentences.append("The scene is framed as a two-shot.")

        wide_prompt = " ".join(sentences)

        T.action_wide(alias, zone_alias, char_assets, wide_prompt)

        T.action_video(
            f"{alias}_VIDEO",
            alias,
            motion_actions,
            duration=5
        )

'''

def render_beats_actions(assets, actions, mappings, T):
    char_aliases = {canonical(c['name']): f"CHAR_{normalize(c['name'])}" for c in assets['characters']}

    for beat in actions:
        all_names = build_identity_map(assets, beat)
        beat_chars = get_beat_characters(beat, all_names)
        if not beat_chars:
            continue

        names = {c: all_names[c] for c in beat_chars}
        zone_alias = resolve_zone_alias(beat['zone'], mappings)

        # --- NEW: starting description integration ---
        start_desc = beat.get('starting_description', {})
        start_sentences = [f"{char} is {desc}." for char, desc in start_desc.items()]

        # --- existing action resolution ---
        resolved_actions = [resolve_pronouns(a, names) for a in beat['actions']]
        pose_action_map, motion_actions = bind_identity_first_only(resolved_actions, names)

        # --- NEW: arc integration ---
        arc = beat.get('arc', "")
        arc_sentence = f"Motion arc: {arc}" if arc else ""

        # --- build prompt ---
        sentences = start_sentences
        for char, action in pose_action_map.items():
            clean_action = strip_leading_name(action, char)
            sentences.append(f"{char} {clean_action}.")
        if arc_sentence:
            sentences.append(arc_sentence)

        wide_prompt = " ".join(sentences)

        alias = f"BEAT_{beat['beat']}_WIDE_ACTION"
        char_assets = " and ".join(f"{char_aliases[c]} asset" for c in beat_chars)

        T.action_wide(alias, zone_alias, char_assets, wide_prompt)
        T.action_video(f"{alias}_VIDEO", alias, motion_actions, duration=5)

# ---------------------------------------------------------
# DIALOG RENDERING
# ---------------------------------------------------------

def _get_per_speaker_value(beat, key, speaker, default):
    val = beat.get(key, default)
    if isinstance(val, dict):
        return val.get(speaker, default)
    return val

def normalize_tone(tone):
    # Map freeform tones into controlled vocabulary
    tone_map = {
        "whispering": "softer",
        "rising": "strained",
        "cracked": "strained",
        "pleading": "strained",
        "hurried": "softer",
        "silent": "neutral"
    }
    return tone_map.get(tone.lower(), tone)

def clean_dialog_line(line):
    # Strip narration cues like 'she whispers' from dialog strings
    # Keep only quoted speech
    # If narration is embedded, remove it
    return re.sub(r'"\s*[^"]*"\s*', lambda m: m.group(0), line).strip()

def render_beats_dialog(assets, actions, mappings, T):
    char_aliases = {
        canonical(c['name']): f"CHAR_{normalize(c['name'])}"
        for c in assets['characters']
    }

    generated_bases = set()

    for beat in actions:
        dialog_list = [
            d for d in (beat.get('dialog') or [])
            if d.get("line") and d.get("line").strip().lower() not in ("", "none")
        ]
        if not dialog_list:
            continue

        zone_alias = resolve_zone_alias(beat['zone'], mappings)
        s_idx = 1

        for dlg in dialog_list:
            speaker = canonical(dlg['speaker'])
            if speaker not in char_aliases:
                continue

            speaker_alias = char_aliases[speaker]
            raw_line = dlg['line']

            # --- NEW: clean dialog line ---
            line = clean_dialog_line(raw_line)

            raw_speaker = dlg['speaker']

            facial_state_map = beat.get('facial_state') or {}
            head_gesture_map = beat.get('head_gesture') or {}
            tone_map         = beat.get('tone') or {}

            facial = facial_state_map.get(raw_speaker, 'neutral')
            head   = head_gesture_map.get(raw_speaker, 'none')
            tone   = normalize_tone(tone_map.get(raw_speaker, 'neutral'))

            # --- NEW: posture alignment from starting_description ---
            start_desc = beat.get('starting_description', {})
            posture = start_desc.get(raw_speaker, None)
            if posture:
                pose_sentence = f"{dlg['speaker']} is {posture}."
            else:
                pose_sentence = ""

            expr_sentence = f"{dlg['speaker']} has a {facial} expression." if facial != "neutral" else ""
            dialog_prompt = " ".join(s for s in [pose_sentence, expr_sentence] if s)

            base_alias = f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_BASE"

            if base_alias not in generated_bases:
                generated_bases.add(base_alias)

                dialog_pose_prompt_close = (
                    f"{dlg['speaker']} ({tone} tone, facial expression {facial})"
                )

                # ONE closeup + ONE medium per beat per speaker
                T.dialog_closeup(base_alias, zone_alias, speaker_alias, dialog_pose_prompt_close)
                T.dialog_medium(f"{base_alias}_medium", zone_alias, speaker_alias, dialog_prompt)

                motion_alias = f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_MOTION_01"
                motion_prompt = T.dialog_motion_prompt(dlg['speaker'], facial, head)
                T.dialog_motion(motion_alias, base_alias, motion_prompt, duration=2)

            final_alias = f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_VIDEO_{s_idx:02d}"
            T.dialog_final(final_alias, base_alias, f"{speaker_alias}_VOICE", line)
            s_idx += 1


'''
def render_beats_dialog(assets, actions, mappings, T):
    # Map canonical character names → CHAR_<NAME> alias
    char_aliases = {
        canonical(c['name']): f"CHAR_{normalize(c['name'])}"
        for c in assets['characters']
    }

    # Cache to ensure we only generate ONE dialog base per (beat, speaker)
    generated_bases = set()

    for beat in actions:
        dialog_list = [
            d for d in (beat.get('dialog') or [])
            if d.get("line") and d.get("line").strip().lower() not in ("", "none")
        ]
        if not dialog_list:
            continue

        zone_alias = resolve_zone_alias(beat['zone'], mappings)
        s_idx = 1

        for dlg in dialog_list:
            speaker = canonical(dlg['speaker'])
            if speaker not in char_aliases:
                continue

            speaker_alias = char_aliases[speaker]
            line = dlg['line']

            raw_speaker = dlg['speaker']  # "Elara", "Nadia"

            facial_state_map = beat.get('facial_state') or {}
            head_gesture_map = beat.get('head_gesture') or {}
            tone_map         = beat.get('tone') or {}

            facial = facial_state_map.get(raw_speaker, 'neutral')
            head   = head_gesture_map.get(raw_speaker, 'none')
            tone   = tone_map.get(raw_speaker, 'neutral')



            # Split dialog into sentences
            sentences = [line]

            # Shared base alias for this beat + speaker
            base_alias = f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_BASE"

            # Only generate the base images ONCE per beat per speaker
            if base_alias not in generated_bases:
                generated_bases.add(base_alias)

                dialog_pose_prompt_close = (
                    f"{dlg['speaker']} ({tone} tone, facial expression {facial})"
                )
                posture = POSTURE.get(speaker, None)
                if posture == "sitting":
                    pose_sentence = f"{dlg['speaker']} is sitting."
                elif posture == "standing":
                    pose_sentence = f"{dlg['speaker']} is standing."
                else:
                    pose_sentence = ""

                expr_sentence = f"{dlg['speaker']} has a {facial} expression." if facial != "neutral" else ""
                dialog_prompt = " ".join(
                    s for s in [pose_sentence, expr_sentence] if s
                )

                # ONE closeup + ONE medium per beat per speaker
                T.dialog_closeup(base_alias, zone_alias, speaker_alias, dialog_pose_prompt_close)
                T.dialog_medium(f"{base_alias}_medium", zone_alias, speaker_alias, dialog_prompt)
                            
                motion_alias = (
                    f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_MOTION_01"
                )

                motion_prompt = T.dialog_motion_prompt(dlg['speaker'], facial, head)
                T.dialog_motion(motion_alias, base_alias, motion_prompt, duration=2)

            # Now generate motion + final video PER SENTENCE
            final_alias = (
                f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_VIDEO_{s_idx:02d}"
            )

            T.dialog_final(final_alias, base_alias, f"{speaker_alias}_VOICE", line)
            s_idx += 1

'''

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    basepath = sys.argv[1]

    with open(f"{basepath}/output/registry.json") as ass:
        assets = json.load(ass)
    with open(f"{basepath}/output/rewrite.json") as act:
        actions = json.load(act)
    
    actions = split_dialog_sentences(actions)

    with open(f"{basepath}/output/complete_segmented.json", 'w') as act:
        json.dump(actions,act,indent=4)

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
