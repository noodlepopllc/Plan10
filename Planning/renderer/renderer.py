#!/usr/bin/env python3
import os, sys, json, re

# ONE import for all template output
from templates import Templates

# ONE import for all shared helpers
from utils import (
    canonical, normalize, soft_normalize,
    resolve_character_mentions, resolve_pronouns,
    create_backdrop_mapping, resolve_zone_alias,
    get_beat_characters, bind_identity_first_only,
    split_dialog_sentences
)

POSTURE = {}


# ---------------------------------------------------------
# CHARACTER SHEETS + VOICES
# ---------------------------------------------------------

def get_identity(assets, T):
    for bio in assets['biographies']:
        #bio = char['biography']
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

    for bio in assets['biographies']:
        #bio = char['biography']
        ckey = canonical(bio['name'])

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
            for backdrop in zone['backdrops']:
                if backdrop['backdrop_name'] not in mappings:
                    continue
                bd_key = mappings[backdrop['backdrop_name']]
                alias = normalize(bd_key)
                T.background(
                    alias,
                    architecture,
                    backdrop['backdrop_definition'],
                    backdrop['visible_background_elements']
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


def render_beats_actions(assets, actions, mappings, T):
    char_aliases = {canonical(c['name']): f"CHAR_{normalize(c['name'])}" for c in assets['biographies']}

    for beat in actions:
        if not beat['action']:
            continue
        all_names = build_identity_map(assets, beat)
        beat_chars = get_beat_characters(beat, all_names)
        if not beat_chars:
            continue

        names = {c: all_names[c] for c in beat_chars}
        zone_alias = resolve_zone_alias(beat['backdrop'], mappings)

        # --- NEW: starting description integration ---
        #start_desc = beat.get('starting_description', {})
        #start_sentences = [f"{char} is {desc}." for char, desc in start_desc.items()]

        # --- existing action resolution ---
        resolved_actions = [resolve_pronouns(a, names) for a in beat['actions']]
        pose_action_map, motion_actions = bind_identity_first_only(resolved_actions, names)

        # --- NEW: arc integration ---
        arc = beat.get('arc', "")
        arc_sentence = f"Motion arc: {arc}" if arc else ""

        # --- build prompt ---
        #sentences = start_sentences
        sentences = [beat['arc']]
        for char, action in pose_action_map.items():
            clean_action = strip_leading_name(action, char)
            sentences.append(f"{char} {clean_action}.")


        wide_prompt = " ".join(sentences[:len(beat_chars)])
        identity = '\n'.join([f'{x.capitalize()} ({y})' for x, y in names.items()])

        alias = f"BEAT_{beat['beat']}_WIDE_ACTION"
        char_assets = " and ".join(f"{char_aliases[c]} asset " for c in beat_chars)
        if len(beat_chars) == 1:
            ide_prompt = sentences[0]
            alias = f"BEAT_{beat['beat']}_MEDIUM_ACTION"
            T.action_medium(alias, zone_alias, char_assets, resolve_character_mentions(arc, names))
        else:
            T.action_wide(alias, zone_alias, char_assets, resolve_character_mentions(arc, names))
        sentences = sentences[:len(beat_chars)]
        if arc_sentence:
            sentences.append(arc_sentence)
        wide_prompt = " ".join(sentences)
        duration = 10 if os.environ.get('WGP','False') == 'True' else 5
        T.action_video(f"{alias}_VIDEO", alias, resolve_character_mentions(arc_sentence, names), duration=duration)

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
        for c in assets['biographies']
    }

    generated_bases = set()

    for beat in actions:
        dialog_list = [
            d for d in (beat.get('dialog') or [])
            if d.get("line") and d.get("line").strip().lower() not in ("", "none")
        ]
        if not dialog_list:
            continue

        zone_alias = resolve_zone_alias(beat['backdrop'], mappings)
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

            facial = beat.get('facial','crazy')
            if not facial:
                facial = 'neutral'

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
                    f"{dlg['speaker']} (facial expression {facial})"
                )

                # ONE closeup + ONE medium per beat per speaker
                T.dialog_closeup(base_alias, zone_alias, speaker_alias, dialog_pose_prompt_close)

            final_alias = f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_VIDEO_{s_idx:02d}"
            T.dialog_final(final_alias, base_alias, f"{speaker_alias}_VOICE", line)
            s_idx += 1


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    from pathlib import Path
    import sys
    
    basepath = sys.argv[1]

    with open(f"{basepath}/output/registry.json") as ass:
        assets = json.load(ass)
    actions = []
    if not Path(f"{basepath}/output/complete_segmented.json").exists():
        with open(f"{basepath}/output/narrative.json") as act:
            for line in act:
                actions.append(json.loads(line))
        
        actions = split_dialog_sentences(actions, assets)

        with open(f"{basepath}/output/complete_segmented.json", 'w') as act:
            json.dump(actions,act,indent=4)
    else:
        actions = json.loads(Path(f"{basepath}/output/complete_segmented.json").read_text())

    T = Templates()   # ← ONE OBJECT

    get_identity(assets, T)
    mappings = create_backdrop_mapping(assets, actions)
    get_backgrounds(assets, mappings, T)

    render_beats_actions(assets, actions, mappings, T)
    render_beats_dialog(assets, actions, mappings, T)

    if len(sys.argv) > 2 and sys.argv[2] in ("images", "all", "videos", "identity", "dialog"):
        mode = sys.argv[2]
    else:
        mode = os.environ.get("MODE", "all")

    T.buffer.dump(mode)


if __name__ == "__main__":
    main()
