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
'''
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
'''


import os

def get_backgrounds(assets, mappings, T, output_dir="backdrops_tmp"):
    os.makedirs(output_dir, exist_ok=True)
    
    for location in assets['locations']:
        location_name = location['name']
        architecture = location['architectural_shell']
        
        for zone in location['zones']:
            zone_name = zone['zone_name']
            zone_def = zone['zone_definition']
            elements = zone.get('visible_background_elements', [])
            
            # Create canonical zone key
            zone_key = f"{location_name}_{zone_name}".replace(' ', '_').replace('/','_').upper()
            
            # 1. Generate the WIDE SHOT (master reference)
            T.background(
                zone_key,
                architecture,
                zone_def,
                ', '.join(elements)
            )
            
            # 2. Generate MIDDLE variant for Two-Shot (shows both characters)
            #T.backdrop(zone_key, zone_key, "middle")
            
            # 3. Generate LEFT variant (for biographies[0])
            T.backdrop(zone_key, zone_key, "left")
            
            # 4. Generate RIGHT variant (for biographies[1])
            T.backdrop(zone_key, zone_key, "right")
            
            # 5. Map zone name to all three variants
            mappings[zone_name] = {
                'LEFT': f"{zone_key}_LEFT_BACKDROP",
                'MIDDLE': f"{zone_key}_BACKGROUND",
                'RIGHT': f"{zone_key}_RIGHT_BACKDROP"
            }


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
    char_index_map = {canonical(bio['name']): i for i, bio in enumerate(assets['biographies'])}
    
    pose_cache = {}
    video_cache = {}

    for beat in actions:
        if not beat['action']:
            continue
        all_names = build_identity_map(assets, beat)
        beat_chars = get_beat_characters(beat, all_names)
        if not beat_chars:
            continue

        names = {c: all_names[c] for c in beat_chars}
        
        zone_name = beat['zone']
        zone_mappings = mappings.get(zone_name, {})
        
        if len(beat_chars) == 1:
            char_name = list(beat_chars)[0]
            char_idx = char_index_map.get(canonical(char_name), 0)
            shot_variant = 'LEFT' if char_idx == 0 else 'RIGHT'
        else:
            shot_variant = 'MIDDLE'
        
        zone_alias = zone_mappings.get(shot_variant, 'UNKNOWN')

        resolved_actions = [resolve_pronouns(a, names) for a in beat['actions']]
        pose_action_map, motion_actions = bind_identity_first_only(resolved_actions, names)

        arc = beat.get('arc', "")
        pose = beat.get('pose', "")

        # 🆕 More robust character ordering with case-insensitive matching
        ordered_chars = []
        for char in beat_chars:
            display_name = names[char]
            canonical_name = char
            
            # Try multiple name variations for matching
            search_names = [display_name, canonical_name, display_name.lower(), canonical_name.lower()]
            
            # Find earliest position in arc or pose
            first_pos = 999999
            for search_name in search_names:
                if arc:
                    pos = arc.lower().find(search_name.lower())
                    if pos != -1 and pos < first_pos:
                        first_pos = pos
                if pose:
                    pos = pose.lower().find(search_name.lower())
                    if pos != -1 and pos < first_pos:
                        first_pos = pos
            
            ordered_chars.append((first_pos, char))
        
        # Sort by position (characters not found go to end in original order)
        ordered_chars.sort(key=lambda x: x[0])
        ordered_char_list = [c[1] for c in ordered_chars]
        
        # Debug output
        #print(f"Beat {beat['beat']}: ordered chars = {ordered_char_list}")
        
        # Build cache key using ordered characters
        pose_key_parts = []
        for char in ordered_char_list:
            posture = beat['posture'].get(char, 'neutral')
            expression = beat.get('facial', 'neutral')
            pose_key_parts.append(f"{char}_{posture}_{expression}")
        
        if len(beat_chars) > 1:
            pose_key_parts.append("two_shot")
        
        pose_key = "_".join(pose_key_parts)
        
        # Build char_assets in the same order as they appear in prompt
        char_assets = " and ".join(f"{char_aliases[c]} asset " for c in ordered_char_list)
        
        # Debug output
        #print(f"  char_assets = {char_assets}")
        
        # Check cache before generating static image
        if pose_key not in pose_cache:
            alias = f"BEAT_{beat['beat']}_WIDE_ACTION"
            if len(beat_chars) == 1:
                alias = f"BEAT_{beat['beat']}_MEDIUM_ACTION"
                T.action_medium(alias, zone_alias, char_assets, resolve_character_mentions(pose, names))
            else:
                T.action_wide(alias, zone_alias, char_assets, resolve_character_mentions(arc, names))
            pose_cache[pose_key] = alias
        else:
            alias = pose_cache[pose_key]
        
        duration = 10 if os.environ.get('WGP','False') == 'True' or os.environ.get('LTX','False') == 'True' else 5
        
        if arc and arc.strip():
            motion_key = f"{pose_key}_{arc.strip()}"
            
            if motion_key not in video_cache:
                existing_count = len([k for k in video_cache if k.startswith(pose_key)])
                video_alias = f"{alias}_VIDEO_{existing_count:02d}"
                ref_alias = alias
                vid_duration = duration
                if existing_count > 0:
                    ref_count = existing_count - 1
                    ref_alias = video_alias = f"{alias}_VIDEO_{ref_count:02d}"
                    vid_duration = 5

                
                T.action_video(
                    video_alias, 
                    ref_alias, 
                    resolve_character_mentions(arc, names), 
                    duration=vid_duration
                )
                video_cache[motion_key] = video_alias

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
    
    char_index_map = {canonical(bio['name']): i for i, bio in enumerate(assets['biographies'])}

    # Cache dialog base images by visual content, not beat number
    dialog_base_cache = {}

    for beat in actions:
        dialog_list = [
            d for d in (beat.get('dialog') or [])
            if d.get("line") and d.get("line").strip().lower() not in ("", "none")
        ]
        if not dialog_list:
            continue

        s_idx = 1

        for dlg in dialog_list:
            speaker = canonical(dlg['speaker'])
            if speaker not in char_aliases:
                continue

            zone_name = beat['zone']
            zone_mappings = mappings.get(zone_name, {})
            
            char_idx = char_index_map.get(speaker, 0)
            shot_variant = 'LEFT' if char_idx == 0 else 'RIGHT'
            zone_alias = zone_mappings.get(shot_variant, 'UNKNOWN')

            speaker_alias = char_aliases[speaker]
            raw_line = dlg['line']
            line = clean_dialog_line(raw_line)
            raw_speaker = dlg['speaker']

            facial_state_map = beat.get('facial_state') or {}
            head_gesture_map = beat.get('head_gesture') or {}
            tone_map = beat.get('tone') or {}

            facial = facial_state_map.get(raw_speaker, 'neutral')
            head = head_gesture_map.get(raw_speaker, 'none')
            tone = normalize_tone(tone_map.get(raw_speaker, 'neutral'))

            facial = beat.get('facial', 'crazy')
            if not facial:
                facial = 'neutral'

            start_desc = beat.get('starting_description', {})
            posture = start_desc.get(raw_speaker, None)
            if posture:
                pose_sentence = f"{dlg['speaker']} is {posture}."
            else:
                pose_sentence = ""

            expr_sentence = f"{dlg['speaker']} has a {facial} expression." if facial != "neutral" else ""
            dialog_prompt = " ".join(s for s in [pose_sentence, expr_sentence] if s)

            # Cache key based on visual content
            dialog_key = f"{speaker}_{facial}_{zone_alias}"

            if dialog_key not in dialog_base_cache:
                # Generate new base image and cache it
                base_alias = f"DIALOG_BASE_{normalize(speaker)}_{facial}_{shot_variant}"
                
                dialog_pose_prompt_close = (
                    f"{dlg['speaker']} (facial expression {facial})"
                )

                T.dialog_closeup(base_alias, zone_alias, speaker_alias, dialog_pose_prompt_close)
                dialog_base_cache[dialog_key] = base_alias
                #print(f"🎭 Generated dialog base: {base_alias}")
            else:
                # Reuse cached base image
                base_alias = dialog_base_cache[dialog_key]
                #print(f"♻️ Reusing cached dialog base: {base_alias} for beat {beat['beat']}")

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
