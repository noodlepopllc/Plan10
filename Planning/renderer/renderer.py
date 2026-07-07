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

sys.path.append('./lib')
from config import load_environ

load_environ()

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

            elements = [] #elements[:3] if len(elements) > 3 else elements
            
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
    video_counters = {}  # Track video count per beat
    
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
        
        # Build pose cache key
        pose_key_parts = []
        for char in sorted(beat_chars):
            posture = beat['posture'].get(char, 'neutral')
            expression = beat.get('facial', 'neutral')
            pose_key_parts.append(f"{char}_{posture}_{expression}")
        
        if len(beat_chars) > 1:
            pose_key_parts.append("two_shot")
        
        pose_key = "_".join(pose_key_parts)
        
        char_assets = " and ".join(f"{char_aliases[c]} asset " for c in sorted(beat_chars))
        
        # Generate static image if not cached
        if pose_key not in pose_cache:
            alias = f"BEAT_{beat['beat']}_ACTION"
            if len(beat_chars) == 1:
                T.action_medium(alias, zone_alias, char_assets, resolve_character_mentions(beat.get('pose', ''), names))
            else:
                T.action_wide(alias, zone_alias, char_assets, resolve_character_mentions(beat.get('arc', ''), names))
            pose_cache[pose_key] = alias
        else:
            alias = pose_cache[pose_key]
        
        # Extract action from arc
        arc = beat.get('arc', '')
        if arc and arc.strip():
            # Initialize counter for this beat if needed
            if beat['beat'] not in video_counters:
                video_counters[beat['beat']] = 0
            
            counter = video_counters[beat['beat']]
            video_alias = f"BEAT_{beat['beat']}_ACTION_{counter:02d}"
            video_counters[beat['beat']] += 1
            
            duration = 10 if os.environ.get('WGP','False') == 'True' or os.environ.get('LTX','False') != 'False' else 5
            
            T.action_video(
                video_alias,
                alias,
                resolve_character_mentions(arc, names),
                duration=duration
            )


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

    # 🆕 Build zone index map
    zone_index_map = {}
    zone_counter = 0
    for beat in actions:
        zone_name = beat['zone']
        if zone_name and zone_name not in zone_index_map:
            zone_index_map[zone_name] = zone_counter
            zone_counter += 1

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
            zone_idx = zone_index_map.get(zone_name, 0)
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

            # --- Cache keys: separate for closeup and OTS ---
            dialog_key_closeup = f"{speaker}_{normalize(facial)}_{zone_alias}_CLOSEUP"
            dialog_key_medium = f"{speaker}_{normalize(facial)}_{zone_alias}_MEDIUM"
            dialog_key_ots = f"{speaker}_{normalize(facial)}_{zone_alias}_OTS"

            # 🎭 Closeup base (unchanged behavior)
            if dialog_key_closeup not in dialog_base_cache:
                base_alias_closeup = f"DIALOG_BASE_{normalize(speaker)}_{normalize(facial)}_{shot_variant}_Z{zone_idx}_CLOSEUP"
                base_alias_medium = f"DIALOG_BASE_{normalize(speaker)}_{normalize(facial)}_{shot_variant}_Z{zone_idx}_MEDIUM"
                
                dialog_pose_prompt_close = (
                    f"{dlg['speaker']} (facial expression {facial})"
                )
                T.dialog_closeup(base_alias_closeup, zone_alias, speaker_alias, dialog_pose_prompt_close)
                T.dialog_medium(base_alias_medium, zone_alias, speaker_alias, dialog_pose_prompt_close)
                dialog_base_cache[dialog_key_closeup] = base_alias_closeup
                dialog_base_cache[dialog_key_medium] = base_alias_medium
            else:
                base_alias_closeup = dialog_base_cache[dialog_key_closeup]
                base_alias_medium = dialog_base_cache[dialog_key_medium] 

            # 📸 Over-the-shoulder base (image only)
            if dialog_key_ots not in dialog_base_cache:
                base_alias_ots = f"DIALOG_BASE_{normalize(speaker)}_{facial}_{shot_variant}_Z{zone_idx}_OTS"
                
                # Identify the non-speaking character (the one whose shoulder we look over)
                other_char_idx = 1 - char_idx  # Flips 0 to 1, or 1 to 0
                other_speaker = next((c for c, i in char_index_map.items() if i == other_char_idx), None)
                other_alias = char_aliases.get(other_speaker) if other_speaker else None
                
                # ⚠️ REQUIREMENT: The person speaking MUST be the second character.
                # 1st asset = Non-speaker (foreground/shoulder)
                # 2nd asset = Speaker (focused face in background)
                if other_alias:
                    ots_char_assets = f"{other_alias}, {speaker_alias}"
                    dialog_pose_prompt_ots = (
                        f"Over-the-shoulder shot from behind {other_alias}'s shoulder, "
                        f"focusing on {speaker_alias} who has a {facial} expression"
                    )
                else:
                    # Fallback if there's somehow no second character in the zone
                    ots_char_assets = speaker_alias
                    dialog_pose_prompt_ots = (
                        f"Over-the-shoulder shot, focusing on {speaker_alias} with a {facial} expression"
                    )
                
                T.dialog_ots(base_alias_ots, zone_alias, ots_char_assets, dialog_pose_prompt_ots)
                dialog_base_cache[dialog_key_ots] = base_alias_ots
            else:
                base_alias_ots = dialog_base_cache[dialog_key_ots]

            # 🎬 Final dialog video still uses the closeup base
            final_alias = f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_VIDEO_{s_idx:02d}"
            T.dialog_final(final_alias, base_alias_closeup, f"{speaker_alias}_VOICE", line, '')
            final_alias = f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_VIDEO_MEDIUM_{s_idx:02d}"
            loc_alias = base_alias_medium

            all_names = build_identity_map(assets, beat)
            beat_chars = get_beat_characters(beat, all_names)

            if os.environ.get('LTX','False') == 'ARC':
                arc = beat.get('arc', '')
                names = {c: all_names[c] for c in beat_chars}
                final_arc = resolve_character_mentions(arc, names)
            else:
                final_arc = ''
            if len(beat_chars) == 2:
                ots_alias = f"BEAT_{beat['beat']}_{normalize(speaker)}_DIALOG_VIDEO_OTS_{s_idx:02d}"
                T.dialog_final(ots_alias, base_alias_ots, f"{speaker_alias}_VOICE", line, final_arc)

            T.dialog_final(final_alias, loc_alias, f"{speaker_alias}_VOICE", line, final_arc)
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

        threshold = 32 if os.environ.get('WGP','False') == 'True' or os.environ.get('LTX','False') == 'True' else 16
        
        actions = split_dialog_sentences(actions, assets, threshold)

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

    if len(sys.argv) > 2 and sys.argv[2] in ("images", "all", "videos", "identity", "dialog", "full"):
        mode = sys.argv[2]
    else:
        mode = os.environ.get("MODE", "all")

    T.buffer.dump(mode)


if __name__ == "__main__":
    main()
