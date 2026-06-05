# utils.py
import re
import unicodedata


# ---------------------------------------------------------
# CANONICAL CHARACTER NAME NORMALIZATION
# ---------------------------------------------------------

def canonical(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if ch.isascii() and ch.isalnum())
    return name.upper()


# ---------------------------------------------------------
# BASIC NORMALIZATION HELPERS
# ---------------------------------------------------------

def normalize(name: str) -> str:
    name = name.replace(' ', '_').replace('/', '_')
    return ''.join([x for x in name.upper() if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'])

def soft_normalize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).lower()

def make_fuzzy_pattern(name: str) -> str:
    return rf"\b{re.escape(name)}\b(?:['’]s)?"


# ---------------------------------------------------------
# CHARACTER MENTION RESOLUTION
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

    if g1 == "female":
        pronoun_map["her"] = c1
        pronoun_map["hers"] = c1
    if g2 == "female":
        pronoun_map["her"] = c2
        pronoun_map["hers"] = c2

    if g1 == "male":
        pronoun_map["him"] = c1
    if g2 == "male":
        pronoun_map["him"] = c2

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

    for c in beat.get("posture", {}).keys():
        c_norm = canonical(c)
        if c_norm in all_names:
            chars.add(c_norm)

    for a in beat.get("actions", []):
        ca = soft_normalize(a)
        for c in all_names:
            if soft_normalize(c) in ca:
                chars.add(c)

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


import sys
sys.path.append('./lib')
from util import segment_sentences

import re

def count_syllables(text):
    """
    Fast syllable estimator based on vowel groups.
    Good enough for dialog segmentation decisions.
    """
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z']", " ", text)

    # Special case: empty or weird input
    if not text.strip():
        return 1

    # Count vowel groups as syllables
    groups = re.findall(r"[aeiouy]+", text)

    # Ensure at least 1 syllable
    return max(1, len(groups))

def recombine_by_syllables(sentences, threshold=16):
    """
    Merge consecutive sentences until each combined segment
    exceeds the syllable threshold.
    """
    combined = []
    buffer = ""

    for sent in sentences:
        sent = sent.strip()

        # If buffer is empty, start it
        if not buffer:
            buffer = sent
            continue

        # Check syllable count of the NEXT sentence
        if count_syllables(sent) < threshold:
            # Merge into buffer
            buffer = buffer + " " + sent
        else:
            # Finalize buffer, start new one
            combined.append(buffer)
            buffer = sent

    # Add leftover buffer
    if buffer:
        combined.append(buffer)

    return combined



def split_dialog_sentences(beats):
    """Return a new list of beats with dialog lines split into sentences."""
    new_beats = []

    for beat in beats:
        new_dialog = []

        for entry in beat.get("dialog", []):
            speaker = entry["speaker"]
            line = entry["line"]
            if count_syllables(line) < 16:
                new_dialog.append({
                    "speaker": speaker,
                    "line": line
                })
                continue

            # Segment into sentences
            sentences = segment_sentences(line)
            sentences = recombine_by_syllables(sentences)

            # Rebuild dialog entries, one per sentence
            for sent in sentences:
                new_dialog.append({
                    "speaker": speaker,
                    "line": sent
                })

        # Replace dialog with segmented version
        beat["dialog"] = new_dialog
        new_beats.append(beat)

    return new_beats

