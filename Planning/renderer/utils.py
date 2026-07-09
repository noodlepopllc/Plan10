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
            lambda m: f"{m.group(0)} ({ident}) ",
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

# utils.py

def create_backdrop_mapping(assets, actions):
    """Create mapping from beat backdrop names to generated asset aliases."""
    mappings = {}
    
    for location in assets['locations']:
        location_name = location['name']
        
        for zone in location['zones']:
            zone_name = zone['zone_name']
            char_positions = zone.get('character_positions', [])
            
            # Two-shot mapping
            two_shot_key = f"{zone_name} - Two-Shot"
            zone_key = f"{location_name}_{zone_name}".replace(' ', '_').upper()
            mappings[two_shot_key] = f"{zone_key}_WIDE_BACKDROP"
            
            # Character-specific mappings
            for cp in char_positions:
                char_name = cp['character']
                position = cp.get('position', '').lower()
                
                # Same unique key format as get_backgrounds
                char_key = f"{location_name}__{zone_name}__{char_name}".replace(' ', '_').upper()
                
                if 'left' in position:
                    shot_type = "LEFT"
                elif 'right' in position:
                    shot_type = "RIGHT"
                else:
                    shot_type = "WIDE"
                
                backdrop_key = f"{zone_name} - {char_name}"
                mappings[backdrop_key] = f"{char_key}_{shot_type}_BACKDROP"
    
    return mappings


def resolve_zone_alias(backdrop_name, mappings):
    """Resolve zone name to backdrop alias based on character position."""
    # backdrop_name is now just the zone name like "Beach Shoreline"
    
    if backdrop_name not in mappings:
        return "UNKNOWN"
    
    zone_mappings = mappings[backdrop_name]
    
    # This function needs access to the current beat to determine which character
    # But it's called from render_beats_actions/dialog without beat context
    # So we need to pass the beat or character info
    
    # For now, return MIDDLE as default (two-shot)
    return zone_mappings.get('MIDDLE', 'UNKNOWN')


def resolve_zone_alias_for_beat(beat, mappings):
    """Resolve zone to backdrop alias based on beat's actor/speaker."""
    zone_name = beat.get('zone', '')
    
    if zone_name not in mappings:
        return "UNKNOWN"
    
    zone_mappings = mappings[zone_name]
    
    actor = beat.get('actor', '')
    speaker = beat.get('speaker', '')
    
    # Both characters present → MIDDLE (two-shot)
    if actor and speaker and actor != speaker:
        return zone_mappings.get('MIDDLE', 'UNKNOWN')
    
    # Solo shot - need to determine left/right
    # This requires knowing character order from biographies
    # For now, default to MIDDLE
    # TODO: Pass biographies or character index to determine LEFT/RIGHT
    
    return zone_mappings.get('MIDDLE', 'UNKNOWN')

    


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
    combined = []
    buffer = ""
    buffer_syllables = 0

    for sent in sentences:
        sent = sent.strip()
        sent_syllables = count_syllables(sent)

        # If adding this sentence stays under threshold → merge
        if buffer and (buffer_syllables + sent_syllables) < threshold:
            buffer = buffer + " " + sent
            buffer_syllables += sent_syllables
        else:
            # Flush old buffer
            if buffer:
                combined.append(buffer)

            # Start new buffer
            buffer = sent
            buffer_syllables = sent_syllables

    # Flush last buffer
    if buffer:
        combined.append(buffer)

    return combined

import re

def filter_empty_beats(beats):
    """Remove beats with no action and no dialog."""
    return [beat for beat in beats if beat.get('action') or beat.get('dialog')]

def clean_action_narrative(beats):
    """Strip narrative, internal states, and causation from action fields."""
    # Patterns to remove
    narrative_patterns = [
        r'\b(in relief|with lingering tension|mixed with|causing|because|due to|as a result)\b.*$',
        r'\b(until he decides|voluntarily|rather than being forced|by external circumstances)\b.*$',
        r'\b(while maintaining|as she maneuvers|causing them to)\b.*$',
        r'\b(typical of someone|despite lacking|without breaking)\b.*$',
    ]
    
    for beat in beats:
        action = beat.get('action', '')
        if action:
            for pattern in narrative_patterns:
                action = re.sub(pattern, '', action, flags=re.IGNORECASE)
            # Clean up trailing whitespace and punctuation
            action = action.strip().rstrip(',').strip()
            beat['action'] = action
    
    return beats

def clean_dialog_narrative(beats):
    """Strip narrative tags from dialog fields."""
    # Patterns like "he said", "she warns", "through gritted teeth"
    narrative_tags = [
        r',?\s*\w+\s+(warns|says|asks|replies|shouts|whispers|mutter|calls)\b[^"]*',
        r',?\s*(through|with)\s+[^"]*?(teeth|voice|tone|expression)\b[^"]*',
    ]
    
    for beat in beats:
        dialog = beat.get('dialog', '')
        if dialog:
            for pattern in narrative_tags:
                dialog = re.sub(pattern, '', dialog, flags=re.IGNORECASE)
            # Clean up extra spaces and punctuation
            dialog = re.sub(r'\s+', ' ', dialog).strip()
            dialog = re.sub(r'""', '"', dialog)  # Remove double quotes
            beat['dialog'] = dialog
    
    return beats

def fix_posture_zone_contradictions(beats):
    """Fix posture when zone implies a different state."""
    for beat in beats:
        zone = beat.get('zone', '').lower()
        posture = beat.get('posture', '')
        
        # If zone implies seated but posture says standing
        if ('seated' in zone or 'dining' in zone or 'table' in zone):
            if posture == 'standing':
                action = beat.get('action', '').lower()
                # Check if action implies standing up
                if 'stand' in action or 'rise' in action or 'get up' in action:
                    beat['posture'] = 'standing'
                elif 'sit' in action:
                    beat['posture'] = 'seated'
                else:
                    # Default to seated if in dining zone and no standing action
                    beat['posture'] = 'seated'
    
    return beats

def deduplicate_dialog(beats):
    """Remove consecutive duplicate dialog lines."""
    deduped = []
    for i, beat in enumerate(beats):
        if i > 0:
            prev = beats[i-1]
            # Skip if exact duplicate of previous dialog
            if (beat.get('dialog') == prev.get('dialog') and 
                beat.get('dialog') and
                beat.get('actor') == prev.get('actor')):
                continue
        deduped.append(beat)
    return deduped

def normalize_speaker_names(beats, biography):
    """Normalize speaker names to match biography canonical names."""
    # Build name mapping
    name_map = {}
    for char in biography.get('biographies', []):
        full_name = char['name']
        name_map[full_name.lower()] = full_name
        # Map first name to full name
        first_name = full_name.split()[0].lower()
        name_map[first_name] = full_name
    
    for beat in beats:
        # Normalize actor
        actor = beat.get('actor', '')
        if actor and actor.lower() in name_map:
            beat['actor'] = name_map[actor.lower()]
        
        # Normalize speaker
        speaker = beat.get('speaker', '')
        if speaker and speaker.lower() in name_map:
            beat['speaker'] = name_map[speaker.lower()]
    
    return beats

def postprocess_beats(beats, biography):
    """Apply all post-processing steps in order."""
    beats = filter_empty_beats(beats)
    #beats = clean_action_narrative(beats)
    beats = clean_dialog_narrative(beats)
    #beats = fix_posture_zone_contradictions(beats)
    beats = deduplicate_dialog(beats)
    beats = normalize_speaker_names(beats, biography)
    return beats


def postprocess_beats(beats, biography):
    """Apply all post-processing steps in order."""
    beats = filter_empty_beats(beats)
    beats = deduplicate_dialog(beats)
    beats = normalize_speaker_names(beats, biography)
    return beats


def rebalance_chunks(chunks, min_words=2, min_syllables=5):
    """Rebalance chunks if the last one is too short."""
    if len(chunks) < 2:
        return chunks
    
    last_chunk = chunks[-1]
    last_words = last_chunk.split()
    last_syllables = count_syllables(last_chunk)
    
    # Check if last chunk is too short
    if len(last_words) < min_words or last_syllables < min_syllables:
        # Get previous chunk
        prev_chunk = chunks[-2]
        prev_words = prev_chunk.split()
        
        # Remove ellipsis from previous chunk if present
        if prev_words[-1].endswith('...'):
            prev_words[-1] = prev_words[-1][:-3]
        
        # Move words from previous to last chunk until balanced
        # Strategy: move words until last chunk meets minimum, or prev chunk gets too short
        while (len(last_words) < min_words or count_syllables(' '.join(last_words)) < min_syllables):
            if len(prev_words) <= min_words:
                break  # Don't make previous chunk too short
            
            # Move last word from prev to front of last
            word_to_move = prev_words.pop()
            last_words.insert(0, word_to_move)
        
        # Rebuild chunks
        chunks[-2] = ' '.join(prev_words) + '...'
        chunks[-1] = ' '.join(last_words)
    
    return chunks


def chunk_long_sentence(sentence, syllable_threshold=22):
    """Chunk a long sentence by syllable count, adding ellipsis as breath pauses."""
    if count_syllables(sentence) <= syllable_threshold:
        return [sentence]
    
    words = sentence.split()
    chunks = []
    current_chunk = []
    current_syllables = 0
    
    for word in words:
        word_syllables = count_syllables(word)
        
        # If adding this word exceeds threshold, flush current chunk
        if current_chunk and (current_syllables + word_syllables) > syllable_threshold:
            chunks.append(' '.join(current_chunk) + '...')
            current_chunk = [word]
            current_syllables = word_syllables
        else:
            current_chunk.append(word)
            current_syllables += word_syllables
    
    # Flush final chunk (no ellipsis on last one)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Rebalance if last chunk is too short
    chunks = rebalance_chunks(chunks, min_words=2, min_syllables=5)
    
    return chunks


def split_dialog_sentences(beats, biography, syllable_threshold=16):
    beats = postprocess_beats(beats, biography)
    new_beats = []
    for ndx, beat in enumerate(beats):
        if 'backdrop' in beat:
            del beat['backdrop']
        line = beat.get("dialog", '')
        expression = beat.get('facial') or 'neutral'
        beat['facial'] = expression
        beat["speaker"] = beat["speaker"].replace("unknown","")
        beat["actor"] = beat["actor"].replace("unknown","")
        speaker = beat["speaker"]
        if not beat['actor']:
            beat['actor'] = speaker
        if not beat['speaker']:
            beat['speaker'] = beat['actor']
        if not speaker:
            continue
        beat['beat'] = ndx
        beat['posture'] = {beat['actor']: f'{beat["posture"]}'}
        beat['actions'] = [beat['action']] if beat.get('action') else []
        beat['dialog'] = []
        beat['pose'] = f"{beat['actor']} {beat['posture'][beat['actor']]} with a {expression} expression"
        beat['arc'] = f"{beat['actor']} {beat['posture'][beat['actor']]} with a {expression} expression, {beat['action']}" if beat.get('action') else ""
        
        if new_beats:
            prev = new_beats[-1]
            same_actor = beat.get('actor') == prev.get('actor')
            same_posture = beat.get('posture') == prev.get('posture')
            same_zone = beat.get('zone') == prev.get('zone')
            same_location = beat.get('location') == prev.get('location')
            
            if same_actor and same_posture and same_zone and same_location:
                if beat.get('action'):
                    prev['actions'].append(beat['action'])
                
                if line:
                    line = line.strip()
                    sentences = segment_sentences(line)
                    
                    if len(sentences) == 1:
                        chunks = chunk_long_sentence(sentences[0], syllable_threshold)
                        for chunk in chunks:
                            prev['dialog'].append({"speaker": speaker, "line": chunk.strip()})
                    else:
                        sentences = recombine_by_syllables(sentences, threshold=syllable_threshold)
                        for sent in sentences:
                            #prev['dialog'].append({"speaker": speaker, "line": sent.strip()})
                            chunks = chunk_long_sentence(sent.strip(), syllable_threshold)
                            for chunk in chunks:
                                prev['dialog'].append({"speaker": speaker, "line": chunk.strip()})
                
                if prev['actions']:
                    prev['arc'] = f"{prev['actor']} {prev['posture'][prev['actor']]} with a {prev['facial']} expression, {', '.join(prev['actions'])}"
                
                continue
        
        if line:
            line = line.strip()
            sentences = segment_sentences(line)
            
            if len(sentences) == 1:
                chunks = chunk_long_sentence(sentences[0], syllable_threshold)
                for chunk in chunks:
                    beat['dialog'].append({"speaker": speaker, "line": chunk.strip()})
            else:
                sentences = recombine_by_syllables(sentences, threshold=syllable_threshold)
                for sent in sentences:
                    beat['dialog'].append({"speaker": speaker, "line": sent.strip()})
        
        new_beats.append(beat)
    
    return new_beats
