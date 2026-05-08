#!/usr/bin/env python3
import json
import sys
import re

def slugify(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text.strip('_') or "asset"

def _get_required_moods(sequence: dict, char_map: dict) -> set:
    """Scan sequence for exact (slug, mood) pairs needed."""
    ALLOWED_MOODS = {"neutral", "confident", "skeptical", "encouraging", "curious",
                     "supportive", "shy", "reassuring", "amused", "eager",
                     "nervous", "patient", "relieved", "stern", "defensive",
                     "frustrated", "overwhelmed", "playful"}
    required = set()
    for beat in sequence.get("beats", []):
        if beat.get("type") == "dialog":
            vis = beat.get("visible_chars", [])
            if vis and vis[0] in char_map:
                slug = char_map[vis[0]]["slug"]
                fa = beat.get("facial_action", "neutral")
                mood = fa.split(",")[0].strip().lower()
                if mood in ALLOWED_MOODS:
                    required.add((slug, mood))
    return required

def render_pipeline(registry_path: str, sequence_path: str) -> str:
    with open(registry_path) as f: registry = json.load(f)
    with open(sequence_path) as f: sequence = json.load(f)

    out = []
    env_slug = slugify(registry.get("environment_alias", "environment"))

    # ========================================================================
    # PHASE 1: ASSETS (bg + char)
    # ========================================================================
    out.append(f'>> ALIAS: bg_{env_slug}')
    out.append(f'create_background prompt="{registry["environment"]}" Height: 832, Width: 480, Seed: -1')

    char_map = {}
    for c in registry["characters"]:
        slug = slugify(c.get("alias_slug", c["name"]))
        char_map[c["id"]] = {"slug": slug, "design": f"design_{slug}", "voice": c["voice"]}
        out.append(f'\n>> ALIAS: char_{slug}')
        out.append(f'create_character_sheet prompt="{c["appearance_prompt"]}" Height: 832, Width: 480, Seed: -1')

    # ========================================================================
    # PHASE 2: REQUIRED CLOSEUPS (generate ONLY needed moods, predictable aliases)
    # ========================================================================
    required = _get_required_moods(sequence, char_map)
    for slug, mood in sorted(required):
        alias = f"compd_{slug}_{mood}"
        action = f"{mood}, mouth completely closed and still, lips sealed shut, zero lip motion, static facial expression, cropped at shoulders, NO hands, NO props"
        out.append(f'\n>> ALIAS: {alias}')
        out.append(f'composite_scene combining=bg_{env_slug}, char_{slug}, shot_type="closeup", action="{action}" Height: 832, Width: 480, Seed: -1')

    # ========================================================================
    # PHASE 3: VOICES (design)
    # ========================================================================
    for c in registry["characters"]:
        slug = slugify(c.get("alias_slug", c["name"]))
        out.append(f'\n>> ALIAS: design_{slug}')
        out.append(f'design_voice voice="{c["voice"]}"')

    # ========================================================================
    # PHASE 4: DIALOG (3-Pass: Pregen Base → I2V Motion → S2V Lip-Sync)
    # ========================================================================
    dialog_idx = 1
    for beat in sequence["beats"]:
        if beat.get("type") != "dialog":
            continue
            
        visible_ids = beat.get("visible_chars", [])
        if not visible_ids or visible_ids[0] not in char_map:
            continue
            
        focus_cid = visible_ids[0]
        ch = char_map[focus_cid]
        slug = ch["slug"]
        
        raw_text = beat.get("text") or ""
        full_action = beat.get("facial_action", "neutral expression")
        
        # 🧹 Parse mood (first word) and motion (everything else)
        parts = full_action.split(",", 1)
        mood = parts[0].strip().lower()
        motion = parts[1].strip() if len(parts) > 1 else "subtle breathing"
        
        # 🧹 Sanitize: move leaked [brackets] from text -> motion
        if raw_text:
            leaked = re.findall(r'\[(.*?)\]', raw_text)
            if leaked:
                raw_text = re.sub(r'\s*\[.*?\]\s*', ' ', raw_text).strip()
                motion = ", ".join(leaked) + ", " + motion

        # PASS 1: Base Headshot (uses predictable alias: compd_{slug}_{mood})
        base_alias = f"compd_{slug}_{mood}"
        
        # PASS 2: Motion Pass (I2V animates posture/gaze, mouth locked)
        i2v_alias = f"vid_motion_{dialog_idx:03d}"
        i2v_prompt = f"{mood}, {motion}, subtle camera drift, mouth completely closed and still, lips sealed shut, zero lip motion"
        out.append(f'\n>> ALIAS: {i2v_alias}')
        out.append(f'image_to_video using={base_alias}, prompt="{i2v_prompt}", duration_sec=2.0 Height: 832, Width: 480, Seed: -1')

        # PASS 3: Lip-Sync Pass (S2V adds speech, preserves motion)
        if raw_text.strip():
            out.append(f'\n>> ALIAS: dialog_{dialog_idx:03d}')
            out.append(f'dialog_to_video using={i2v_alias}, audio={ch["design"]}, text="{raw_text}", prompt="lips moving naturally, preserve existing head motion, NO extra gestures, NO facial drift" Height: 832, Width: 480, Seed: -1')
        else:
            out.append(f'// Pure reaction: {i2v_alias} is final output')

        dialog_idx += 1

    return "\n".join(out)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python renderer.py registry.json sequence.json", file=sys.stderr)
        sys.exit(1)
    print(render_pipeline(sys.argv[1], sys.argv[2]))