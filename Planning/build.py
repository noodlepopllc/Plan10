#!/usr/bin/env python3
import json
import sys
import re

def slugify(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text.strip('_') or "asset"

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
    # PHASE 2: COMPOSITES (compd + compv)
    # ========================================================================
    compd_refs = []
    compv_refs = []
    char_seq = {cid: {"compd": 0, "compv": 0} for cid in char_map}

    for beat in sequence["beats"]:
        visible_ids = beat.get("visible_chars") or [beat.get("char_id")]
        valid_visible = [cid for cid in visible_ids if cid in char_map and cid is not None][:2]
        if not valid_visible: continue

        visible_aliases = [f"char_{char_map[cid]['slug']}" for cid in valid_visible]
        combining_str = f"bg_{env_slug}, " + ", ".join(visible_aliases)

        focus_cid = valid_visible[0]
        ch = char_map[focus_cid]
        slug = ch["slug"]

        if beat["type"] == "dialog":
            char_seq[focus_cid]["compd"] += 1
            idx = char_seq[focus_cid]["compd"]
            alias = f"compd_{slug}_{idx:02d}"
            face = beat.get("facial_action") or "neutral expression"
            compd_refs.append({"alias": alias, "design": ch["design"], "text": beat["text"], "face": face})
            out.append(f'\n>> ALIAS: {alias}')
            out.append(f'composite_scene combining={combining_str}, shot_type="closeup", action="{face}, cropped at shoulders, NO hands, NO props, static pose" Height: 832, Width: 480, Seed: -1')

        elif beat["type"] == "action":
            shot = beat.get("shot_type") or "medium"
            if shot == "closeup": shot = "medium"

            # 🎯 FRAME 0: Compositor sets the exact starting position
            starting_pose = beat.get("starting_pose") or "standing relaxed, weight centered, hands at sides"
            if "no mouth movement" not in starting_pose.lower() and "no speech" not in starting_pose.lower():
                starting_pose += ", NO mouth movement, NO speech animation"

            # 🎯 FRAMES 1→N: Video model handles temporal motion
            motion = beat.get("motion_prompt") or starting_pose
            if "subtle camera drift" not in motion.lower():
                motion += ", subtle camera drift"
            if "no mouth movement" not in motion.lower() and "no speech" not in motion.lower():
                motion += ", NO mouth movement, NO speech animation"

            # NO CACHE: Every action beat gets a fresh composite for its exact starting pose
            char_seq[focus_cid]["compv"] += 1
            idx = char_seq[focus_cid]["compv"]
            alias = f"compv_{slug}_{idx:02d}"
            
            out.append(f'\n>> ALIAS: {alias}')
            out.append(f'composite_scene combining={combining_str}, shot_type="{shot}", action="{starting_pose}" Height: 832, Width: 480, Seed: -1')
            
            compv_refs.append({"alias": alias, "motion": motion})

    # ========================================================================
    # PHASE 3: VOICES (design)
    # ========================================================================
    for c in registry["characters"]:
        slug = slugify(c.get("alias_slug", c["name"]))
        out.append(f'\n>> ALIAS: design_{slug}')
        out.append(f'design_voice voice="{c["voice"]}"')

    # ========================================================================
    # PHASE 4: DIALOG (dialog)
    # ========================================================================
    dialog_idx = 1
    for ref in compd_refs:
        out.append(f'\n>> ALIAS: dialog_{dialog_idx:03d}')
        out.append(f'dialog_to_video using={ref["alias"]}, audio={ref["design"]}, text="{ref["text"]}", prompt="{ref["face"]}, lips moving naturally, NO head turns, NO hand gestures, NO prop interaction" Height: 832, Width: 480, Seed: -1')
        dialog_idx += 1

    # ========================================================================
    # PHASE 5: VIDEO (action clips) — 3.0s duration, motion prompt only
    # ========================================================================
    video_idx = 1
    for ref in compv_refs:
        out.append(f'\n>> ALIAS: video_{video_idx:03d}')
        out.append(f'image_to_video using={ref["alias"]}, prompt="{ref["motion"]}", duration_sec=3.0 Height: 832, Width: 480, Seed: -1')
        video_idx += 1

    return "\n".join(out)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python renderer.py registry.json sequence.json", file=sys.stderr)
        sys.exit(1)
    print(render_pipeline(sys.argv[1], sys.argv[2]))