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
        shot = beat.get("shot_type") or "medium"
        if (shot == "ots" or shot == "two_shot"):
            visible_ids = [1,2]
    
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
                starting_pose += ", mouth completely closed and still, lips sealed shut, zero lip motion, static facial expression"

            # 🎯 FRAMES 1→N: Video model handles temporal motion
            motion = beat.get("motion_prompt") or starting_pose
            if "subtle camera drift" not in motion.lower():
                motion += ", subtle camera drift"
            if "no mouth movement" not in motion.lower() and "no speech" not in motion.lower():
                motion += ", mouth completely closed and still, lips sealed shut, zero lip motion, static facial expression"

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
    # PHASE 4: DIALOG (3-Pass: Mood Base → I2V Motion → S2V Lip-Sync)
    # ========================================================================
    dialog_idx = 1
    for ref in compd_refs:
        raw_text = ref.get("text") or ""
        slug = ref["alias"].split("_")[1]  # Extract character slug from alias
        full_action = ref.get("face", "neutral expression")
        
        # 🧹 Parse mood (first word) and motion (everything else)
        parts = full_action.split(",", 1)
        mood = parts[0].strip()
        motion = parts[1].strip() if len(parts) > 1 else "subtle breathing"

        # PASS 1: Base Headshot (Static expression)
        out.append(f'\n>> ALIAS: compd_{slug}_{dialog_idx:02d}')
        out.append(f'composite_scene combining=bg_{env_slug}, char_{slug}, shot_type="closeup", action="{mood}, cropped at shoulders, NO hands, NO props, static pose" Height: 832, Width: 480, Seed: -1')

        # PASS 2: Motion Pass (I2V animates posture/gaze/action)
        i2v_prompt = f"{motion}, subtle camera drift, mouth completely closed and still, lips sealed shut, zero lip motion, static facial expression"
        out.append(f'\n>> ALIAS: vid_motion_{dialog_idx:03d}')
        out.append(f'image_to_video using=compd_{slug}_{dialog_idx:02d}, prompt="{i2v_prompt}", duration_sec=2, Height: 832, Width: 480, Seed: -1')

        # PASS 3: Lip-Sync Pass (S2V adds speech, preserves motion)
        if raw_text.strip():
            out.append(f'\n>> ALIAS: dialog_{dialog_idx:03d}')
            out.append(f'dialog_to_video using=vid_motion_{dialog_idx:03d}, audio=design_{slug}, text="{raw_text}", prompt="lips moving naturally, preserve existing head motion, NO extra gestures, NO facial drift" Height: 832, Width: 480, Seed: -1')
        else:
            # Pure reaction: I2V is the final output
            out.append(f'// Pure reaction: vid_motion_{dialog_idx:03d} is final output')

        dialog_idx += 1

    # ========================================================================
    # PHASE 5: VIDEO (action clips) — 3.0s duration, motion prompt only
    # ========================================================================
    video_idx = 1
    for ref in compv_refs:
        out.append(f'\n>> ALIAS: video_{video_idx:03d}')
        out.append(f'image_to_video using={ref["alias"]}, prompt="{ref["motion"]}", duration_sec=5 Height: 832, Width: 480, Seed: -1')
        video_idx += 1

    return "\n".join(out)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python renderer.py registry.json sequence.json", file=sys.stderr)
        sys.exit(1)
    print(render_pipeline(sys.argv[1], sys.argv[2]))