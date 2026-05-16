#!/usr/bin/env python3
import json
import sys
import re
import os

sys.path.append('./lib')
from config import load_environ

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED","-1"))

def slugify(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text.strip('_') or "asset"

def main():
    if len(sys.argv) < 3:
        print("Usage: python actions.py registry.json actions.json [--images-only]", file=sys.stderr)
        sys.exit(1)
    
    registry_path = sys.argv[1]
    actions_path = sys.argv[2]
    images_only = "--images-only" in sys.argv
    
    with open(registry_path) as f:
        registry = json.load(f)
    with open(actions_path) as f:
        actions_seq = json.load(f)
    
    out = []
    env_slug = slugify(registry.get("environment_alias", "environment"))
    master_env_alias = f"bg_{env_slug}"
    master_prompt = registry["environment"]
    
    # ========================================================================
    # PHASE 1: AGGREGATE PROPS & GENERATE ACTION BACKGROUNDS
    # ========================================================================
    
    # 1. Scan sequence for ALL props needed across all action beats
    all_scene_props = set()
    for beat in actions_seq.get("beats", []):
        if beat.get("type") == "action" and beat.get("props"):
            all_scene_props.update(p.strip() for p in beat["props"].split(",") if p.strip())
            
    props_suffix = f", featuring {', '.join(sorted(all_scene_props))}" if all_scene_props else ""

    # 2. Build character map and action backdrops
    # We generate ONE backdrop per character containing ALL required props
    char_map = {}
    action_backdrop_map = {}
    
    for c in registry["characters"]:
        slug = slugify(c.get("alias_slug", c["name"]))
        char_map[c["id"]] = {"slug": slug}
        
        base_zone = c.get("background_zone", "center of the room")
        # Integrate ALL props into this character's zone
        zone_prompt = f"{base_zone.rstrip('.')}{props_suffix}"
        
        # Create unique alias for action backgrounds
        action_alias = f"bd_{env_slug}_{slug}_action"
        action_backdrop_map[slug] = action_alias
        
        out.append(f'\n>> ALIAS: {action_alias}')
        out.append(f'generate_backdrop media={master_env_alias}, zone="{zone_prompt}", output={action_alias}, Width: 1328, Height: 1328, Seed: {SEED}')

    # ========================================================================
    # PHASE 2: ACTION COMPOSITES
    # ========================================================================
    action_idx = 1
    for beat in actions_seq.get("beats", []):
        if beat.get("type") != "action":
            continue
            
        visible_ids = beat.get("visible_chars", [])
        if not visible_ids:
            continue
            
        shot_type = beat.get("shot_type", "medium")
        motion_prompt = beat.get("motion_prompt", "")
        facial_action = beat.get("facial_action", "neutral")

        if shot_type in ["two_shot","ots"] and len(visible_ids) == 1:
            shot_type = "medium"
        
        if shot_type == "closeup":
            shot_type = "medium"
        
        # Determine focus character & backdrop routing
        focus_cid = visible_ids[0]
        focus_slug = char_map[focus_cid]["slug"]
        
        if shot_type == "ots" and len(visible_ids) > 1:
            backdrop = action_backdrop_map.get(char_map[visible_ids[1]]["slug"], master_env_alias)
        else:
            # Use the character's pre-generated action backdrop
            backdrop = action_backdrop_map.get(focus_slug, master_env_alias)
        
        if len(visible_ids) > 1:
            char_refs = ", ".join([f"char_{char_map[cid]['slug']}" for cid in visible_ids])
        else:
            char_refs = f"char_{focus_slug}"
        
        # Build action prompt
        action_parts = []
        if facial_action:
            mood = facial_action.split(",")[0].strip().lower()
            action_parts.append(f"{mood}")
        if motion_prompt:
            clean_motion = re.sub(r'^\w+:\s*', '', motion_prompt).strip()
            action_parts.append(clean_motion)
        '''
        action_parts.extend([
            "mouth completely closed and still",
            "lips sealed shut",
            "zero lip motion",
            "NO text overlay",
            "NO speech animation"
        ])
        '''
        action = ", ".join([p for p in action_parts if p])
        
        alias = f"action_{action_idx:03d}"
        out.append(f'\n>> ALIAS: {alias}')
        out.append(f'composite_scene combining={backdrop}, {char_refs}, shot_type="{shot_type}", action="{action}" Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}')
        
        # Optional I2V motion pass
        if motion_prompt and beat.get("motion_type") != "static" and not images_only:
            motion_alias = f"vid_action_{action_idx:03d}"
            motion_prompt_clean = re.sub(r'^\w+:\s*', '', motion_prompt).strip()
            out.append(f'\n>> ALIAS: {motion_alias}')
            out.append(f'image_to_video using={alias}, prompt="{motion_prompt_clean}, subtle camera drift, preserve facial expression", duration_sec=5 Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}')
        
        action_idx += 1

    print("\n".join(out))

if __name__ == "__main__":
    main()