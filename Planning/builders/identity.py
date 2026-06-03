#!/usr/bin/env python3
import json, sys, re, os
from PIL import Image

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
    if len(sys.argv) != 2:
        print("Usage: python identity.py registry.json", file=sys.stderr)
        sys.exit(1)
    
    registry_path = sys.argv[1]
    with open(registry_path) as f:
        registry = json.load(f)
    
    out = []
    env_slug = slugify(registry.get("environment_alias", "environment"))
    master_env_alias = f"bg_{env_slug}"
    master_prompt = registry["environment"]
    
    # 1a. Master environment
    out.append(f'>> ALIAS: {master_env_alias}')
    out.append(f'create_background prompt="{master_prompt}" Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}')
    
    # 1b. Zone backdrops (bare-bones: no prop aggregation here)
    for c in registry["characters"]:
        slug = slugify(c.get("alias_slug", c["name"]))
        zone = c.get("background_zone", "center of the room")
        zone_slug = slugify(zone)[:20]
        zone_alias = f"bd_{env_slug}_zone_{zone_slug}"
        
        char_ref_path = f"assets/char_{slug}.png"
        should_bake = c.get("staged_character", False) and os.path.exists(char_ref_path)
        
        out.append(f'\n>> ALIAS: {zone_alias}')
        if should_bake:
            out.append(f'generate_backdrop media={master_env_alias}, zone="{zone}", master_prompt="{master_prompt}", char_image="{char_ref_path}", output={zone_alias}, Width: 1328, Height: 1328, Seed: {SEED}')
        else:
            out.append(f'generate_backdrop media={master_env_alias}, zone="{zone}", master_prompt="{master_prompt}", output={zone_alias}, Width: 1328, Height: 1328, Seed: {SEED}')
    
    # 1c. Character sheets + voice designs
    for c in registry["characters"]:
        slug = slugify(c.get("alias_slug", c["name"]))
        
        out.append(f'\n>> ALIAS: char_{slug}')
        out.append(f'create_character_sheet prompt="{c["appearance_prompt"]}" Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}')
        
        out.append(f'\n>> ALIAS: design_{slug}')
        out.append(f'design_voice voice="{c["voice"]}"')
    
    print("\n".join(out))

if __name__ == "__main__":
    main()