#!/usr/bin/env python3
import json, sys, re, os

sys.path.append('./lib')
from config import load_environ
load_environ()

WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))
SEED = int(os.environ.get("SEED","-1"))

# ------------------------------------------------------------
# SLUGIFY
# ------------------------------------------------------------
def slugify(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text.strip('_') or "asset"

# ------------------------------------------------------------
# LOAD SINGLE SCENE FROM MULTI-SCENE REGISTRY
# ------------------------------------------------------------
def load_scene_registry(registry, scene_number=None):
    """
    If scene_number is provided → return a normalized single-scene registry.
    If not → return full registry (for global character sheets).
    """
    if scene_number is None:
        return registry

    for scene in registry["scenes"]:
        if scene.get("scene_id") == scene_number:
            return {
                "scene_id": scene_number,
                "scene_alias": slugify(scene["scene_alias"]),
                "environment_alias": slugify(scene["environment_alias"]),
                "environment": scene["environment"],
                "zones": scene["zones"],
                "characters": registry["characters"]
            }

    raise ValueError(f"Scene {scene_number} not found in registry")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python identity.py registry.json [scene_number]", file=sys.stderr)
        sys.exit(1)

    registry_path = sys.argv[1]
    scene_number = int(sys.argv[2]) if len(sys.argv) > 2 else None

    with open(registry_path) as f:
        registry = json.load(f)

    reg = load_scene_registry(registry, scene_number)
    out = []

    # ------------------------------------------------------------
    # GLOBAL CHARACTER SHEETS + VOICES (ONLY IF NO SCENE FILTER)
    # ------------------------------------------------------------
    if scene_number is None:
        for c in registry["characters"]:
            slug = slugify(c["alias_slug"])

            # Character sheet
            out.append(f'\n>> ALIAS: char_{slug}')
            out.append(
                f'create_character_sheet prompt="{c.get("appearance_prompt", "")}" '
                f'Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}'
            )

            # Voice design
            out.append(f'\n>> ALIAS: design_{slug}')
            out.append(f'design_voice voice="{c["voice"]}"')

        print("\n".join(out))
        return

    # ------------------------------------------------------------
    # SCENE-SCOPED SLUGS
    # ------------------------------------------------------------
    scene_slug = f"scene{reg['scene_id']}_{reg['scene_alias']}"
    env_alias = f"bg_{scene_slug}"
    env_prompt = reg["environment"]

    # ------------------------------------------------------------
    # 1. MASTER BACKGROUND FOR THIS SCENE
    # ------------------------------------------------------------
    out.append(f'\n>> ALIAS: {env_alias}')
    out.append(
        f'create_background prompt="{env_prompt}" '
        f'Height: {HEIGHT}, Width: {WIDTH}, Seed: {SEED}'
    )

    # ------------------------------------------------------------
    # 2. ZONE BACKDROPS PER CHARACTER
    # ------------------------------------------------------------
    for zone in reg["zones"]:
        char = next(c for c in reg["characters"] if c["id"] == zone["character_id"])
        char_slug = slugify(char["alias_slug"])
        zone_slug = slugify(zone["background_zone"])[:20]

        zone_alias = f"bd_{scene_slug}_{char_slug}_{zone_slug}"

        out.append(f'\n>> ALIAS: {zone_alias}')
        out.append(
            f'generate_backdrop media={env_alias}, '
            f'zone="{zone["background_zone"]}", '
            f'master_prompt="{env_prompt}", '
            f'output={zone_alias}, Width: 1328, Height: 1328, Seed: {SEED}'
        )

    print("\n".join(out))


if __name__ == "__main__":
    main()
