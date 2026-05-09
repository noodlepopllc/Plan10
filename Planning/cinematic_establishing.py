#!/usr/bin/env python3
import sys, json, re

def main():
    registry_path = sys.argv[1]
    shot_count = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    with open(registry_path) as f:
        reg = json.load(f)

    env_slug = 'bg_' + reg["environment_alias"]
    chars = reg.get("characters", [])

    out = []

    # 1️⃣ GENERATE CHARACTER-SPECIFIC ZONE BACKDROPS
    zone_aliases = {}
    for c in chars:
        # Sanitize zone text into a safe CLI alias slug
        zone_slug = re.sub(r'[^a-z0-9]', '_', c["background_zone"].lower())[:24].strip('_')
        backdrop_alias = f"{env_slug}_zone_{zone_slug}"
        zone_aliases[c["alias_slug"]] = backdrop_alias

        out.append(f"\n>> ALIAS: {backdrop_alias}")
        out.append(f"generate_backdrop using={env_slug}, zone=\"{c['background_zone']}\", output={backdrop_alias}, Width: 1328, Height: 1328, Seed: -1")

    # 2️⃣ GENERATE SHOT SEQUENCES
    templates = [
        {"shot_type": "two_shot", "desc": "wide environmental framing, both characters centered"},
        {"shot_type": "ots",      "desc": "over-the-shoulder perspective, cinematic focus separation"},
        {"shot_type": "medium",   "desc": "waist-up single framing, relaxed posture, soft background"},
        {"shot_type": "medium",   "desc": "waist-up single framing, relaxed posture, soft background"},
    ]

    for i in range(shot_count):
        t = templates[i % len(templates)]

        # Determine focus character & matching zone backdrop
        if t["shot_type"] in ("two_shot", "ots"):
            # Wide/OTS use first character's zone as base anchor
            focus_alias = chars[0]["alias_slug"]
            char_refs = ", ".join([f"char_{c['alias_slug']}" for c in chars])
        else:
            # Rotate focus for medium shots
            focus_idx = i % len(chars)
            focus_alias = chars[focus_idx]["alias_slug"]
            char_refs = f"char_{focus_alias}"

        base_backdrop = zone_aliases[focus_alias]
        alias = f"cin_{env_slug}_{t['shot_type']}_{i:02d}"
        action = f"{t['desc']}, NO mouth movement, NO speech animation"

        out.append(f"\n>> ALIAS: {alias}")
        out.append(f"composite_scene combining={base_backdrop}, {char_refs}, shot_type=\"{t['shot_type']}\", action=\"{action}\" Height: 832, Width: 480, Seed: -1")

        vid_alias = f"vid_{alias}"
        out.append(f"\n>> ALIAS: {vid_alias}")
        out.append(f"image_to_video using={alias}, prompt=\"subtle camera drift, atmospheric light shift, NO mouth movement, NO speech animation\", duration_sec=5 Height: 832, Width: 480, Seed: -1")

    print("\n".join(out))

if __name__ == "__main__":
    main()