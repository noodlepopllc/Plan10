#!/usr/bin/env python3
import sys, json

def main():
    registry_path = sys.argv[1]
    shot_count = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    with open(registry_path) as f:
        reg = json.load(f)

    env_slug = reg["environment_alias"]
    chars = reg.get("characters", [])

    templates = [
        {"shot_type": "two_shot", "desc": "wide environmental framing, both characters centered"},
        {"shot_type": "ots",      "desc": "over-the-shoulder perspective, cinematic focus separation"},
        {"shot_type": "closeup",  "desc": "tight single portrait, shallow depth, expression focused"},
        {"shot_type": "medium",   "desc": "waist-up single framing, relaxed posture, soft background"}
    ]

    out = []
    for i in range(shot_count):
        t = templates[i % len(templates)]
        
        # 🔍 DYNAMIC CHARACTER ORDERING
        # Tool uses pop(0) for single shots, so we put the target character FIRST.
        if t["shot_type"] in ("two_shot", "ots"):
            ordered_aliases = [f"char_{c['alias_slug']}" for c in chars]
        else:
            # Rotate focus: alternates between char 0 and char 1 across shots
            focus_idx = i % len(chars)
            ordered_aliases = [f"char_{chars[focus_idx]['alias_slug']}"]
            # Optional: keep second char in list for context (will be dropped by tool anyway)
            if len(chars) > 1:
                other_idx = (focus_idx + 1) % len(chars)
                ordered_aliases.append(f"char_{chars[other_idx]['alias_slug']}")

        char_refs = ", ".join(ordered_aliases)
        alias = f"cin_{env_slug}_{t['shot_type']}_{i:02d}"
        action = f"{t['desc']}, NO mouth movement, NO speech animation"

        out.append(f"\n>> ALIAS: {alias}")
        out.append(f"composite_scene combining={env_slug}, {char_refs}, shot_type=\"{t['shot_type']}\", action=\"{action}\" Height: 832, Width: 480, Seed: -1")

        vid_alias = f"vid_{alias}"
        out.append(f"\n>> ALIAS: {vid_alias}")
        out.append(f"image_to_video using={alias}, prompt=\"subtle camera drift, atmospheric light shift, NO mouth movement, NO speech animation\", duration_sec=5.0 Height: 832, Width: 480, Seed: -1")

    print("\n".join(out))

if __name__ == "__main__":
    main()