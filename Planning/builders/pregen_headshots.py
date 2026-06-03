#!/usr/bin/env python3
import sys, json, re, os

sys.path.append('./lib')

from config import load_environ

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))

def slugify(text: str) -> str:
    """Generate CLI-safe slugs from zone text or names."""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text.strip('_') or "asset"

def main():
    registry_path = sys.argv[1]
    max_beats = int(sys.argv[2]) if len(sys.argv) > 2 else 8  # Pre-gen pool size per character

    with open(registry_path) as f:
        reg = json.load(f)

    chars = reg.get("characters", [])
    env_slug = reg["environment_alias"]
    master_env_alias = f"bg_{env_slug}"

    # 🔑 Standardized closeup actions (mouth-locked, face-only, compositing-ready)
    ALLOWED_MOODS = [
        "neutral", "confident", "skeletal", "encouraging", "curious", 
        "supportive", "shy", "reassuring", "amused", "eager", 
        "nervous", "patient", "relieved", "stern", "defensive", 
        "frustrated", "overwhelmed", "playful"
    ]

    # 🔑 Pre-build zone backdrop map (matches renderer.py logic)
    zone_backdrop_map = {}
    for c in chars:
        slug = c["alias_slug"]
        zone = c.get("background_zone", "center of the room")
        zone_slug = slugify(zone)[:20]
        zone_backdrop_map[slug] = f"{master_env_alias}_zone_{zone_slug}"

    out = []
    
    for c in chars:
        slug = c["alias_slug"]
        # 🔑 ROUTE TO CHARACTER'S ZONE BACKDROP
        backdrop = zone_backdrop_map.get(slug, master_env_alias)
        
        for i in range(1, max_beats + 1):
            mood = ALLOWED_MOODS[(i - 1) % len(ALLOWED_MOODS)]
            base_alias = f"compd_{slug}_{i:02d}"
            action = f"{mood}, mouth completely closed and still, lips sealed shut, zero lip motion, static facial expression, cropped at shoulders, NO hands, NO props, 1:1 reference framing"

            # ── PASS 1: Static Composite Base ──
            out.append(f"\n>> ALIAS: {base_alias}")
            out.append(f"composite_scene combining={backdrop}, char_{slug}, shot_type=\"closeup\", action=\"{action}\" Height: {HEIGHT}, Width: {WIDTH}, Seed: -1")

            # ── PASS 2: Motion Version (I2V) ──
            motion_alias = f"vid_motion_{slug}_{i:02d}"
            motion_prompt = f"{mood}, subtle head tilt, micro breathing, eyes blink naturally, mouth completely closed and still, zero lip motion"
            out.append(f"\n>> ALIAS: {motion_alias}")
            out.append(f"image_to_video using={base_alias}, prompt=\"{motion_prompt}\", duration_sec=2 Height: {HEIGHT}, Width: {WIDTH}, Seed: -1")

            # ── PASS 3a: Dialog from STATIC image (fallback if video end-frame drifts) ──
            dialog_static_alias = f"dialog_static_{slug}_{i:02d}"
            out.append(f"\n>> ALIAS: {dialog_static_alias}")
            out.append(f"dialog_to_video using={base_alias}, audio=design_{slug}, text=\"[DIALOG PLACEHOLDER]\", prompt=\"lips moving naturally, preserve facial structure, NO head motion, NO background drift\" Height: {HEIGHT}, Width: {WIDTH}, Seed: -1")

            # ── PASS 3b: Dialog from MOTION video (primary path when end-frame is clean) ──
            dialog_motion_alias = f"dialog_motion_{slug}_{i:02d}"
            out.append(f"\n>> ALIAS: {dialog_motion_alias}")
            out.append(f"dialog_to_video using={motion_alias}, audio=design_{slug}, text=\"[DIALOG PLACEHOLDER]\", prompt=\"lips moving naturally, preserve existing head motion, NO extra gestures, NO facial drift\" Height: 832, Width: 480, Seed: -1")

            # ── OPTIONAL: Final selection alias (pick best result in post) ──
            # out.append(f"// Choose: {dialog_static_alias} OR {dialog_motion_alias} for final cut")

    print("\n".join(out))

if __name__ == "__main__":
    main()