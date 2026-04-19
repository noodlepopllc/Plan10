import json
import os

class LocationPairGenerator:
    DEFAULT_JSON_PATH = "tests/locations.json"

    def __init__(self, filepath=None):
        self.filepath = filepath or self.DEFAULT_JSON_PATH
        self.locations = self._load()

    def _load(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Location file not found: {self.filepath}")
        with open(self.filepath, "r") as f:
            return json.load(f)

    def get_pair(self, key: str, bg_color: str, lighting: str, camera: str) -> tuple[str, str]:
        loc = self.locations.get(key)
        if not loc:
            raise KeyError(f"Location '{key}' not found. Available: {list(self.locations.keys())}")

        shared_str = "; ".join(loc["shared"])
        base_prompt = f"Location: {loc['desc']}. Shared elements: {shared_str}."
        
        # CRITICAL FIX: Clear staging zone without breaking architecture
        staging_constraint = (
            "STAGING CONSTRAINT: Leave a clear, unobstructed floor/ground zone in the lower-center for character placement. "
            "Foreground architecture (railings, steps, low walls, curbs) must be positioned at frame edges or behind this zone. "
            "Zero objects blocking the center staging area. Maintain natural ground plane."
        )

        # --- VIEW A ---
        va = loc["view_a"]
        fg_a = va.get("foreground", "open space, no foreground obstruction")
        prompt_a = (
            f"{base_prompt} "
            f"VIEW A: "
            f"FOREGROUND: {fg_a}. {staging_constraint} "
            f"LEFT: {va['left']}. RIGHT: {va['right']}. "
            f"BACKGROUND: {va['background']} visible. "
            f"Color tone: {bg_color}. Lighting: {lighting}. Technical: {camera}. "
            f"Cinematic, atmospheric, empty of characters or text."
        )

        # --- VIEW B (Reverse Angle) ---
        vb = loc["view_b"]
        fg_b = vb.get("foreground", "open space, no foreground obstruction")
        prompt_b = (
            f"{base_prompt} "
            f"VIEW B (180° REVERSE ANGLE): "
            f"FOREGROUND: {fg_b}. {staging_constraint} "
            f"LEFT: {vb['left']}. RIGHT: {vb['right']}. "
            f"BACKGROUND: {vb['background']} (View A background is now behind camera). "
            f"Color tone: {bg_color}. Lighting: {lighting}. Technical: {camera}. "
            f"Cinematic, atmospheric, empty of characters or text."
        )

        return prompt_a, prompt_b

    def get_closeup_background(self, key: str, bg_color: str, lighting: str, camera: str) -> str:
        loc = self.locations.get(key)
        if not loc: raise KeyError(f"Location '{key}' not found.")
        
        return (
            f"Location atmosphere: {loc['desc']}. "
            f"EXTREME CLOSE-UP BACKDROP: Already heavily defocused with cinematic bokeh. "
            f"Color tone: {bg_color}. Lighting: {lighting}. Technical: {camera}. "
            f"ZERO distinct objects, props, architecture, or hard edges. "
            f"Only smooth ambient light gradients, color wash, and atmospheric haze. "
            f"Pure spatially-neutral environment for tight facial framing. "
            f"Cinematic, atmospheric, empty."
        )