import json
import os

class LocationPairGenerator:
    DEFAULT_JSON_PATH = "tests/locations.json"

    # TOD-specific window lighting + mandatory ambient fill to prevent spot-only dimness
    TOD_WINDOW_MAP = {
        "golden_hour": "warm golden-hour sunlight streaming through windows",
        "midday": "bright diffused daylight through windows",
        "twilight": "cool blue twilight glow through windows",
        "night": "dark exterior with faint ambient city glow through windows",
        "overcast": "soft diffused overcast light through windows"
    }

    def __init__(self, filepath=None):
        self.filepath = filepath or self.DEFAULT_JSON_PATH
        self.locations = self._load()

    def _load(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Location file not found: {self.filepath}")
        with open(self.filepath, "r") as f:
            return json.load(f)

    def _build_lighting(self, base_lighting: str, time_of_day: str) -> str:
        window_str = self.TOD_WINDOW_MAP.get(time_of_day, self.TOD_WINDOW_MAP["midday"])
        
        # Explicitly add ambient fill + global illumination to counteract spot-only dimness
        return (
            f"{base_lighting}, "
            f"{window_str}, "
            f"balanced with bright warm ambient fill, "
            f"even global illumination filling all shadows, "
            f"high-key subject exposure, no harsh spot-only contrast"
        )

    def get_pair(self, key: str, bg_color: str, lighting: str, camera: str, time_of_day: str = "midday") -> tuple[str, str]:
        loc = self.locations.get(key)
        if not loc:
            raise KeyError(f"Location '{key}' not found. Available: {list(self.locations.keys())}")

        required = ["desc", "shared", "staging", "view_a", "view_b"]
        for r in required:
            if r not in loc:
                raise ValueError(f"Location '{key}' missing required field '{r}' in V2 schema.")

        for view in ("view_a", "view_b"):
            for field in ("left", "right", "background"):
                if field not in loc[view]:
                    raise ValueError(f"Location '{key}' view '{view}' missing '{field}'.")

        shared_str = "; ".join(loc["shared"])
        staging_str = loc["staging"]
        full_lighting = self._build_lighting(lighting, time_of_day)

        base_prompt = (
            f"Location: {loc['desc']}. "
            f"Shared elements: {shared_str}. "
            f"Center staging zone: {staging_str}. "
        )

        va = loc["view_a"]
        vb = loc["view_b"]

        prompt_a = (
            f"{base_prompt}"
            f"VIEW A: LEFT: {va['left']}. RIGHT: {va['right']}. "
            f"BACKGROUND: {va['background']}. "
            f"Color tone: {bg_color}. Lighting: {full_lighting}. Technical: {camera}. "
            f"Cinematic, atmospheric, empty of characters or text."
        )

        prompt_b = (
            f"{base_prompt}"
            f"VIEW B (180° REVERSE ANGLE): LEFT: {vb['left']}. RIGHT: {vb['right']}. "
            f"BACKGROUND: {vb['background']}. "
            f"Color tone: {bg_color}. Lighting: {full_lighting}. Technical: {camera}. "
            f"Cinematic, atmospheric, empty of characters or text."
        )

        return prompt_a, prompt_b
