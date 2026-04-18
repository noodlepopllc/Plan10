import json
import os

class LocationPairGenerator:
    """Generates coordinated reverse-shot pairs from a JSON library."""
    
    DEFAULT_JSON_PATH = "tests/locations.json"

    def __init__(self, filepath=None):
        self.filepath = filepath or self.DEFAULT_JSON_PATH
        self.locations = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                return json.load(f)
        raise FileNotFoundError(f"Location file not found: {self.filepath}")

    def get_pair(self, key: str, bg_color: str, lighting: str, camera: str) -> tuple[str, str]:
        loc = self.locations.get(key)
        if not loc:
            raise KeyError(f"Location '{key}' not in {self.filepath}. Keys: {list(self.locations.keys())}")
        
        shared_str = "; ".join(loc["shared"])
        
        # Shot A
        prompt_a = (
            f"{loc['desc']}. "
            f"Shared elements: {shared_str}. "
            f"LEFT: {loc['anchor_a']}. RIGHT: {loc['anchor_b']}. "
            f"BACKGROUND: {loc['forward_bg']} visible. "
            f"Background color tone: {bg_color}. "
            f"Lighting: {lighting}. Technical: {camera}. "
            f"Cinematic, atmospheric, empty of characters or text."
        )
        
        # Shot B (Reverse)
        prompt_b = (
            f"{loc['desc']}. "
            f"Shared elements: {shared_str}. "
            f"LEFT: {loc['anchor_b']}. RIGHT: {loc['anchor_a']}. "
            f"BACKGROUND: {loc['forward_bg']} now behind camera, replaced by consistent opposite architectural view. "
            f"Background color tone: {bg_color}. "
            f"Lighting: {lighting}. Technical: {camera}. "
            f"Cinematic, atmospheric, empty of characters or text."
        )
        
        return prompt_a, prompt_b

    def random_key(self) -> str:
        import random
        return random.choice(list(self.locations.keys()))