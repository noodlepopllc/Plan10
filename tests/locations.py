import json
import os

class LocationPairGenerator:
    """Standalone location manager for generating coordinated reverse-shot pairs."""
    
    # Default dictionary for immediate testing
    DEFAULT_LOCATIONS = {
        "rooftop_bar": {
            "desc": "rooftop bar at night, city lights bokeh",
            "anchors": ["glass railing with LED strip", "outdoor bar counter"],
            "bg_swap": "distant city skyline"
        },
        "poolside": {
            "desc": "luxury poolside at golden hour, turquoise reflections",
            "anchors": ["limestone pool edge", "teal lounge chairs"],
            "bg_swap": "calm pool water reflecting sky"
        },
        "neon_alley": {
            "desc": "neon-lit alleyway, rain-slicked asphalt, humid atmosphere",
            "anchors": ["brick wall with glowing neon sign", "metal fire escape"],
            "bg_swap": "distant street intersection"
        },
        "hotel_lobby": {
            "desc": "grand hotel lobby, marble floors, evening chandeliers",
            "anchors": ["marble pillar with gold trim", "curved velvet seating"],
            "bg_swap": "illuminated reception desk"
        },
        "moroccan_riad": {
            "desc": "traditional Moroccan riad, zellige tilework, dappled shade",
            "anchors": ["arched corridor with wooden doors", "potted plants with mosaic bench"],
            "bg_swap": "second-story balcony with bright sky"
        }
    }

    def __init__(self, locations_dict=None):
        """Initialize with a dictionary or load from JSON."""
        self.locations = locations_dict or dict(self.DEFAULT_LOCATIONS)

    def get_pair(self, key: str, bg_color: str, lighting: str, camera: str) -> tuple[str, str]:
        """
        Returns (prompt_a, prompt_b) for a given location key.
        Maintains identical lighting/camera/color, swaps spatial anchors, hides forward landmarks.
        """
        loc = self.locations.get(key)
        if not loc:
            raise KeyError(f"Location key '{key}' not found. Available: {list(self.locations.keys())}")
        
        desc = loc["desc"]
        anchors = loc["anchors"]
        bg_swap = loc["bg_swap"]
        
        # Assign anchors for Shot A (Left/Right)
        left_a = anchors[0]
        right_a = anchors[1] if len(anchors) > 1 else "open architectural space"
        
        # Shot A: Forward view
        prompt_a = (
            f"{desc}. "
            f"LEFT: {left_a}. RIGHT: {right_a}. "
            f"BACKGROUND: {bg_swap} visible. "
            f"Background color tone: {bg_color}. "
            f"Lighting: {lighting}. Technical: {camera}. "
            f"Cinematic, atmospheric, empty of characters or text."
        )
        
        # Shot B: Reverse view (swap anchors, hide forward landmark)
        prompt_b = (
            f"{desc}. "
            f"LEFT: {right_a}. RIGHT: {left_a}. "
            f"BACKGROUND: {bg_swap} now behind camera, replaced by consistent opposite view. "
            f"Background color tone: {bg_color}. "
            f"Lighting: {lighting}. Technical: {camera}. "
            f"Cinematic, atmospheric, empty of characters or text."
        )
        
        return prompt_a, prompt_b

    def export_to_json(self, filepath: str):
        """Save current locations to a JSON file for migration."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.locations, f, indent=2)
        print(f"✅ Locations exported to {filepath}")

    @classmethod
    def from_json(cls, filepath: str) -> "LocationPairGenerator":
        """Load locations from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(locations_dict=data)