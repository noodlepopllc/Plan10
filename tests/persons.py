import sys, random, re, os
from pathlib import Path
from json import load
sys.path.append('./lib')
from image_gen import CreateCharacterSheet, CreateBackground
from config import load_environ
from locations import  LocationPairGenerator

load_environ()

def translate_hairskin(description):
    mapping = {
        "fair-neutral": "fair", "fair-pink": "fair",
        "light-olive": "light", "light-neutral": "light", "light-yellow": "light",
        "medium-olive": "medium", "medium-yellow": "medium", "medium-brown": "medium",
        "golden-brown": "tan", "tan-golden": "tan", "tan-olive": "tan",
        "brown": "tan", "light-brown": "tan",
        "dark-brown": "brown", "deep-brown": "deep", "rich-brown": "deep",
        "dark-umber": "deep", "dark": "deep",
        "dark-blonde": "blonde", "platinum": "blonde", "auburn": "red",
    }

    def normalize_token(token):
        match = re.match(r"^([^\w\-]*)([\w\-]+)([^\w\-]*)$", token)
        if not match:
            return token
        prefix, core, suffix = match.groups()
        normalized_core = mapping.get(core, core)
        return f"{prefix}{normalized_core}{suffix}"

    tokens = description.split()
    normalized = [normalize_token(t) for t in tokens]
    return " ".join(normalized)


class CharacterRecord:
    def __init__(self, name, gender, character_description):
        self.name = name
        self.gender = gender
        self.character_description = "light adult facial definition around the jawline and cheekbones, clearly adult proportions. " + translate_hairskin(character_description)

    @classmethod
    def from_json(cls, data):
        return cls(name=data["name"], gender=data["gender"], character_description=data["characterDescription"])


class ShuffleBag:
    def __init__(self, items):
        self.original = list(items)
        self.bag = []

    def get(self):
        if not self.bag:
            self.bag = self.original[:]
            random.shuffle(self.bag)
        return self.bag.pop()
    
    def reset(self):
        self.bag = []


class DuoPOVScene:
    def __init__(self):
        # Outfits and textures
        self.outfits = ShuffleBag([
            "{color} plunging neckline crop top with matching back coverage and {color} high-waisted mini skirt with continuous back fabric",
            "sheer mesh top over {color} bralette with same back strap layout, mesh top continuing fully around the back, and fitted shorts showing two leg openings in back",
            "{color} off-shoulder crop top with same back neckline and low-rise jeans with consistent waist height",
            "{color} backless sundress with fully open back and same strap layout, same length front and back",
            "{color} strappy bralette with identical back strap pattern, unbuttoned shirt continuing around the back with the same drape, and bike shorts with clear back leg openings",
            "{color} deep V-neck bodysuit with matching back coverage and high-cut legs visible from both sides",
            "{color} cropped tank top with same back neckline and low-cut denim shorts with two visible back leg openings",
            "{color} halter neck crop top with same back strap placement and {color} side-tie mini skirt with continuous back fabric",
            "{color} lace camisole with same back strap layout and {color} satin slip skirt with smooth uninterrupted back panel",
            "{color} cutout bodycon dress with matching back cutout pattern and identical silhouette front and back",
            "{color} tube top with same back height and cargo mini skirt with continuous back fabric and no leg openings",
            "{color} wrap top with matching back coverage and {color} flowing pants with consistent waist height and length",
            "{color} mesh panel crop top with same back panel layout and leather pants with consistent fit and waist height",
            "{color} asymmetric one-shoulder top with matching back asymmetry and {color} micro mini skirt with continuous back fabric"
        ])



        self.textures = ShuffleBag([
            "soft knit with visible stitch detail and natural drape",
            "sheer mesh with realistic transparency and texture",
            "smooth satin with authentic sheen and fluid movement",
            "matte cotton with natural fiber appearance",
            "silky fabric with realistic flow and light catch",
            "delicate lace with intricate pattern and authentic detail",
            "leather with realistic texture and subtle wear",
            "chiffon with airy translucency and natural movement",
            "ribbed fabric with visible texture and stretch",
            "velvet with authentic depth and light absorption"
        ])

        self.color_schemes = ShuffleBag([
            {"char1": "vibrant red", "char2": "electric blue", "background": "deep charcoal"},
            {"char1": "hot pink", "char2": "pure white", "background": "matte black"},
            {"char1": "coral", "char2": "emerald green", "background": "warm beige"},
            {"char1": "gold", "char2": "silver", "background": "deep purple"},
            {"char1": "turquoise", "char2": "magenta", "background": "slate gray"},
            {"char1": "pure white", "char2": "deep navy", "background": "warm sunset orange"},
            {"char1": "emerald green", "char2": "gold", "background": "cool gray"},
            {"char1": "silver", "char2": "burgundy", "background": "warm brown"}
        ])

        self.lighting = ShuffleBag([
            "dramatic rim lighting with warm backlight separating both silhouettes",
            "moody chiaroscuro with high contrast shadows and detail",
            "neon glow with colored rim lights in pink and blue",
            "golden hour backlight with lens flare and warm tones",
            "soft key light with deep shadow play on both subjects",
            "three-point lighting with strong hair light accent on both",
            "practical lighting from neon signs with color cast",
            "dramatic side lighting creating sculptural shadows",
            "backlit silhouette with edge glow and mystery",
            "warm ambient with directional key light for dimension"
        ])

        self.camera_terms = ShuffleBag([
            "shot on full-frame camera, 85mm f/1.4",
            "professional portrait photography, 50mm f/1.2",
            "high-end fashion photography, 135mm f/1.8",
            "editorial quality, medium format, 80mm f/2.8",
            "cinematic portrait, full-frame, 85mm f/1.8",
            "magazine quality, professional lighting, 70mm f/2.0"
        ])

        self.footwear = ShuffleBag([
            "ankle boots",
            "strappy heels",
            "sleek stilettos",
            "chunky sneakers",
            "open-toe heels"
        ])

        self.color_blocking = ShuffleBag([
            "with subtle contrasting trim",
            "with a simple two-tone palette",
            "with light accent detailing",
            "with minimal color contrast"
        ])

    def generate_chars(self, char1, char2, seed=None):
        if seed is not None:
            random.seed(seed)
        colors = self.color_schemes.get()
        
        return {
            "char1_prompt": f'{char1.name.capitalize()} is {char1.character_description} '
                            f"wearing {self.textures.get()} {self.outfits.get().format(color=colors['char1'])}"
                            f"{self.footwear.get()}."
                            "Realistic skin with pores, 8K, studio lighting",
            "char2_prompt": f'{char2.name.capitalize()} is {char2.character_description} '
                            f"wearing {self.textures.get()} {self.outfits.get().format(color=colors['char2'])}"
                            f"{self.footwear.get()}."
                            "Realistic skin with pores, 8K, studio lighting",
            "data": {
                "background": colors["background"],
                "lighting": self.lighting.get(),
                "camera_term": self.camera_terms.get()
            }
        }

    def reset_all_bags(self):
        """Reset all ShuffleBags."""
        for attr in dir(self):
            if isinstance(getattr(self, attr), ShuffleBag):
                getattr(self, attr).reset()

if __name__ == '__main__':
    seed = random.randint(0, 100000000)
    random.seed(seed)
    VARIATIONS_PER_PAIR = 1
    
    with open(sys.argv[1], 'r') as ch:
        output = load(ch) 
    
    scene = DuoPOVScene()

    # --- Argument handling for two characters ---
    if len(sys.argv) >= 4:
        char1_name = sys.argv[2].lower()
        char2_name = sys.argv[3].lower()
        characters = [x for x in output['characters'] 
                     if x['name'].lower() == char1_name or x['name'].lower() == char2_name]
        if len(characters) != 2:
            print(f"❌ Error: Could not find both '{sys.argv[2]}' and '{sys.argv[3]}' in character data")
            sys.exit(1)
        char1_data = [c for c in characters if c['name'].lower() == char1_name][0]
        char2_data = [c for c in characters if c['name'].lower() == char2_name][0]
        character_pairs = [(char1_data, char2_data)]
    elif len(sys.argv) >= 3:
        char_name = sys.argv[2].lower()
        primary_chars = [x for x in output['characters'] if char_name in x['name'].lower()]
        if not primary_chars:
            print(f"❌ Error: Could not find '{sys.argv[2]}' in character data")
            sys.exit(1)
        other_chars = [c for c in output['characters'] if c not in primary_chars]
        character_pairs = [(primary_chars[0], random.choice(other_chars)) for _ in range(5)]
    else:
        all_chars = output['characters']
        character_pairs = [random.sample(all_chars, 2) for _ in range(5)]

    location_gen = LocationPairGenerator()

    for i, (char1_json, char2_json) in enumerate(character_pairs):
        char1 = CharacterRecord.from_json(char1_json)
        char2 = CharacterRecord.from_json(char2_json)
        dirname = f'tests/{char1.name}_{char2.name}'
        Path(dirname).mkdir(exist_ok=True)
        print("#"*50,dirname.upper(),"#"*50)
        scene.reset_all_bags()

        result = scene.generate_chars(char1, char2)
        c1path = f'{dirname}/char1.png'
        c2path = f'{dirname}/char2.png'
        # Generate Character Sheets
        if not Path(c1path).exists():
            CreateCharacterSheet(result["char1_prompt"], c1path, seed=seed)
        if not Path(c2path).exists():
            CreateCharacterSheet(result["char2_prompt"], c2path, seed=seed)
        
        # Extract coordinated dynamic elements
        d = result["data"]
        bg_color = d["background"]
        lighting_desc = d["lighting"]
        camera_desc = d["camera_term"]

        # Pick a random location key, or use a specific one
        location_key = random.choice(list(location_gen.locations.keys()))
        
        prompt_a, prompt_b = location_gen.get_pair(
            key=location_key,
            bg_color=bg_color,
            lighting=lighting_desc,
            camera=camera_desc
        )
        
        print(f"📍 Location: {location_key} | Seed: {seed}")
        l1path = f'{dirname}/location.png'
        l2path= f'{dirname}/location_reverse.png'
        if not Path(l1path).exists():
            CreateBackground(prompt_a, l1path, seed=seed)
        if not Path(l2path).exists():
            CreateBackground(prompt_b, l2path, seed=seed)

        print("#"*100)
