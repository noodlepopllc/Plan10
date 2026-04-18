#!/usr/bin/env python3
"""
DUO CHARACTER GENERATOR - LOW-RISK POSES
Two-character portrait generation with clean anatomy.
High-risk contact poses removed to avoid limb artifacts.

Usage:
    python generate_duo_chars.py characters.json [char1] [char2]
    Example: python generate_duo_chars.py characters.json Aria Jade
"""

import sys, random
from glob import glob
from pathlib import Path
from json import load
import re, os
import sys
sys.path.append('./lib')
from image_gen import CreateCharacterSheet, CreateBackground, GenerateReverseBackground
from config import load_environ

load_environ()
print(os.environ)

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
        self.character_description = translate_hairskin(character_description)

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
            "plunging neckline crop top with high-waisted mini skirt",
            "sheer mesh top over bralette with fitted shorts",
            "off-shoulder crop top with low-rise jeans",
            "backless sundress with thigh-high slit",
            "strappy bralette with unbuttoned shirt and bike shorts",
            "deep V-neck bodysuit with high-cut legs",
            "cropped tank top with low-cut denim shorts",
            "halter neck crop top with side-tie mini skirt",
            "lace camisole with satin slip skirt",
            "cutout bodycon dress with strategic openings",
            "tube top with cargo mini skirt",
            "wrap top with deep plunge and flowing pants",
            "mesh panel crop top with leather pants",
            "asymmetric one-shoulder top with micro mini skirt"
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

        self.locations = ShuffleBag([
            "rooftop bar at night with city lights bokeh",
            "luxury poolside with turquoise water reflections",
            "neon-lit alleyway with vibrant urban glow",
            "penthouse balcony with skyline backdrop",
            "beach at golden hour with ocean waves",
            "rain-soaked street with neon reflections",
            "luxury hotel lobby with marble and chandeliers",
            "yacht deck at sunset with ocean horizon",
            "dimly lit speakeasy with vintage ambiance",
            "Moroccan riad with intricate tilework",
            "Tokyo street at night with neon signs",
            "Santorini terrace with white architecture and sea view",
            "Dubai rooftop with futuristic skyline",
            "Barcelona balcony with Mediterranean sunset"
        ])

        # 🔹 9:16 OPTIMIZED: Vertical framing poses - NO PHYSICAL CONTACT
        self.pov = ShuffleBag([
            "medium two-shot, both subjects centered vertically with slight overlap",
            "eye-level portrait framing both faces in upper third of composition",
            "tight vertical crop focusing on upper bodies and faces",
            "straight-on portrait with both subjects filling vertical frame",
            "slight upward angle emphasizing height and presence",
            "intimate close two-shot with faces occupying upper half of frame",
            "side-by-side portrait with both faces clearly visible in upper frame",
            "vertical composition with subjects at similar height, balanced framing"
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

        self.expressions = ShuffleBag([
            "both sharing genuine laughter with crinkled eyes, pure joy",
            "playful smirks with sparkling eyes and mischievous energy",
            "warm smiles reaching the eyes with authentic warmth",
            "candid mid-laugh expressions with natural joy visible",
            "soft smiles with warm eyes and inviting expressions",
            "bold direct gazes with slight smiles and magnetic presence",
            "laughing expressions with genuine happiness, eyes lit up",
            "relaxed expressions with natural comfort between them"
        ])

        # 🔹 LOW-RISK POSES ONLY - NO CONTACT, NO OVERLAPPING LIMBS
        self.poses = ShuffleBag([
            "standing side-by-side with slight stagger, both facing camera, arms relaxed at sides",
            "both leaning against wall, bodies parallel, no physical contact",
            "standing close but not touching, natural personal space, vertical alignment",
            "one slightly in front of other, depth separation, no overlapping limbs",
            "standing side-by-side, weight shifted naturally, hands out of frame or visible",
            "both facing camera with relaxed postures, shoulders aligned, no contact",
            "standing with slight angle toward each other, engaged but not touching",
            "side-by-side stance with natural gap between them, clean vertical composition",
            "both standing straight, relaxed shoulders, arms naturally positioned",
            "standing close with slight depth offset, no limb overlap, clean framing"
        ])

        self.camera_terms = ShuffleBag([
            "shot on full-frame camera, 85mm f/1.4",
            "professional portrait photography, 50mm f/1.2",
            "high-end fashion photography, 135mm f/1.8",
            "editorial quality, medium format, 80mm f/2.8",
            "cinematic portrait, full-frame, 85mm f/1.8",
            "magazine quality, professional lighting, 70mm f/2.0"
        ])

        self.quality_tags = ShuffleBag([
            "natural skin texture with visible pores, no plastic appearance",
            "realistic subsurface scattering, authentic skin tones",
            "high-fidelity detail, no AI artifacts, real people",
            "accurate anatomical proportions, no distortions",
            "natural eye moisture and reflections, lifelike gazes",
            "realistic fabric detail with authentic texture",
            "dimensional lighting with shadow modeling, not flat",
            "professional color grading, natural tones, no waxy finish"
        ])

    def generate_chars(self, char1, char2, seed=None):
        if seed is not None:
            random.seed(seed)
        colors = self.color_schemes.get()
        data = {
            "outfit_1": self.outfits.get(),
            "outfit_2": self.outfits.get(),
            "texture_1": self.textures.get(),
            "texture_2": self.textures.get(),
            "char1_color": colors["char1"],
            "char2_color": colors["char2"],
            "background": colors["background"],
            "location": self.locations.get(),
            "lighting": self.lighting.get(),
            "camera_term": self.camera_terms.get()
        }
        return [
            (f'{char1.name.capitalize()} is {char1.character_description} ' 
            f"wearing {data['char1_color']} {data['texture_1']} {data['outfit_1']}. "),         
            (f'{char2.name.capitalize()} is {char2.character_description} ' 
            f"wearing {data['char2_color']} {data['texture_2']} {data['outfit_2']}. "),
            (f"Location: {data['location']} with {data['background']} background. "
            f"Lighting: {data['lighting']}. "
            f"Technical: {data['camera_term']}. ")
        ]

    def reset_all_bags(self):
        """Reset all ShuffleBags."""
        for attr in dir(self):
            if isinstance(getattr(self, attr), ShuffleBag):
                getattr(self, attr).reset()


def generate_duo_image(pipe, prompt, output_path, generation_seed):
    """Simple generation without identity verification."""
    try:
        image = pipe(prompt, width=928, height=1664, cfg_scale=1.0, 
                    seed=generation_seed, num_inference_steps=8)
        image.save(output_path)
        return True
    except Exception as e:
        print(f"   ⚠️  Generation failed: {e}")
        return False


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


    for i, (char1_json, char2_json) in enumerate(character_pairs):
        char1 = CharacterRecord.from_json(char1_json)
        char2 = CharacterRecord.from_json(char2_json)
        dirname = f'tests/{char1.name}_{char2.name}'
        os.mkdir(dirname)
        print("#"*50,dirname.upper(),"#"*50)
        scene.reset_all_bags()
        var_prompts = scene.generate_chars(char1, char2)
        fn, prompt = (f'{dirname}/char1.png',var_prompts.pop(0))
        print(fn, prompt)
        CreateCharacterSheet(prompt, fn)
        fn, prompt = (f'{dirname}/char2.png',var_prompts.pop(0))
        print(fn, prompt)
        CreateCharacterSheet(prompt, fn)
        fn, prompt = (f'{dirname}/location.png',var_prompts.pop())
        print(fn, prompt)
        CreateBackground(prompt, fn)
        GenerateReverseBackground(f'{dirname}/location.png', f'{dirname}/location_reverse.png')
        print("#"*100)

