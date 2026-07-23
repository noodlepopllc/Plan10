import json
from pathlib import Path
import random

BODY_TYPE_VISUAL = {
    "thin": "narrow shoulders, long slender torso, slim arms, narrow hips, slender legs",
    "fit": "balanced shoulders, defined waist, visible arm tone, proportional hips, toned legs",
    "athletic": "broad shoulders, muscular upper body, defined arms, medium hips, strong legs",
    "soft": "rounded shoulders, soft torso, gentle waist definition, fuller hips, soft legs"
}

AGE_MORPHOLOGY_VISUAL = {
    "youthful": "full cheeks, smooth luminous skin, minimal under‑eye definition",
    "young_adult": "smooth skin with subtle definition around the eyes and cheeks",
    "adult": "fine lines around the eyes and mouth, reduced cheek fullness",
    "mature": "visible skin texture, deeper nasolabial folds, low soft‑tissue volume"
}

class CharacterLUT:
    def __init__(self, lut_path="LUT"):
        self.body_lut = self._load_json(Path(lut_path) / "body.json")
        self.gender_ethnicity_lut = self._load_json(Path(lut_path) / "gender_ethnicity.json")
        self.skin_hair_lut = self._load_json(Path(lut_path) / "skin_hair.json")
        self.hair_silhouette_lut = self._load_json(Path(lut_path) / "hair_silhouette.json")
        self.contrast_lut = self._load_json(Path(lut_path) / "contrast.json")
    
    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

class CharacterIdentity:
    def __init__(
        self,
        height,
        ethnicity,
        gender,
        age,
        body_type,
        skin_tone,
        hair_color,
        hair_texture=None,
        hair_silhouette=None,
        contrast="MEDIUM",
        overrides=None,
        lut_path="LUT",
        mode='default',
        seed=None,
        lut=None
    ):

        self.height = height
        self.ethnicity = ethnicity
        self.gender = gender
        self.age = age
        self.body_type = body_type
        self.skin_tone = skin_tone
        self.hair_color = hair_color
        self.hair_texture = hair_texture
        self.hair_silhouette = hair_silhouette

        self.contrast = contrast.upper() if contrast.upper() in ['LOW','MEDIUM','HIGH'] else 'MEDIUM'
        self.overrides = overrides or {}

        if lut:
            self.body_lut = lut.body_lut
            self.gender_ethnicity_lut = lut.gender_ethnicity_lut
            self.skin_hair_lut = lut.skin_hair_lut
            self.hair_silhouette_lut = lut.hair_silhouette_lut
            self.contrast_lut = lut.contrast_lut

        else:
            # load LUTs
            self.body_lut = self._load_json(Path(lut_path) / "body.json")
            self.gender_ethnicity_lut = self._load_json(Path(lut_path) / "gender_ethnicity.json")
            self.skin_hair_lut = self._load_json(Path(lut_path) / "skin_hair.json")
            self.hair_silhouette_lut = self._load_json(Path(lut_path) / "hair_silhouette.json")
            self.contrast_lut = self._load_json(Path(lut_path) / "contrast.json")

        self.gender_hair = self.hair_silhouette_lut['GENDER_HAIR'][self.gender]
        self.hair_description = self.hair_silhouette_lut["HAIR_DESCRIPTION"]

        # unpack modules
        self.height_data = self.body_lut["HEIGHT"]
        self.body_data = self.body_lut["BODY_TYPE"]
        self.age_data = self.body_lut["AGE_MORPHOLOGY"]

        self.gender_data = self.gender_ethnicity_lut["GENDER_FACE"]
        self.ethnicity_data = self.gender_ethnicity_lut["ETHNICITY_DEFAULTS"]

        self.skin_tone_data = self.skin_hair_lut["SKIN_TONE"]
        self.hair_color_data = self.skin_hair_lut["HAIR_COLOR"]
        self.hair_silhouette_data = self.hair_silhouette_lut["HAIR_SILHOUETTE"]
        self.mode = mode
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random

        # resolve final identity block
        self.resolved = self._resolve()

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _resolve(self):
        merged = {}

        # merge in deterministic order
        merge_order = [
            self.height_data.get(self.height, {}),
            self.ethnicity_data.get(self.ethnicity, {}),
            self.gender_data.get(self.gender, {}),
            self.age_data.get(self.age, {}),
            self.body_data.get(self.body_type, {}),
        ]
        eth = self.ethnicity_data.get(self.ethnicity, {})
        # hair texture from ethnicity unless overridden
        if self.hair_texture is None:
            if "hairTexture" in eth:
                merged["hairTexture"] = eth["hairTexture"]["default"]
        else:
            merged["hairTexture"] = self.hair_texture

        # hair silhouette selection
        if self.hair_silhouette is None:
            if "hairSilhouette" in eth:
                if self.mode == "default":
                    merged["hairSilhouette"] = eth["hairSilhouette"]["default"]
                else:
                    ethnic = set(eth["hairSilhouette"].get("allowed"))
                    gender = set(self.gender_hair['silhouetteBias'])

                    allowed = sorted(ethnic & gender)
                    merged["hairSilhouette"] = self.rng.choice(allowed) if allowed else eth["hairSilhouette"]["default"]
        else:
            merged["hairSilhouette"] = self.hair_silhouette

        # skin tone semantic selection
        if self.skin_tone in self.skin_tone_data:
            merged["skinTone"] = self.skin_tone_data[self.skin_tone][0]

        # hair color semantic selection
        if self.hair_color in self.hair_color_data:
            merged["hairColor"] = self.hair_color_data[self.hair_color][0]
            
        COMPOSITE_FIELDS = {"hairSilhouette"}

        # apply all LUT layers
        for layer in merge_order:
            for key, value in layer.items():
                if key in COMPOSITE_FIELDS:
                    continue
                if isinstance(value, dict):
                    if self.mode == "default":
                        merged[key] = value.get("default")
                    else:
                        allowed = value.get("allowed")
                        if allowed:
                            merged[key] = self.rng.choice(allowed)
                        else:
                            merged[key] = value.get("default")
                else:
                    merged[key] = value

        # apply overrides last
        for key, value in self.overrides.items():
            merged[key] = value

        # melanin contrast layer
        if self.contrast in self.contrast_lut:
            for key, value in self.contrast_lut[self.contrast].items():
                merged[key] = value

        return merged

    def describe(self):
        r = self.resolved

        face = (
            f"{r.get('jawlineContour')} jawline, "
            f"{r.get('cheekboneHeight')} cheekbones, "
            f"{r.get('eyeShape')} eyes, "
            f"{r.get('noseBridgeShape')} nasal bridge, "
            f"{r.get('noseTipShape')} nose tip"
        )
        contrast_bits = (
            f"{r.get('shadowResponse')} shadow response, "
            f"{r.get('browDensity')} brow density, "
            f"{r.get('lashDefinition')} lash definition"
        )


        # --- AGE ---
        age_visual = AGE_MORPHOLOGY_VISUAL[self.age]
        age_detail = (
            f"{r.get('softTissueVolume')} soft‑tissue volume, "
            f"{r.get('cheekFullness')} cheek fullness, "
            f"{r.get('underEyeDefinition')} under‑eye definition, "
            f"{r.get('nasolabialFoldDepth')} nasolabial fold depth, "
            f"{r.get('skinTexture')} skin texture"
        )

        age = f"{age_visual}, with {age_detail}"

        # --- BODY ---
        body_visual = BODY_TYPE_VISUAL[self.body_type]
        body_detail = (
            f"{r.get('shoulderWidth')} shoulders, "
            f"{r.get('torsoProportion')} torso proportion, "
            f"{r.get('waistDefinition')} waist definition, "
            f"{r.get('hipWidth')} hips, "
            f"{r.get('muscleDefinition')} muscle definition"
        )

        body = f"{body_visual}, with {body_detail}"
        hair_token = r.get('hairSilhouette', '').replace(' ', '_').replace('-', '_').lower()

        hair_description = self.hair_description.get(hair_token)

        hair = (
            f"{hair_description} with "
            f"{r.get('hairTexture')} texture and "
            f"{r.get('hairColor')} color"
        )


        return (
            f"A {self.age.replace('_', ' ')} {self.gender} presentation "
            f"{self.ethnicity.replace('_', ' ')} individual. "
            f"Facial structure includes a {face}. "
            f"Contrast features include {contrast_bits}. "

            f"Skin tone is {r.get('skinTone')}. "
            f"Hair has {hair}. "
            f"Age morphology shows {age}. "
            f"Body silhouette is {body}."
        )
    def to_character_json(self, name=""):
        return {
            "name": name,
            "role": "",
            "biography": "",
            "personality": "",
            "clothingDescription": "",
            "theme": "",
            "setting": "",
            "visualStyle": "",
            "notes": "",
            "characterDescription": self.describe(),
            "height": self.height,
            "ethnicity": self.ethnicity,
            "gender": self.gender,
            "age": self.age,
            "body_type": self.body_type,
            "skin_tone": self.skin_tone,
            "hair_color": self.hair_color,
            "hair_texture": self.hair_texture,
            "hair_silhouette": self.hair_silhouette,
            "identity": self.resolved
        }

    @classmethod
    def from_json(cls, data, lut_path="LUT"):
        return cls(
            height=data["height"],
            ethnicity=data["ethnicity"],
            gender=data["gender"],
            age=data["age"],
            body_type=data["body_type"],
            skin_tone=data["skin_tone"],
            hair_color=data["hair_color"],
            hair_texture=data.get("hair_texture"),
            hair_silhouette=data.get("hair_silhouette"),
            overrides=data.get("identity", {}),
            lut_path=lut_path
    )


def export_characters(*characters):
    return {
        "characters": [char.to_character_json() for char in characters]
    }

    @classmethod
    def from_resolved(cls, data, lut_path="LUT"):
        obj = cls.__new__(cls)  # bypass __init__
        
        # raw semantic fields
        obj.height = data["height"]
        obj.ethnicity = data["ethnicity"]
        obj.gender = data["gender"]
        obj.age = data["age"]
        obj.body_type = data["body_type"]
        obj.skin_tone = data["skin_tone"]
        obj.hair_color = data["hair_color"]
        obj.hair_texture = data.get("hair_texture")
        obj.overrides = {}

        # load LUTs (optional, but harmless)
        obj.body_lut = obj._load_json(Path(lut_path) / "body.json")
        obj.gender_ethnicity_lut = obj._load_json(Path(lut_path) / "gender_ethnicity.json")
        obj.skin_hair_lut = obj._load_json(Path(lut_path) / "skin_hair.json")

        obj.height_data = obj.body_lut["HEIGHT"]
        obj.body_data = obj.body_lut["BODY_TYPE"]
        obj.age_data = obj.body_lut["AGE_MORPHOLOGY"]

        obj.gender_data = obj.gender_ethnicity_lut["GENDER_FACE"]
        obj.ethnicity_data = obj.gender_ethnicity_lut["ETHNICITY_DEFAULTS"]

        obj.skin_tone_data = obj.skin_hair_lut["SKIN_TONE"]
        obj.hair_color_data = obj.skin_hair_lut["HAIR_COLOR"]

        # the important part:
        obj.resolved = data["identity"]

        return obj


class CharacterRegistry:
    def __init__(self):
        self.characters = {}

    def add(self, name, character_identity, clothing_description="", seed=-1):
        entry = character_identity.to_character_json(name=name)
        entry["clothingDescription"] = clothing_description
        entry["seed"] = seed
        self.characters[name] = entry

    def get(self, name):
        return self.characters.get(name)

    def update_clothing(self, name, clothing_description):
        if name in self.characters:
            self.characters[name]["clothingDescription"] = clothing_description

    def update_description(self, name, new_description):
        if name in self.characters:
            self.characters[name]["characterDescription"] = new_description

    def export(self):
        return {
            "characters": list(self.characters.values())
        }

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

class Outfit:
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

        self.footwear = ShuffleBag([
            "ankle boots",
            "strappy heels",
            "sleek stilettos",
            "chunky sneakers",
            "open-toe heels"
        ])


    def generate_outfit(self, seed=None):
        if seed is not None:
            random.seed(seed)
        colors = self.color_schemes.get()
        
        return ( f" wearing {self.textures.get()} {self.outfits.get().format(color=colors['char1'])} "
                 f"{self.footwear.get()}. "
                  "Realistic skin with pores, 8K, studio lighting ")

female_names = [
    'Mary',
    'Elizabeth',
    'Patricia',
    'Jennifer',
    'Linda',
    'Barbara',
    'Susan',
    'Jessica',
    'Sarah',
    'Lisa',
    'Margaret',
    'Nancy',
    'Karen',
    'Kimberly',
    'Michelle',
    'Laura',
    'Amanda',
    'Emily',
    'Sandra',
    'Rebecca'
]

lut = CharacterLUT('./tests/LUT')
genders = ["masculine","feminine","androgynous"]
contrast = ["LOW","MEDIUM","HIGH"]
ethnicities = ["east_asian", "south_asian", "sub_saharan_african", "middle_eastern", "northern_european", "southern_european","latinx_mestizo"]
hair_colors = ["black","blonde","brown","red","gray"]
skin_tones = ["fair","light","medium","tan","deep"]
heights = ["short","average","tall"]
body_types = ["thin","fit","athletic","soft"]
ages = ["youthful","young adult","adult","mature"]
hair_silhouettes = lut.hair_silhouette_lut["HAIR_SILHOUETTE"]["allowed"][0]

def reroll_clothing(name):
    with open(f'./tests/{name}/{name}.json', 'r') as char:
        this_char = json.load(char)
    this_char['characters'][0]['clothingDescription'] = Outfit().generate_outfit()
    CreateCharacterSheet(
        prompt=f'{this_char['characters'][0]['characterDescription']} wearing {this_char['characters'][0]['clothingDescription']}', 
        output=f'{output}/{name}.png', 
        seed=int(this_char['characters'][0]['seed']))
    with open(f'./tests/{name}/{name}.json', 'w') as char:
        this_char = json.dump(this_char, char, indent=4)

def main():
    import argparse, json
    import random

    
    parser = argparse.ArgumentParser()
    parser.add_argument('-G', '--gender', type=str, default='feminine', help=f"Gender must be one of {', '.join(genders)}")
    parser.add_argument('-D', '--randomize', action='store_true', help='random person')
    parser.add_argument('-R', '--ethnicity', type=str, default='east_asian', help=f"Ethnicity must be one of {', '.join(ethnicities)}")
    parser.add_argument('-A', '--age', type=str, default='young_adult', help=f"Age must be one of {', '.join(ages)}")
    parser.add_argument('-S', '--hair-style', type=str, default='tight single ponytail', help=f"Hair Style must be one of {', '.join(hair_silhouettes)}")
    parser.add_argument('-C', '--hair-color', type=str, default='brown', help=f"Hair color must be one of {', '.join(hair_colors)}")
    parser.add_argument('-T', '--skin-tone', type=str, default='medium', help=f"Skin tone must be one of {', '.join(skin_tones)}")
    parser.add_argument('-H', '--height', type=str, default='average', help=f"Height must be one of {', '.join(heights)}")
    parser.add_argument('-B', '--body-type', type=str, default='thin', help=f"Body type must be one of {', '.join(body_types)}")
    parser.add_argument('-N', '--name', type=str, default='Emily', help="Name of the character")
    parser.add_argument('-U', '--outfit', type=str, default='', help='Clothing character is wearing')
    parser.add_argument('-E', '--seed', type=int, default=42, help='seed')
    parser.add_argument('-Z', '--reroll-clothing', action="store_true", help='reroll the character of name')
    args = parser.parse_args()

    if args.randomize:
        args.ethnicity = random.choice(ethnicities) if args.ethnicity == 'random' else args.ethnicity
        args.name = random.choice(female_names) if args.name == 'random' else args.name.capitalize()
        args.hair_color = random.choice(hair_colors) if args.hair_color == 'random' else args.hair_color
        args.hair_style = random.choice(hair_silhouettes) if args.hair_style == 'random' else args.hair_style
        args.seed = random.randint(0,1000000)
        args.height = random.choice(heights) if args.height == 'random' else args.height

    import sys, os
    sys.path.append('./lib')
    from config import load_environ

    load_environ()
    if os.environ.get('ANIME','False') != 'False':
        from anime_gen import ImageGen, CreateCharacterSheet
    else:
        from image_gen import ImageGen, CreateCharacterSheet
    from pathlib import Path

    output = f'./tests/{args.name}'
    p = Path(output)
    if p.exists():
        if args.reroll_clothing:
            reroll_clothing(args.name)
        else:
            print(f'{args.name} Already exists')
        sys.exit(1)
    Path(output).mkdir(exist_ok=True)

    outfit = args.outfit
    if not outfit:
        outfit = Outfit().generate_outfit(seed=args.seed)

    char = None

    seed = random.randint(0,1000000) if args.seed == -1 else args.seed

    with ImageGen() as igen:
        char = CharacterIdentity(
            height=args.height,
            ethnicity=args.ethnicity,
            gender=args.gender,
            age=args.age,
            body_type=args.body_type,
            skin_tone=args.skin_tone,
            hair_color=args.hair_color,
            hair_silhouette=args.hair_style,
            contrast="LOW",
            mode='random',
            seed=seed,
            lut_path='./tests/LUT',
            lut=lut
        )
        CreateCharacterSheet(
            prompt=f'{char.describe()} wearing {outfit}', 
            output=f'{output}/{args.name}.png', 
            seed=seed, 
            imagegen=igen)

    registry = CharacterRegistry()
    registry.add(
        name=args.name,
        character_identity=char,
        clothing_description=outfit,
        seed=seed
    )
    reg = registry.export()
    with open(f'{output}/{args.name}.json', 'w') as js:
        json.dump(reg, js, indent=4)

if __name__ == '__main__':
    main()

