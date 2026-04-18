import sys, random
from glob import glob
from pathlib import Path
from json import load
import re, os
import sys
sys.path.append('./lib')
from image_gen import CreateCharacterSheet, CreateBackground, GenerateReverseBackground
from config import load_environ
from locations import LocationPairGenerator

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

        # REPLACES self.locations -> self.location_layouts
        self.location_layouts = ShuffleBag([
            {"base": "modern penthouse balcony at night", 
             "view_a": "LEFT: floor-to-ceiling sliding glass doors. RIGHT: black glass railing. CENTER: open lounge space. BACKGROUND: {bg} city skyline visible.", 
             "view_b": "LEFT: black glass railing. RIGHT: floor-to-ceiling sliding glass doors. CENTER: open lounge space. BACKGROUND: dimly lit interior lounge, skyline replaced by warm architectural details."},
            {"base": "luxury poolside at golden hour", 
             "view_a": "LEFT: white limestone pool edge. RIGHT: two lounge chairs under large umbrella. CENTER: open tiled area. BACKGROUND: calm water reflecting {bg} tones.", 
             "view_b": "LEFT: two lounge chairs under umbrella. RIGHT: white limestone pool edge. CENTER: open tiled area. BACKGROUND: tropical garden wall with dappled foliage, pool not in view."},
            {"base": "neon-lit urban alleyway at night", 
             "view_a": "LEFT: brick facade with glowing vending machines. RIGHT: narrow service entrance. CENTER: rain-slicked asphalt. BACKGROUND: {bg} street intersection with traffic bokeh.", 
             "view_b": "LEFT: narrow service entrance. RIGHT: brick facade with glowing vending machines. CENTER: rain-slicked asphalt. BACKGROUND: layered poster wall and utility pipes, street glow behind camera."},
            {"base": "grand hotel lobby in evening", 
             "view_a": "LEFT: marble pillar with gold trim. RIGHT: curved velvet seating area. CENTER: wide polished floor. BACKGROUND: illuminated reception desk, {bg} ambient glow.", 
             "view_b": "LEFT: curved velvet seating area. RIGHT: marble pillar with gold trim. CENTER: wide polished floor. BACKGROUND: grand double doors, soft dusk light filtering through glass."},
            {"base": "traditional Moroccan riad courtyard", 
             "view_a": "LEFT: arched corridor with carved wooden doors. RIGHT: lush potted plants and mosaic benches. CENTER: geometric tile path. BACKGROUND: second-story balcony, bright {bg} sky above.", 
             "view_b": "LEFT: lush potted plants and mosaic benches. RIGHT: arched corridor with carved wooden doors. CENTER: geometric tile path. BACKGROUND: ornate courtyard entrance, filtered daylight on floor."},
            {"base": "Parisian café terrace on overcast day", 
             "view_a": "LEFT: café glass window with chalkboard menu. RIGHT: row of bistro tables under striped awning. CENTER: wet cobblestone path. BACKGROUND: blurred {bg} street traffic and distant signs.", 
             "view_b": "LEFT: row of bistro tables under striped awning. RIGHT: café glass window with chalkboard menu. CENTER: wet cobblestone path. BACKGROUND: warm interior café glow visible through glass, empty booths."},
            {"base": "desert modernist courtyard at sunset", 
             "view_a": "LEFT: smooth terracotta wall with recessed planter. RIGHT: concrete bench with copper fire pit. CENTER: gravel area. BACKGROUND: low mountain ridge silhouetted against {bg} sky.", 
             "view_b": "LEFT: concrete bench with copper fire pit. RIGHT: smooth terracotta wall with recessed planter. CENTER: gravel area. BACKGROUND: glass-walled pavilion, warm interior light spilling out."},
            {"base": "mid-century modern backyard at late afternoon", 
             "view_a": "LEFT: wooden deck with lounge chairs. RIGHT: pool coping with diving board. CENTER: clear water. BACKGROUND: privacy fence with climbing ivy, {bg} sky.", 
             "view_b": "LEFT: pool coping with diving board. RIGHT: wooden deck with lounge chairs. CENTER: clear water. BACKGROUND: glass sliding doors to kitchen, warm indoor lighting visible."},
            {"base": "prohibition-style speakeasy interior", 
             "view_a": "LEFT: brick wall with vintage liquor shelves. RIGHT: dark leather booths with brass lamps. CENTER: worn hardwood walkway. BACKGROUND: polished mahogany bar, {bg} mirror glow.", 
             "view_b": "LEFT: dark leather booths with brass lamps. RIGHT: brick wall with vintage liquor shelves. CENTER: worn hardwood walkway. BACKGROUND: heavy wooden entrance door, dim hallway light."},
            {"base": "contemporary art gallery in daylight", 
             "view_a": "LEFT: white partition with large abstract painting. RIGHT: open sightline to adjacent room. CENTER: polished concrete floor. BACKGROUND: distant sculpture, {bg} natural window light.", 
             "view_b": "LEFT: open sightline to adjacent room. RIGHT: white partition with large abstract painting. CENTER: polished concrete floor. BACKGROUND: gallery entrance doors, soft street daylight visible."},
            {"base": "rustic mountain cabin porch in morning mist", 
             "view_a": "LEFT: wooden rocking chair with wool blanket. RIGHT: stone fireplace with stacked firewood. CENTER: wide plank deck. BACKGROUND: dense pine trees fading into {bg} misty sky.", 
             "view_b": "LEFT: stone fireplace with stacked firewood. RIGHT: wooden rocking chair with wool blanket. CENTER: wide plank deck. BACKGROUND: cabin cedar exterior, warm glow from front window."},
            {"base": "luxury yacht deck at twilight", 
             "view_a": "LEFT: stainless railing with coiled rope. RIGHT: white canopy frame with integrated lighting. CENTER: teak floor. BACKGROUND: calm horizon, fading {bg} gradient, distant marker light.", 
             "view_b": "LEFT: white canopy frame with integrated lighting. RIGHT: stainless railing with coiled rope. CENTER: teak floor. BACKGROUND: outdoor lounge area, warm cabin spill through glass."},
            {"base": "Victorian glass conservatory in daylight", 
             "view_a": "LEFT: curved glass wall with climbing orchids. RIGHT: wrought iron bench surrounded by ferns. CENTER: mosaic tile path. BACKGROUND: tall palm fronds, bright {bg} sky through dome.", 
             "view_b": "LEFT: wrought iron bench surrounded by ferns. RIGHT: curved glass wall with climbing orchids. CENTER: mosaic tile path. BACKGROUND: ornate double glass doors, bright exterior light."},
            {"base": "glass-enclosed urban skybridge at night", 
             "view_a": "LEFT: tempered glass showing corporate interior. RIGHT: brushed steel handrail with LED strip. CENTER: wide walkway. BACKGROUND: opposite tower facade, {bg} city grid.", 
             "view_b": "LEFT: brushed steel handrail with LED strip. RIGHT: tempered glass showing corporate interior. CENTER: wide walkway. BACKGROUND: connecting corridor with brushed metal doors, overhead glow."},
            {"base": "industrial converted loft in overcast daylight", 
             "view_a": "LEFT: exposed brick wall with vintage posters. RIGHT: tall steel-framed windows. CENTER: open hardwood floor. BACKGROUND: bright {bg} sky through glass, distant fire escapes.", 
             "view_b": "LEFT: tall steel-framed windows. RIGHT: exposed brick wall with vintage posters. CENTER: open hardwood floor. BACKGROUND: interior workspace with drafting table, hanging Edison bulbs."}
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
        layout = self.location_layouts.get()
        
        return {
            "char1_prompt": f'{char1.name.capitalize()} is {char1.character_description} '
                            f"wearing {colors['char1']} {self.textures.get()} {self.outfits.get()}.",
            "char2_prompt": f'{char2.name.capitalize()} is {char2.character_description} '
                            f"wearing {colors['char2']} {self.textures.get()} {self.outfits.get()}.",
            "data": {
                "layout": layout,
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
        os.mkdir(dirname)
        print("#"*50,dirname.upper(),"#"*50)
        scene.reset_all_bags()

        result = scene.generate_chars(char1, char2)
        
        # Generate Character Sheets
        CreateCharacterSheet(result["char1_prompt"], f'{dirname}/char1.png', seed=seed)
        CreateCharacterSheet(result["char2_prompt"], f'{dirname}/char2.png', seed=seed)
        
        # Extract coordinated dynamic elements
        d = result["data"]
        layout = d["layout"]
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
        CreateBackground(prompt_a, f'{dirname}/location.png', seed=seed)
        CreateBackground(prompt_b, f'{dirname}/location_reverse.png', seed=seed)
        print("#"*100)
