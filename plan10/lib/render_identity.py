import os, sys, json
from plan10.lib.config import load_environ

load_environ()

def normalize(name: str) -> str:
    name = name.replace(' ', '_').replace('/', '_')
    return ''.join([x for x in name.upper() if x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'])

class CommandBuffer:
    def __init__(self):
        self.identity = []
        self.backgrounds = []
        self.voices = []

    def dump(self, mode="all"):
        mode = mode.lower()
        for c in self.identity:
            print(c)
        for c in self.backgrounds:
            print(c)
        for c in self.voices:
            print(c)

class Templates:
    def __init__(self):
        self.SEED = int(os.environ.get("SEED", "123456"))
        self.buffer = CommandBuffer()

    def character_sheet(self, alias, description):
        self.buffer.identity.append(f"""
>> ALIAS: {alias}
create a character sheet of {description}, Seed: {self.SEED}
""")

    def voice_design(self, alias, voice_desc):
        self.buffer.voices.append(f"""
>> ALIAS: {alias}_VOICE
design a voice for {voice_desc}
""")

    def background(self, alias, architecture, definition, anchored):
        self.buffer.backgrounds.append(f"""
>> ALIAS: {alias}_BACKGROUND
create_background cinematic composition with tighter framing focused on the primary functional area,
minimize negative space at the frame edges,
center the back wall as the dominant architectural surface,
include only the objects positioned against or near the back wall,
preserve natural perspective and room geometry,
Architecture: {architecture},
Description: {definition},
Anchored objects: {anchored},
Seed: {self.SEED}
""")

def get_identity(assets, T):
    for bio in assets['biographies']:
        name = bio['name']
        alias = f"CHAR_{normalize(name)}"

        description = (
            f"{bio['gender']}, Age: {bio['age']}, "
            f"{bio['race']}/{bio['ethnicity_species']}, "
            f"{bio['appearance']},{bio['hair']}, {bio['clothing']}"
        )

        T.character_sheet(alias, description)
        T.voice_design(alias, ",".join(description.split(",")[:3]))

def get_backgrounds(assets, T):  # Fixed: added closing parenthesis
    for location in assets['locations']:
        location_name = location['name']
        architecture = location['architectural_shell']
        
        for zone in location['zones']:
            zone_name = zone['zone_name']
            zone_def = zone['zone_definition']
            elements = zone.get('visible_background_elements', [])
            
            zone_key = f"{location_name}_{zone_name}".replace('"','').replace(' ', '_').replace('/','_').upper()

            T.background(
                zone_key,
                architecture,
                zone_def,
                ', '.join(elements)
            )

def main():
    from pathlib import Path
    
    registry_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None  # Optional output file

    with open(registry_path, 'r') as ass:
        assets = json.load(ass)

    T = Templates()

    get_identity(assets, T)
    get_backgrounds(assets, T)

    if output_path:
        # Write to file for bot consumption
        with open(output_path, 'w') as f:
            for c in T.buffer.identity:
                f.write(c)
            for c in T.buffer.backgrounds:
                f.write(c)
            for c in T.buffer.voices:
                f.write(c)
        print(f"Wrote {output_path}")
    else:
        # Print to stdout
        T.buffer.dump()

if __name__ == "__main__":
    main()