# templates.py
import os, sys
from plan10.lib.config import load_environ

load_environ()


class CommandBuffer:
    def __init__(self):
        self.identity = []
        self.videos = []
        self.images = []
        self.video_images = []
        self.ots = []
        self.ots_images = []
        self.closeup = []
        self.closeup_images = []
        self.medium = []
        self.medium_images = []
        self.full = []

    def dump(self, mode="all"):
        mode = mode.lower()
        for c in self.identity:
            print(c)
        for c in self.images:
            print(c)
        if mode in ("images"):
            for c in self.video_images:
                print(c)
            for c in self.dialog_images:
                print(c)
            for c in self.closeup_images:
                print(c)
            for c in self.medium_images:
                print(c)
        if mode in ("all", "videos"):
            for c in self.video_images:
                print(c)
            for c in self.videos:
                print(c)
        if mode in ("all", "dialog", "ots"):
            for c in self.ots_images:
                print(c)
            for c in self.ots:
                print(c)
        if mode in ("all", "dialog", "closeup"):
            for c in self.closeup_images:
                print(c)
            for c in self.closeup:
                print(c)
        if mode in ("all", "dialog", "medium"):
            for c in self.medium_images:
                print(c)
            for c in self.medium:
                print(c)
        if mode in ("full"):
            for c in self.dialog_images:
                print(c)
            for c in self.full:
                print(c)


class Templates:
    def __init__(self):
        self.WIDTH  = int(os.environ.get("WIDTH", "832"))
        self.HEIGHT = int(os.environ.get("HEIGHT", "480"))
        self.SEED   = int(os.environ.get("SEED", "123456"))

        # one buffer for everything
        self.buffer = CommandBuffer()

    # ---------------------------------------------------------
    # CHARACTER SHEET + VOICE
    # ---------------------------------------------------------

    def character_sheet(self, alias, description):
        self.buffer.identity.append(f"""
>> ALIAS: {alias}
create a character sheet of {description}, Seed: {self.SEED}
""")

    def voice_design(self, alias, voice_desc):
        self.buffer.identity.append(f"""
>> ALIAS: {alias}_VOICE
design a voice for {voice_desc}
""")

    # ---------------------------------------------------------
    # BACKGROUND
    # ---------------------------------------------------------

    def background(self, alias, architecture, definition, anchored):
        self.buffer.images.append(f"""
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

    def backdrop(self, zone_alias, char_alias, shot_type):
        self.buffer.images.append(f"""
>> ALIAS: {char_alias}_{shot_type.upper()}_BACKDROP
composite_background {zone_alias}_BACKGROUND asset, shot_type: {shot_type}, Width: {self.WIDTH}, Height: {self.HEIGHT}, Seed: {self.SEED}
""")

    # ---------------------------------------------------------
    # ACTION
    # ---------------------------------------------------------

    def action_medium(self, alias, zone_alias, char_alias, prompt, arc=None):
        arc_text = f" Motion arc: {arc}" if arc else ""
        self.buffer.video_images.append(f"""
>> ALIAS: {alias}
composite_scene {zone_alias} asset and {char_alias} ,
shot_type: "medium",
prompt: "{prompt}",
Width: {self.WIDTH}, Height: {self.HEIGHT}, Seed: {self.SEED}
""")

    def action_wide(self, alias, zone_alias, char_assets, prompt, arc=None):
        arc_text = f" Motion arc: {arc}" if arc else ""
        self.buffer.video_images.append(f"""
>> ALIAS: {alias}
composite_scene {zone_alias} asset and {char_assets},
shot_type: "two_shot",
prompt: "{prompt}",
Width: {self.WIDTH}, Height: {self.HEIGHT}, Seed: {self.SEED}
""")

    def action_video(self, alias, image_alias, motion, duration=5):
        self.buffer.videos.append(f"""
>> ALIAS: {alias}
image_to_video {image_alias} asset, {motion},
Width: {self.WIDTH}, Height: {self.HEIGHT}, Duration: {duration}, Seed: {self.SEED}
""")

    # ---------------------------------------------------------
    # DIALOG
    # ---------------------------------------------------------

    def dialog_closeup(self, alias, zone_alias, char_alias, prompt):
        self.buffer.closeup_images.append(f"""
>> ALIAS: {alias}
composite_scene {zone_alias} asset and {char_alias} asset,
shot_type: "closeup",
prompt: "{prompt}",
Width: {self.WIDTH}, Height: {self.HEIGHT}, Seed: {self.SEED}
""")

    def dialog_ots(self, alias, zone_alias, char_assets, prompt):
        self.buffer.ots_images.append(f"""
>> ALIAS: {alias}
composite_scene {zone_alias} asset and {char_assets},
shot_type: "ots",
prompt: "{prompt}",
Width: {self.WIDTH}, Height: {self.HEIGHT}, Seed: {self.SEED}
""")

    def dialog_medium(self, alias, zone_alias, char_alias, prompt):
        self.buffer.medium_images.append(f"""
>> ALIAS: {alias}
composite_scene {zone_alias} asset and {char_alias} asset,
shot_type: "medium",
prompt: "{prompt}",
Width: {self.WIDTH}, Height: {self.HEIGHT}, Seed: {self.SEED}
""")

    def dialog_motion(self, alias, base_alias, motion_prompt, duration=2):
        self.buffer.dialog.append(f"""
>> ALIAS: {alias}
image_to_video {base_alias}_medium asset, "{motion_prompt}",
Width: {self.WIDTH}, Height: {self.HEIGHT}, Duration: {duration}, Seed: {self.SEED}
""")

    def dialog_final(self, alias, base_alias, voice_alias, text, prompt=""):
        final_alias = alias + '_full' if prompt else alias
        final = f"""
>> ALIAS: {final_alias}
dialog_to_video media={base_alias} asset
audio={voice_alias} asset
text="{text}",
prompt="{prompt}",
Width: {self.WIDTH}, Height: {self.HEIGHT}, Seed: {self.SEED}
"""
        if prompt:
            self.buffer.full.append(final)
        elif "CLOSEUP" in base_alias:
            self.buffer.closeup.append(final)
        elif "MEDIUM" in base_alias:
            self.buffer.medium.append(final)
        else:
            self.buffer.ots.append(final)

    # ---------------------------------------------------------
    # DIALOG MOTION PROMPT
    # ---------------------------------------------------------

    def dialog_motion_prompt(self, speaker, facial, head):
        return (
        f"{speaker}, neutral expression, "
        f"soft breathing motion in chest and shoulders, "
        f"eyes performing tiny natural micro‑saccades, "
        f"head stable and still, "
        f"lips fully closed with no speech motion, "
        f"jaw relaxed and unmoving, "
        f"maintain facial expression {facial}, "
        f"head gesture {head}, "
        f"no large body motion"
    )

