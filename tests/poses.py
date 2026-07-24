from glob import glob
import sys

from plan10.lib.config import load_config
load_config()

from plan10.lib.image_edit import ImageEditQwen, EditImage
from plan10.lib.image_gen import CreateBackground
from PIL import Image
from pathlib import Path
from plan10.lib.image_gen import add_metadata_char
from plan10.lib.util import video_to_img



WIDTH = int(os.environ.get("WIDTH","480"))
HEIGHT = int(os.environ.get("HEIGHT","832"))

def main():
    person = sys.argv[1]
    person_pth = Path(person)
    img = video_to_img(person)
    width = WIDTH
    height = HEIGHT 

    outdir = person_pth.parent

    background = 'A nice park in a futuristic metropolitan city'
    #background = 'Plain white background with studio lighting'

    CreateBackground(background, output=f'{outdir}/background.png')
    background_img = video_to_img(f'{outdir}/background.png')

    with ImageEditQwen() as edit:
        poses = glob('./tests/poses/headshots/*.png')
        poses.sort()
        for pose in poses:
            base_pose = Path(pose).stem
            img_pose = Image.open(pose)
            prompt = f"Transform the person in image 2 to match the pose in image 3 (open pose head shot). Keep outfit and hair identical. Medium close-up shot. Use the background: {background} from image 1"
            print(EditImage(prompt=prompt, images=[background_img, img, img_pose], output=f'{outdir}/{person_pth.stem}_{base_pose}.png', width=width, height=height, seed=42, img_edit=edit))
        poses = glob('./tests/poses/*.png')
        poses.sort()
        for pose in poses:
            base_pose = Path(pose).stem
            
            img_pose = Image.open(pose)
            prompt = f"Transform the person in image 2 to match the pose in image 3. Keep outfit and hair identical. Use the background: {background} from image 1"
            print(EditImage(prompt=prompt, images=[background_img, img, img_pose], output=f'{outdir}/{person_pth.stem}_{base_pose}.png', width=width, height=height, seed=42, img_edit=edit))

if __name__ == '__main__':
    main()


