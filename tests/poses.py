from glob import glob
import sys
sys.path.append('./lib')

from image_edit import ImageEditQwen, EditImage
from PIL import Image
from pathlib import Path
from image_gen import add_metadata_char
from util import video_to_img

if __name__ == '__main__':
    person = sys.argv[1]
    person_pth = Path(person)
    img = video_to_img(person)
    width = img.width
    height = img.height

    outdir = person_pth.parent
    with ImageEditQwen() as edit:
        poses = glob('./tests/poses/headshots/*.png')
        poses.sort()
        for pose in poses:
            base_pose = Path(pose).stem
            img_pose = Image.open(pose)
            background = 'A nice park in a futuristic metropolitan city'
            #background = 'Plain white background with studio lighting'
            prompt = f"Transform the person in image 1 to match the pose in image 2 (open pose head shot). Keep outfit and hair identical. Medium close-up shot. Background: {background}"
            print(EditImage(prompt=prompt, images=[img, img_pose], output=f'{outdir}/{base_pose}.png', width=width, height=height, seed=42, img_edit=edit))
        poses = glob('./tests/poses/*.png')
        poses.sort()
        for pose in poses:
            base_pose = Path(pose).stem
            
            img_pose = Image.open(pose)
            background = 'A nice park in a futuristic metropolitan city'
            #background = 'Plain white background with studio lighting'
            prompt = f"Transform the person in image 1 to match the pose in image 2. Keep outfit and hair identical. Background: {background}"
            print(EditImage(prompt=prompt, images=[img, img_pose], output=f'{outdir}/{base_pose}.png', width=width, height=height, seed=42, img_edit=edit))

