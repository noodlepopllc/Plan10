import cv2
import numpy as np
from PIL import Image
from uniface.detection import RetinaFace

import numpy as np
from PIL import Image
from image_analysis import AnalyzeImage

from image_edit import ImageEdit

class CameraMoveEngine:
    def __init__(self, step=0.10):
        self.step = step  # percent of dimension per move

    def pan_left(self, img: Image.Image):
        w, h = img.size
        shift = int(self.step * w)

        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        arr = np.array(canvas)
        src = np.array(img)

        # content shifts RIGHT → empty region on LEFT
        arr[:, shift:] = src[:, :w-shift]

        return Image.fromarray(arr)

    def pan_right(self, img: Image.Image):
        w, h = img.size
        shift = int(self.step * w)

        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        arr = np.array(canvas)
        src = np.array(img)

        # content shifts LEFT → empty region on RIGHT
        arr[:, :w-shift] = src[:, shift:]

        return Image.fromarray(arr)

class CameraZoomEngine:
    def __init__(self, step=0.10):
        self.detector = RetinaFace()
        self.step = step

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def pil_to_cv2(pil_img):
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(cv_img):
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    # -----------------------------
    # Face selection logic
    # -----------------------------
    def pick_face_center(self, cv_img, character=None):
        """
        character: None, "left", "right", "center"
        Returns normalized (cx, cy)
        """
        h, w = cv_img.shape[:2]
        faces = self.detector.detect(cv_img)

        # ----------------------------------------
        # CASE 1 — No faces → use image center
        # ----------------------------------------
        if not faces:
            return 0.5, 0.5

        # ----------------------------------------
        # CASE 2 — User explicitly specifies character
        # ----------------------------------------
        if character in ("left", "right", "center"):
            # compute centers for all faces
            centers = []
            for f in faces:
                x1, y1, x2, y2 = f.bbox
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                centers.append((cx, cy))

            if character == "left":
                return min(centers, key=lambda c: c[0])
            if character == "right":
                return max(centers, key=lambda c: c[0])
            if character == "center":
                return min(centers, key=lambda c: abs(c[0] - 0.5))

        # ----------------------------------------
        # CASE 3 — No character specified → pick closest face
        # (largest bounding box = closest to camera)
        # ----------------------------------------
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        return cx, cy

    # -----------------------------
    # Zoom step
    # -----------------------------
    def zoom_step(self, cv_img, cx, cy):
        h, w = cv_img.shape[:2]

        z = 1.0 + self.step  # always zoom-in
        new_w = int(w / z)
        new_h = int(h / z)

        cx_px = cx * w
        cy_px = cy * h

        x1 = int(cx_px - new_w / 2)
        y1 = int(cy_px - new_h / 2)

        x1 = max(0, min(x1, w - new_w))
        y1 = max(0, min(y1, h - new_h))

        crop = cv_img[y1:y1+new_h, x1:x1+new_w]
        out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
        return out

    # -----------------------------
    # Public API
    # -----------------------------
    def zoom_in(self, pil_img, character=None):
        """
        character: None, "left", "right", "center"
        """
        cv_img = self.pil_to_cv2(pil_img)

        cx, cy = self.pick_face_center(cv_img, character=character)
        zoomed = self.zoom_step(cv_img, cx, cy)

        return self.cv2_to_pil(zoomed)

prompt = '''
    Extend the scene naturally into the masked region only.
    Do not modify the existing character, pose, face, skin texture, clothing, lighting, or framing.
    Preserve all visible pixels exactly as they are.
    Only generate new background/environment details inside the masked area.
    Do not move, rotate, resize, or re-center the character.
    Do not add tattoos, markings, or new features.
    Match the style, lighting, and perspective of the original image.
    '''

if __name__ == '__main__':
    from PIL import Image
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Cinematic Image Pipeline')
    parser.add_argument('-I', '--image', type=str, default='', help='Input images')
    parser.add_argument('-T', '--target', type=str, default=None, help='Target of the zoom, left, center, right or none')
    parser.add_argument('-S', '--steps', type=float, default=10, help='Percent of the frame to move, max 30')
    parser.add_argument('-C', '--camera-move', type=str, default='zoom', help='type of camera movement zoom, pan-left, pan-right')
    parser.add_argument('-E', '--seed', type=int, default=42)
    parser.add_argument('-O', '--output', type=str, default='output.png')
    args = parser.parse_args()
    steps = 0.10
    if args.steps > 9 and args.steps < 31:
        steps = args.steps / 100.0
    img = Image.open(args.image)
    shifted_image = None
    if args.camera_move == 'zoom':
        camera = CameraZoomEngine(steps)
        camera.zoom_in(img, character=args.target).save(args.output)
    else:
        camera = CameraMoveEngine(steps)
        if 'left' in args.camera_move:
            shifted_image = camera2.pan_left(img)
        else:
            shifted_image = camera2.pan_left(img)
    
    if shifted_image:
        shifted_image.save(args.output)
        edit = ImageEdit()
        status = edit.generate(prompt, [shifted_image], args.output, img.width, img.height, -1)
