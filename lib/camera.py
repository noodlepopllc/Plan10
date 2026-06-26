import cv2
import numpy as np
from PIL import Image
from uniface.detection import RetinaFace

from image_analysis import AnalyzeImage
from image_edit import ImageEdit
from image_gen import add_metadata_char, GenerateImage
from util import video_to_img, wait_for_file
import torch, os, sys
from safetensors import safe_open
from image_to_video import GenerateVideo
from compositor import _classify_scene

from config import load_environ

load_environ()
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("WIDTH", "480"))

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

import torch
from safetensors import safe_open

def find_submodule(root, path):
    parts = path.split(".")
    module = root
    for p in parts:
        if p.isdigit():
            module = module[int(p)]
        else:
            module = getattr(module, p)
    return module


def make_lora_hook(A, B, alpha):
    # keep A/B on CPU; we’ll move them lazily to the right device
    A_cpu = A.clone()
    B_cpu = B.clone()

    def hook(module, inputs, output):
        # During tracing / fake tensor passes, output is on 'meta' → skip LoRA
        if getattr(output, "device", None) is not None and output.device.type == "meta":
            return output

        x = inputs[0]

        # Make sure we’re on the same device/dtype as the output
        device = output.device
        dtype = output.dtype

        A_ = A_cpu.to(device=device, dtype=dtype)
        B_ = B_cpu.to(device=device, dtype=dtype)

        # x: (..., in_features)
        # A_: (rank, in_features) → A_.t(): (in_features, rank)
        # B_: (out_features, rank) → B_.t(): (rank, out_features)
        lora_out = (x @ A_.t()) @ B_.t()

        return output + alpha * lora_out

    return hook

def attach_qwen_lora_runtime(pipe, lora_path, alpha=1.0):
    from safetensors import safe_open

    dit = pipe.dit

    with safe_open(lora_path, framework="pt") as f:
        keys = list(f.keys())
        groups = {}
        for k in keys:
            if k.endswith("lora_A.weight"):
                base = k[:-len(".lora_A.weight")]
                groups.setdefault(base, {})["A"] = f.get_tensor(k)
            elif k.endswith("lora_B.weight"):
                base = k[:-len(".lora_B.weight")]
                groups.setdefault(base, {})["B"] = f.get_tensor(k)

        attached = 0
        for base, tensors in groups.items():
            if "A" not in tensors or "B" not in tensors:
                continue

            if base.startswith("transformer."):
                module_path = base.split(".", 1)[1]
            else:
                module_path = base

            try:
                module = find_submodule(pipe.dit, module_path)
            except Exception:
                continue

            hook = make_lora_hook(tensors["A"], tensors["B"], alpha)
            module.register_forward_hook(hook)
            attached += 1

    print(f"[Qwen-LoRA] Attached runtime LoRA to {attached} modules.")
    return attached



class CameraGimbal:

    azimuth = {0: 'front view', 45: 'front-right quarter view', 90: 'right side view', 
                135: 'back-right quarter view', 180: 'back view',  225: 'back-left quarter view', 
                270: 'left side view', 315: ' 	front-left quarter view'}

    elevation = {-30: 'low-angle shot', 0: 'eye-level shot', 30: 'elevated shot', 60: 'high-angle shot' }

    distance = {0.6: 'close-up', 1.0: 'medium shot', 1.8: 'wide shot'}

    def __init__(self, azimuth, elevation, distance):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance

    def get_prompt(self):
        return f"<sks> {CameraGimbal.azimuth[self.azimuth]} {CameraGimbal.elevation[self.elevation]} {CameraGimbal.distance[self.distance]}"

    def generate(self, image, output, width, height, seed):
        from image_edit import ImageEditQwen
        editor = ImageEditQwen()
        attach_qwen_lora_runtime(
            editor.get_pipe(),
            "./loras/qwen-image-edit-2511-multiple-angles-lora.safetensors",
            alpha=1.0,
        )

        prompt = self.get_prompt()
        return editor.generate(prompt, [image], output, width, height, seed)

def ApplyGimbalShot(media="", output="", angle="front", height="eye", distance="medium", seed=-1, static=True):
    AZ_MAP = {"front": 0, "front_right": 45, "right": 90, "back_right": 135, 
              "back": 180, "back_left": 225, "left": 270, "front_left": 315}
    EL_MAP = {"low": -30, "eye": 0, "high": 30, "very_high": 60}
    DIST_MAP = {"closeup": 0.6, "medium": 1.0, "wide": 1.8}

    if not os.path.exists(media):
        raise FileNotFoundError(f"Source image not found: {media}")

    img = video_to_img(media)
    
    # Your existing class handles LoRA attachment + Qwen pipeline
    gimbal = CameraGimbal(AZ_MAP.get(angle, 0), EL_MAP.get(height, 0), DIST_MAP.get(distance, 1.0))
    
    # generate() already saves to disk and returns a dict/status
    result =  gimbal.generate(img, output, img.width, img.height, seed)
    #end_frame = video_to_img(result['output_path'])
    if not static:
        GenerateVideo(prompt='Camera moves to a new field of view', media=[media, result['output_path']], output=output.replace('png','mp4'), 
                  duration_sec=5, width=img.width, height=img.height, seed=seed)
    return result

def ApplyGimbalImage(media="", output="", angle="front", height="eye", distance="medium", seed=-1):
    AZ_MAP = {"front": 0, "front_right": 45, "right": 90, "back_right": 135, 
              "back": 180, "back_left": 225, "left": 270, "front_left": 315}
    EL_MAP = {"low": -30, "eye": 0, "high": 30, "very_high": 60}
    DIST_MAP = {"closeup": 0.6, "medium": 1.0, "wide": 1.8}

    if not os.path.exists(media):
        raise FileNotFoundError(f"Source image not found: {media}")

    img = video_to_img(media)
    
    # Your existing class handles LoRA attachment + Qwen pipeline
    gimbal = CameraGimbal(AZ_MAP.get(angle, 0), EL_MAP.get(height, 0), DIST_MAP.get(distance, 1.0))
    
    # generate() already saves to disk and returns a dict/status
    return gimbal.generate(img, output, img.width, img.height, seed)


def GimbalShotSchema():
    return {
        "type": "function",
        "function": {
            "name": "apply_gimbal_shot",
            "description": "Reframe a character using a validated multi-angle LoRA. Generates a clean keyframe for video interpolation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "media": {"type": "string", "description": "Source image alias or path"},
                    "output": {"type": "string", "description": "Output path for result"},
                    "angle": {
                        "type": "string",
                        "enum": ["front", "front_right", "right", "back_right", "back", "back_left", "left", "front_left"],
                        "description": "Horizontal camera angle"
                    },
                    "height": {
                        "type": "string",
                        "enum": ["low", "eye", "high", "very_high"],
                        "description": "Vertical camera height"
                    },
                    "distance": {
                        "type": "string",
                        "enum": ["closeup", "medium", "wide"],
                        "description": "Camera distance/focal length"
                    },
                    "seed": {"type": "integer", "default": -1}
                },
                "required": ["media", "output", "angle", "height", "distance"]
            }
        }
    }

zoom_prompt = (
    "Enhance the face in the main image using the second image strictly as a texture and feature detail swatch. "
    "Keep the exact framing, camera distance, head size, and shoulder placement from the main image completely unchanged. "
    "Keep the same outfit and pose. "
    "Do not zoom, crop, reframe, or alter the subject's scale. "
    "Only refine micro-details: skin pores, hair strands, eye reflections, and subtle lighting. "
    "Character context: {chars_desc}"
)

if __name__ == '__main__':
    from PIL import Image
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Cinematic Image Pipeline')
    parser.add_argument('-I', '--images', action='append', default=[], help='Input images')
    parser.add_argument('-T', '--target', type=str, default=None, help='Target of the zoom, left, center, right or none')
    parser.add_argument('-S', '--steps', type=float, default=10, help='Percent of the frame to move, max 30')
    parser.add_argument('-C', '--camera-move', type=str, default='zoom', help='type of camera movement zoom, pan-left, pan-right, gimbal')
    parser.add_argument('-E', '--seed', type=int, default=42)
    parser.add_argument('-O', '--output', type=str, default='output.png')
    parser.add_argument('-A', '--azimuth', type=int, default=0, help='45 degree increments 0-315')
    parser.add_argument('-L', '--elevation', type=int, default=0, help='30 degree increments -30 - 60')
    parser.add_argument('-D', '--distance', type=float, default=1.0, help='scale factor for zoom 0.6, 1.0, 1.8')
    parser.add_argument('-M', '--movement', action='store_true')
    args = parser.parse_args()
    steps = 0.10
    status = {}
    if args.steps > 9 and args.steps < 51:
        steps = args.steps / 100.0
    wait_for_file(args.images[0])
    img = video_to_img(args.images[0])
    img2 = None
    if len(args.images) > 1:
        img2 = video_to_img(args.images[1])
    shifted_image = None
    if args.camera_move == 'zoom':
        camera = CameraZoomEngine(steps)
        print(args.output)
        camera.zoom_in(img, character=args.target).save('tmp.png')
        output1 = video_to_img('tmp.png')
        desc = img2.info.get('Description', 'character')
        if desc == 'character':
            desc = add_metadata_char(args.images[1])
        desc = f"{desc}. Preserve adult facial proportions, light cheekbone definition, and subtle jawline contour."
        edit = ImageEdit()
        status = edit.generate(zoom_prompt.format(chars_desc=desc), [output1, img2], args.output, output1.width, output1.height, args.seed)
    elif args.camera_move == 'gimbal':
        camera = CameraGimbal(args.azimuth, args.elevation, args.distance)
        status = camera.generate(img, args.output, img.width, img.height, args.seed)
    else:
        prompt = '''
            Inpaint ONLY the masked region on the {side} edge.
            Preserve all non-masked pixels exactly as they are—do not modify, warp, stretch, or reinterpret them.
            If any subject, face, or object touches the masked boundary, leave it partially cropped exactly as-is.
            Do NOT complete, regenerate, pull back into frame, or re-center any existing elements.
            If the masked region contains only background, extend walls, floor, sky, props, or lighting naturally to match perspective and style.
            Match the original image's color palette, lighting direction, grain, and perspective exactly.
            '''.format(side='left' if 'left' in args.camera_move else 'right')
        camera = CameraMoveEngine(steps)
        if 'left' in args.camera_move:
            print('pan left')
            shifted_image = camera.pan_left(img)
        else:
            print('pan right')
            shifted_image = camera.pan_right(img)
    
    if shifted_image:
        shifted_image.save(f'{args.camera_move}.png')
        edit = ImageEdit()
        status = edit.generate(prompt, [shifted_image], args.output, img.width, img.height, -1)
    if args.movement:
        GenerateVideo(prompt='Camera moves to a new field of view', media=[img, args.output], output=args.output.replace('png','mp4'), 
            duration_sec=5, width=img.width, height=img.height, seed=args.seed)
    print(status)
