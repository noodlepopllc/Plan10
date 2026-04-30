from PIL import Image

from PIL import Image

def video_to_img(vid, width=832, height=480, resize=False, getlast=True):
    # Handle images (resize optional)
    if vid.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(vid).convert("RGB")
        if resize:
            return img.resize((width, height), Image.Resampling.LANCZOS)
        return img

    # Handle videos (ALWAYS native resolution)
    import cv2
    cap = cv2.VideoCapture(vid)
    try:
        # Enforce dimension match when resize is explicitly requested
        if resize:
            v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if v_w != width or v_h != height:
                raise ValueError(
                    f"Dimension mismatch: requested {width}x{height}, "
                    f"but video is {v_w}x{v_h}. Video resizing is disabled."
                )

        if getlast:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 2))
                ret, frame = cap.read()
        else:
            ret, frame = cap.read()

        if not ret or frame is None:
            raise ValueError(f"Failed to extract frame from: {vid}")

        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()