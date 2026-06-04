from PIL import Image
import os, time

def wait_for_file(path: str, timeout: float = 5.0, min_size: int = 1024):
    """Wait until file exists and has reasonable size"""
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(path) and os.path.getsize(path) >= min_size:
            return True
        time.sleep(0.05)
    print(f"⚠️  File {path} not ready after {timeout}s (size: {os.path.getsize(path) if os.path.exists(path) else 'N/A'})")
    return False

def video_to_img(vid, width=832, height=480, resize=False, getlast=True):
    # Handle images (resize optional)
    wait_for_file(vid)
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

# ---------------------------------------------------------
# SENTENCE SEGMENTATION (SpaCy sentencizer)
# ---------------------------------------------------------

from spacy_download import load_spacy
NLP = load_spacy("en_core_web_sm", exclude=["parser", "tagger"])
NLP.add_pipe("sentencizer")

def segment_sentences(text: str):
    """Return a clean list of sentences using SpaCy's sentencizer."""
    doc = NLP(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]
