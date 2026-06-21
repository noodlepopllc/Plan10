from PIL import Image
import os, time

def wait_for_file(path: str, timeout: float = 30.0, min_size: int = 1024, stable_for: float = 1.5):
    """Wait until file exists, has reasonable size, AND stops growing."""
    start = time.time()
    last_size = -1
    stable_start = None
    
    while time.time() - start < timeout:
        if not os.path.exists(path):
            time.sleep(0.1)
            continue
        
        current_size = os.path.getsize(path)
        
        if current_size < min_size:
            time.sleep(0.1)
            continue
        
        # File meets minimum size, now check stability
        if current_size != last_size:
            # Size changed, reset stability timer
            last_size = current_size
            stable_start = time.time()
            time.sleep(0.1)
            continue
        
        # Size unchanged, check if stable long enough
        if stable_start and (time.time() - stable_start) >= stable_for:
            return True
        
        time.sleep(0.1)
    
    size = os.path.getsize(path) if os.path.exists(path) else 'N/A'
    print(f"⚠️  File {path} not ready after {timeout}s (size: {size})")
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

