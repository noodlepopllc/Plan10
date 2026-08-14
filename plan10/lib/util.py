from PIL import Image
import os, time, cv2, gc, torch
import numpy as np

import librosa
import soundfile as sf

def fix_minimax_audio(input_path, output_path, target_sr=44100, target_duration=11.0):
    # 1. Load, convert to mono, and automatically resample to 44.1 kHz
    # (Setting sr=target_sr forces librosa to resample upon loading)
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    
    # 2. Peak normalize to -3 dB to prevent gain clipping and distortion
    # We find the max amplitude, scale it to 1.0, and multiply by 10^(-3/20) ~ 0.707
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = (y / max_val) * 0.707

    # 3. Pad with digital silence to pass the 10-second requirement
    current_samples = len(y)
    required_samples = int(target_duration * sr)
    
    if current_samples < required_samples:
        padding_needed = required_samples - current_samples
        silence = np.zeros(padding_needed)
        # Append the silence to the tail end of the audio array
        y = np.concatenate([y, silence])
        print(f"Padded track with {padding_needed / sr:.2f} seconds of silence.")
    
    # 4. Export as a standard 16-bit PCM WAV file
    # 'PCM_16' forces the 24-bit downsample to avoid bit-depth parsing bugs
    sf.write(output_path, y, sr, subtype='PCM_16')
    print(f"Successfully saved fixed file to: {output_path}")


def wait_for_file(path: str, timeout: float = 60.0, min_size: int = 1024, stable_for: float = 1.5):
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
    cap = cv2.VideoCapture(vid)
    try:
        # Enforce dimension match when resize is explicitly requested

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

def cleanup():
    """Clear VRAM after model usage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def extract_frame(media_path, width, height, output_path=None):
    """Extracts a frame from video or loads image, returns PIL Image and path."""
    media_path = str(media_path)
    ext = media_path.split('.')[-1].lower()
    
    if ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
        frame = video_to_img(media_path, width, height, True, True)
        if output_path:
            frame.save(str(output_path))
            return frame, str(output_path)
        return frame, media_path
    else:
        img = Image.open(media_path)
        return img, media_path

def resize_image(img, max_dim=640):
    """Resizes image keeping aspect ratio so the largest side is max_dim."""
    if isinstance(img, Image.Image):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0
    
    scale = max_dim / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

