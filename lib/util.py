from PIL import Image

def video_to_img(vid, width=832, height=480, resize = False):
    if vid.endswith('.png') or vid.endswith('.jpg'):
        img = Image.open(vid).convert("RGB")
        if resize:
            img = img.resize((width, height))
        return img

    import cv2
    
    cap = cv2.VideoCapture(vid)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 2)  # -2 because some codecs drop the exact last frame
    ret, frame = cap.read()
    cap.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))