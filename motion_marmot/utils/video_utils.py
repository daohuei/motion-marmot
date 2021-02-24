import cv2
import numpy as np


def extract_video(video: str):
    cap = cv2.VideoCapture(video)
    video_meta = {}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_meta['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_meta['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_meta['fps'] = int(cap.get(cv2.CAP_PROP_FPS))

    video_frames = np.empty(
        (
            frame_count,
            video_meta['height'],
            video_meta['width'],
            3
        ),
        np.dtype('uint8')
    )
    i = 0
    ret = True
    while (i < frame_count and ret):
        ret, video_frames[i] = cap.read()
        i += 1
    cap.release()
    return video_frames, video_meta


def frame_convert(frame):
    display_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return display_frame_rgb.transpose((2, 0, 1))


def frame_resize(frame):
    return cv2.resize(frame.copy(), (320, 180)).copy()
