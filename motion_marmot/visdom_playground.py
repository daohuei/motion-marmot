import typer
import time
import cv2
import numpy as np
from visdom import Visdom
from advanced_motion_filter import AdvancedMotionFilter, BoundingBox, MaskArea


def frame_convert(frame):
    display_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return display_frame_rgb.transpose((2, 0, 1))


def frame_resize(frame):
    return cv2.resize(frame.copy(), (320, 180)).copy()


class VisdomPlayground():
    def __init__(self, video):
        self.viz = Visdom(port=8090)
        self.viz.close(env="visdom_playground")
        self.frame_list = self.init_frame(video)
        self.amf = AdvancedMotionFilter('model/scene_knn_model')

    def init_frame(self, video):
        cap = cv2.VideoCapture(video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

        video_frames = np.empty(
            (
                frame_count,
                self.frame_height,
                self.frame_width,
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
        return video_frames

    def motion_detection(self, frame):
        mask = self.amf.mog2_mf.apply(frame.copy())
        display_frame = frame.copy()
        contours = self.amf.calculate_contours(mask)
        mask_area = MaskArea(contours)
        frame_scene = self.amf.ssc.predict(
            mask_area.avg,
            mask_area.std,
            self.frame_width,
            self.frame_height
        )
        for contour in contours:
            if not self.amf.mog2_is_detected(contour, frame_scene):
                continue
            box = BoundingBox(*cv2.boundingRect(contour))
            self.amf.draw_detection_box(box, display_frame)
        self.viz.image(
            mask,
            win="mask_window",
            opts=dict(
                width=320,
                height=250,
            ),
            env="visdom_playground"
        )
        return display_frame

    def stream_video(self):
        for frame in self.frame_list:
            resized_frame = frame_resize(frame.copy())
            display_frame = self.motion_detection(resized_frame)
            disp_image = frame_convert(display_frame)
            self.viz.image(
                disp_image,
                win="video_window",
                opts=dict(
                    width=320,
                    height=250,
                ),
                env="visdom_playground"
            )

            time.sleep(1.0/self.frame_fps)

    def start(self):
        print("Visdom Playground Activating")
        while True:
            self.stream_video()


app = typer.Typer()


@app.command()
def run(video: str):
    vp = VisdomPlayground(video)
    vp.start()


def main():
    """Main program"""
    app()


if __name__ == "__main__":
    main()
