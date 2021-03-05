import typer
import time
import cv2
from visdom import Visdom
from motion_marmot.advanced_motion_filter import AdvancedMotionFilter, BoundingBox, MotionMaskMetadata
from motion_marmot.utils.video_utils import extract_video, frame_convert, frame_resize


class VisdomPlayground():
    DEFAULT_CONFIG = {
        "bounding_box_threshold": 200,
        "variance": False,
        "variance_threshold": 100,
        "variance_sample_amount": 5,
        "large_bg_movement": False,
        "dynamic_bbx": False
    }

    def __init__(self, video):
        self.viz = Visdom(port=8090)
        self.viz.close(env="visdom_playground")
        self.viz_config = self.DEFAULT_CONFIG
        self.ctl = None
        self.frame_list, video_meta = extract_video(video)
        self.frame_width = video_meta['width']
        self.frame_height = video_meta['height']
        self.frame_fps = video_meta['fps']
        self.amf = AdvancedMotionFilter('model/scene_knn_model')
        self.init_control_panel()

    def init_control_panel(self):
        def update(name):
            return self.viz.properties([
                {
                    "type": "number",
                    "name": "Bounding Box Threshold",
                    "value": self.viz_config.get('bounding_box_threshold', 200),
                },
                {
                    "type": "checkbox",
                    "name": "History Variance",
                    "value": self.viz_config.get('variance', False),
                },
                {
                    "type": "number",
                    "name": "History Variance Threshold",
                    "value": self.viz_config.get('variance_threshold', 100),
                },
                {
                    "type": "number",
                    "name": "History Variance Sample Amount",
                    "value": self.viz_config.get('variance_sample_amount', 5),
                },
                {
                    "type": "checkbox",
                    "name": "Large Background Movement",
                    "value": self.viz_config.get('large_bg_movement', False),
                },
                {
                    "type": "checkbox",
                    "name": "Dynamic Bounding Box",
                    "value": self.viz_config.get('dynamic_bbx', False),
                }
            ], win=name, env="visdom_playground")

        def trigger(context):
            if context["event_type"] != "PropertyUpdate":
                return
            if context["target"] != self.ctl.panel:
                return
            property_name = context \
                .get('pane_data') \
                .get('content')[context.get('propertyId')] \
                .get('name')
            if property_name == 'Bounding Box Threshold':
                self.viz_config['bounding_box_threshold'] = int(context.get('value'))
            elif property_name == 'History Variance':
                self.viz_config['variance'] = context.get('value')
            elif property_name == 'History Variance Threshold':
                self.viz_config['variance_threshold'] = int(context.get('value'))
            elif property_name == 'History Variance Sample Amount':
                self.viz_config['variance_sample_amount'] = int(context.get('value'))
            elif property_name == 'Large Background Movement':
                self.viz_config['large_bg_movement'] = context.get('value')
            elif property_name == 'Dynamic Bounding Box':
                self.viz_config['dynamic_bbx'] = context.get('value')
            self.ctl.panel = update("Control Panel")

        self.ctl = VisdomControlPanel(
            self.viz,
            update,
            trigger,
            "Control Panel"
        )

    def motion_detection(self, frame):
        mask = self.amf.mog2_mf.apply(frame.copy())
        display_frame = frame.copy()
        contours = self.amf.calculate_contours(mask)
        mask_area = MotionMaskMetadata(contours)
        frame_scene = self.amf.ssc.predict(
            mask_area.avg,
            mask_area.std,
            self.frame_width,
            self.frame_height
        )
        dynamic_bbx_thresh = mask_area.avg + mask_area.std
        variance = self.amf.calculate_variance(mask_area.std)
        for contour in contours:
            if not self.amf.mog2_is_detected(
                contour=contour,
                scene=frame_scene,
                dynamic_bbx_thresh=dynamic_bbx_thresh,
                variance=variance,
                bounding_box_threshold=self.viz_config.get('bounding_box_threshold'),
                history_variance=self.viz_config.get('variance'),
                variance_threshold=self.viz_config.get('variance_threshold'),
                variance_sample_amount=self.viz_config.get('variance_sample_amount'),
                large_bg_movement=self.viz_config.get('large_bg_movement'),
                dynamic_bbx=self.viz_config.get('dynamic_bbx')
            ):
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


class VisdomControlPanel():
    def __init__(self, viz, update_callback, trigger_callback, name):
        self.panel = None
        self.viz = viz
        self.update = update_callback
        self.trigger = trigger_callback
        self.name = name
        self.panel = self.update(self.name)
        self.viz.register_event_handler(
            self.trigger, self.panel
        )


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
