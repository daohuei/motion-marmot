import time
import typer
from visdom import Visdom
from motion_marmot.advanced_motion_filter import AdvancedMotionFilter, BoundingBox
from motion_marmot.utils.video_utils import extract_video, frame_convert, frame_resize


class AMFVisdom():
    DEFAULT_CONFIG = {
        "bounding_box_threshold": 200
    }

    def __init__(self, video):
        self.viz = Visdom(port=8090)
        self.viz.close(env="visdom_playground")
        self.viz_config = self.DEFAULT_CONFIG
        self.ctl = None
        self.frame_list, video_meta = extract_video(video)
        self.frame_fps = video_meta['fps']
        self.amf = AdvancedMotionFilter(
            ssc_model='model/scene_knn_model',
            frame_width=video_meta['width'],
            frame_height=video_meta['height']
        )
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
                    "value": self.amf.amf_history_variance,
                },
                {
                    "type": "number",
                    "name": "History Variance Threshold",
                    "value": self.amf.amf_variance_threshold,
                },
                {
                    "type": "number",
                    "name": "History Variance Sample Amount",
                    "value": self.amf.amf_variance_sample_amount,
                },
                {
                    "type": "checkbox",
                    "name": "Large Background Movement",
                    "value": self.amf.amf_drop_large_bg_motion,
                },
                {
                    "type": "checkbox",
                    "name": "Dynamic Bounding Box",
                    "value": self.amf.amf_dynamic_bbx,
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
                self.amf.amf_history_variance = context.get('value')
            elif property_name == 'History Variance Threshold':
                self.amf.amf_variance_threshold = int(context.get('value'))
            elif property_name == 'History Variance Sample Amount':
                self.amf.amf_variance_sample_amount = int(context.get('value'))
            elif property_name == 'Large Background Movement':
                self.amf.amf_drop_large_bg_motion = context.get('value')
            elif property_name == 'Dynamic Bounding Box':
                self.amf.amf_dynamic_bbx = context.get('value')
            self.ctl.panel = update("Control Panel")

        self.ctl = VisdomControlPanel(
            self.viz,
            update,
            trigger,
            "Control Panel"
        )

    def motion_detection(self, frame):
        mask = self.amf.apply(frame.copy())
        display_frame = frame.copy()
        motion_bbxes = self.amf.detect_motion(
            mask,
            self.viz_config.get('bounding_box_threshold', 200)
        )
        for bbx in motion_bbxes:
            box = BoundingBox(*bbx)
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
def run_amf(video: str):
    amf_visdom = AMFVisdom(video)
    amf_visdom.start()


def main():
    """Main program"""
    app()


if __name__ == "__main__":
    main()
