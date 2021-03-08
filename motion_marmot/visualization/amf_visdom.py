import os
import time
import typer
import cv2
import numpy as np
from csv import writer
from visdom import Visdom
from motion_marmot.advanced_motion_filter import AdvancedMotionFilter, BoundingBox, FD
from motion_marmot.utils.video_utils import extract_video, frame_convert, frame_resize


class AMFVisdom():
    DEFAULT_CONFIG = {
        "bounding_box_threshold": 200
    }

    def __init__(self, video):
        self.viz = Visdom(port=8090)
        self.viz_config = self.DEFAULT_CONFIG
        self.ctl = None
        self.frame_list, video_meta = extract_video(video)
        self.frame_fps = video_meta['fps']
        self.amf = AdvancedMotionFilter(
            ssc_model='model/scene_knn_model',
            frame_width=video_meta['width'],
            frame_height=video_meta['height']
        )

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
            ], win=name, env="amf")

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
            env="amf"
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
                env="amf"
            )

            time.sleep(1.0/self.frame_fps)

    def start(self):
        print("Visdom Playground Activating")
        self.viz.close(env="amf")
        self.init_control_panel()
        while True:
            self.stream_video()


class AMFParamVisdom(AMFVisdom):
    def __init__(self, video):

        super(AMFParamVisdom, self).__init__(video)

        self.mog2 = cv2.createBackgroundSubtractorMOG2()
        self.fd = FD()

        # Init the panel to get the frame by frame number
        self.get_frame_panel = None
        self.frame_number = 0

        # For labeling the scene of SSC
        self.label_panel = None
        self.label_list = np.zeros(len(self.frame_list)-1)
        self.label_start_index = 0
        self.label_end_index = 0
        self.label_data_dir = ''
        self.label_class_value = 0

        # Basic features extracted from the MOG2 mask of each frame
        self.mask_list = []
        self.total_mask_list = []
        self.avg_mask_list = []
        self.std_mask_list = []

        # FD features
        self.fd_mask_list = []

        # Contours Count
        self.mog2_contour_count_list = []
        self.fd_contour_count_list = []

        # Variance of standard deviation according to the history frames
        self.variance_list = []

    def apply_mog2(self, frame):
        mask = self.amf.apply(frame)
        return mask

    def apply_fd(self, frame):
        mask = self.fd.apply(frame)
        return mask

    def init_viz(self):
        self.init_get_frame_panel()
        self.init_label_panel()

    def init_get_frame_panel(self):
        def update(name):
            return self.viz.properties([
                {
                    "type": "number",
                    "name": "Frame Number",
                    "value": self.frame_number,
                },
                {
                    "type": "number",
                    "name": "Bounding Box Threshold",
                    "value": self.viz_config.get('bounding_box_threshold', 200)
                }
            ], win=name, env="amf_params")

        def trigger(context):
            if context["event_type"] != "PropertyUpdate":
                return
            if context["target"] != self.get_frame_panel.panel:
                return
            property_name = context \
                .get('pane_data') \
                .get('content')[context.get('propertyId')] \
                .get('name')
            if property_name == 'Frame Number':
                self.frame_number = int(context.get('value'))
            elif property_name == 'Bounding Box Threshold':
                self.viz_config['bounding_box_threshold'] = int(context.get('value'))

            resized_frame = frame_resize(self.frame_list[self.frame_number].copy())

            mog2_disp_mask = self.mask_list[self.frame_number]
            mog2_motion_bbxs = self.amf.detect_motion(
                mog2_disp_mask, self.viz_config.get('bounding_box_threshold', 200)
            )

            fd_disp_mask = self.fd_mask_list[self.frame_number]
            fd_motion_bbxs = self.amf.detect_motion(
                fd_disp_mask, self.viz_config.get('bounding_box_threshold', 200)
            )

            for bbx in mog2_motion_bbxs:
                self.amf.draw_detection_box(
                    BoundingBox(*bbx),
                    resized_frame,
                    (0, 255, 0)
                )

            for bbx in fd_motion_bbxs:
                self.amf.draw_detection_box(
                    BoundingBox(*bbx),
                    resized_frame,
                    (255, 0, 0)
                )

            disp_image = frame_convert(resized_frame)
            self.show_frame_images(
                disp_image,
                mog2_disp_mask,
                fd_disp_mask,
                self.frame_number
            )
            self.get_frame_panel.panel = update("Get Frame Panel")

        self.get_frame_panel = VisdomControlPanel(
            self.viz,
            update,
            trigger,
            "Get Frame Panel"
        )

    def init_label_panel(self):
        def update(name):
            return self.viz.properties([
                {
                    "type": "number",
                    "name": "Start Index",
                    "value": self.label_start_index,
                },
                {
                    "type": "number",
                    "name": "End Index",
                    "value": self.label_end_index,
                },
                {
                    "type": "number",
                    "name": "Label Class",
                    "value": self.label_class_value,
                },
                {
                    "type": "text",
                    "name": "Label Data Directory Folder Name",
                    "value": self.label_data_dir
                },
                {
                    "type": "button",
                    "name": "Store Label Value",
                    "value": "label it!"
                }
            ], win=name, env="amf_params")

        def trigger(context):
            if context["event_type"] != "PropertyUpdate":
                return
            if context["target"] != self.label_panel.panel:
                return
            property_name = context \
                .get('pane_data') \
                .get('content')[context.get('propertyId')] \
                .get('name')
            if property_name == 'Start Index':
                self.label_start_index = int(context.get('value'))
            elif property_name == 'End Index':
                self.label_end_index = int(context.get('value'))
            elif property_name == 'Label Class':
                self.label_class_value = int(context.get('value'))
            elif property_name == 'Label Data Directory Folder Name':
                self.label_data_dir = context.get('value')
            elif property_name == 'Store Label Value':
                if(self.label_start_index > self.label_end_index):
                    print("The start position need to be before the end")
                elif not self.label_data_dir:
                    print("Please input label data directory folder name")
                else:
                    print("Stored the label value")
                    self.export_train_data(self.label_data_dir)

            self.label_panel.panel = update("Label Panel")

        self.label_panel = VisdomControlPanel(
            self.viz,
            update,
            trigger,
            "Label Panel"
        )

    def export_train_data(self, data_dir):
        path = f'./data/{data_dir}'
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = f"{path}/scene.csv"
        with open(file_name, 'a+', newline='') as train_dataset:
            # Create a writer object from csv module
            train_dataset_writer = writer(train_dataset)
            for i in list(range(self.label_start_index, self.label_end_index+1)):
                train_dataset_writer.writerow([
                    self.avg_mask_list[i],
                    self.std_mask_list[i],
                    self.amf.frame_width,
                    self.amf.frame_height,
                    self.label_class_value
                ])

    def show_frame_images(
        self,
        disp_image,
        mog2_disp_mask,
        fd_disp_mask,
        frame_number
    ):
        self.viz.image(
            disp_image,
            win="show_image",
            opts=dict(
                title=f"frame number: {frame_number}",
                width=320,
                height=250,
            ),
            env="amf_params"
        )
        self.viz.image(
            mog2_disp_mask,
            win="show_mog2_mask",
            opts=dict(
                title=f"Display MOG2 Mask: {frame_number}",
                width=320,
                height=250,
            ),
            env="amf_params"
        )
        self.viz.image(
            fd_disp_mask,
            win="show_fd_mask",
            opts=dict(
                title=f"Display FD Mask: {frame_number}",
                width=320,
                height=250,
            ),
            env="amf_params"
        )

    def update_mask_size_graph(self):
        n = len(self.frame_list)
        x = np.linspace(0, n-1, num=n)
        total_y = np.array(self.total_mask_list)
        avg_y = np.array(self.avg_mask_list)
        std_y = np.array(self.std_mask_list)
        mog2_count_y = np.array(self.mog2_contour_count_list)
        fd_count_y = np.array(self.fd_contour_count_list)
        var_y = np.array(self.variance_list)

        self.viz.line(
            X=x,
            Y=total_y,
            opts=dict(
                title="total mask size",
                showlegend=True
            ),
            env="amf_params",
            win="total_mask_graph"
        )

        self.viz.line(
            X=x,
            Y=avg_y,
            opts=dict(
                title="average mask size",
                showlegend=True
            ),
            env="amf_params",
            win="avg_mask_graph"
        )

        self.viz.line(
            X=x,
            Y=std_y,
            opts=dict(
                title="standard deviation of mask size",
                showlegend=True
            ),
            env="amf_params",
            win="std_mask_graph"
        )

        self.viz.line(
            X=x,
            Y=mog2_count_y,
            opts=dict(
                title="contours count",
                showlegend=True
            ),
            env="amf_params",
            win="mog2_contour_count_graph"
        )

        self.viz.line(
            X=x,
            Y=fd_count_y,
            opts=dict(
                title="fd contours count",
                showlegend=True
            ),
            env="amf_params",
            win="fd_contour_count_graph"
        )

        self.viz.line(
            X=x,
            Y=var_y,
            opts=dict(
                title="variance of history frame",
                showlegend=True
            ),
            env="amf_params",
            win="variance_graph"
        )

    def store_params(
        self,
        mog2_mask,
        fd_mask,
        total_area,
        avg_area,
        std_area,
        mog2_contour_count,
        fd_contour_count,
        variance
    ):
        self.mask_list.append(mog2_mask)
        self.fd_mask_list.append(fd_mask)

        self.total_mask_list.append(total_area)
        self.avg_mask_list.append(avg_area)
        self.std_mask_list.append(std_area)

        self.mog2_contour_count_list.append(mog2_contour_count)
        self.fd_contour_count_list.append(fd_contour_count)

        self.variance_list.append(variance)

    def show_params_graph(self):
        self.viz.close(env="amf_params")
        print("Start running through the video")
        frame_length = len(self.frame_list)
        progress_count = 1
        # Run Through Video
        for frame in self.frame_list:
            print(f'{int(progress_count/frame_length*100)} %', end='\r')
            resized_frame = frame_resize(frame.copy())
            mog2_mask = self.apply_mog2(resized_frame)
            fd_mask = self.apply_fd(resized_frame)
            mog2_mask_metadata = self.amf.calculate_mask_metadata(mog2_mask)
            fd_mask_metadata = self.amf.calculate_mask_metadata(fd_mask)

            self.store_params(
                mog2_mask,
                fd_mask,
                mog2_mask_metadata.total,
                mog2_mask_metadata.avg,
                mog2_mask_metadata.std,
                len(mog2_mask_metadata.contours),
                len(fd_mask_metadata.contours),
                self.amf.calculate_variance(mog2_mask_metadata.std)
            )
            progress_count += 1

        print("Completed running through the video")
        print("Start Showing AMF Parameters")

        self.init_viz()
        self.update_mask_size_graph()

        while True:
            time.sleep(1)


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


@app.command()
def params_graph(video: str):
    amf_visdom = AMFParamVisdom(video)
    amf_visdom.show_params_graph()


def main():
    """Main program"""
    app()


if __name__ == "__main__":
    main()
