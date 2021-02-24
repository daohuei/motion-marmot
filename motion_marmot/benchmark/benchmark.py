from motion_marmot.utils.video_utils import extract_video
from motion_marmot.advanced_motion_filter import AdvancedMotionFilter, MaskArea
import typer
import time
import threading
import os
# from datetime import datetime

app = typer.Typer()
contour_count = 0


def motion_detection(amf: AdvancedMotionFilter, frame, meta, config):
    mask = amf.mog2_mf.apply(frame.copy())
    contours = amf.calculate_contours(mask)
    mask_area = MaskArea(contours)
    frame_scene = amf.ssc.predict(
        mask_area.avg,
        mask_area.std,
        meta['width'],
        meta['height']
    )
    dynamic_bbx_thresh = mask_area.avg + mask_area.std
    variance = amf.calculate_variance(mask_area.std)
    for contour in contours:
        global contour_count
        contour_count += 1
        if amf.mog2_is_detected(
            contour=contour,
            scene=frame_scene,
            dynamic_bbx_thresh=dynamic_bbx_thresh,
            variance=variance,
            history_variance=config.get('variance'),
            large_bg_movement=config.get('large_bg_movement'),
            dynamic_bbx=config.get('dynamic_bbx')
        ):
            break


def run_motion_filter(amf, video_frames: list, video_meta, config, flag):
    for frame in video_frames:
        motion_detection(
            amf=amf,
            frame=frame,
            meta=video_meta,
            config=config
        )
        if not flag():
            break


def recur_motion_filter(flag, video, config):
    video_frames, video_meta = extract_video(video)
    amf = AdvancedMotionFilter('model/scene_knn_model')
    while flag():
        run_motion_filter(amf, video_frames, video_meta, config, flag)
    global contour_count
    print(contour_count)


@app.command()
def evaluate_cpu(
    video: str,
    interval: int,
    count: int,
    variance: bool = typer.Option(False),
    large_bg_movement: bool = typer.Option(False),
    dynamic_bbx: bool = typer.Option(False)
):
    pid = os.getpid()
    thread_flag = True
    config = {
        'variance': variance,
        'large_bg_movement': large_bg_movement,
        'dynamic_bbx': dynamic_bbx
    }

    def flag_trigger():
        return thread_flag

    amf_thread = threading.Thread(target=recur_motion_filter, args=(flag_trigger, video, config))
    amf_thread.start()
    time.sleep(3.5)
    # run the pidstat
    pidstat_command = f'pidstat -u -p {pid} {interval} {count}'
    pidstat_process = os.popen(pidstat_command, 'r')
    print(pidstat_process.read())
    pidstat_process.close()
    # kill thread
    print('killing thread')
    thread_flag = False


@app.command()
def run():
    print("dummy command")


def main():
    """Main program"""
    app()


if __name__ == "__main__":
    main()
