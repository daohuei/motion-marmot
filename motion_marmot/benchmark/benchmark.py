from motion_marmot.utils.video_utils import extract_video
from motion_marmot.advanced_motion_filter import (
    AdvancedMotionFilter,
    MotionMaskMetadata,
)
from datetime import datetime
import typer
import time
import threading
import os

app = typer.Typer()
contour_count = 0
frame_count = 0


def motion_detection(amf: AdvancedMotionFilter, frame, meta, config):
    mask = amf.mog2_mf.apply(frame.copy())
    contours = amf.calculate_contours(mask)
    is_variance_activated = config.get("variance")
    is_large_bg_movement_activated = config.get("large_bg_movement")
    is_dynamic_bbx_activated = config.get("dynamic_bbx")
    variance = 0
    frame_scene = 0
    dynamic_bbx_thresh = 0
    if (
        is_variance_activated
        or is_large_bg_movement_activated
        or is_dynamic_bbx_activated
    ):
        mask_area = MotionMaskMetadata(contours)
        if is_variance_activated:
            variance = amf.calculate_variance(mask_area.std)
        if is_large_bg_movement_activated or is_dynamic_bbx_activated:
            frame_scene = amf.ssc.predict(
                mask_area.avg, mask_area.std, meta["width"], meta["height"]
            )
            if is_dynamic_bbx_activated:
                dynamic_bbx_thresh = mask_area.avg + mask_area.std
    for contour in contours:
        global contour_count
        contour_count += 1
        if amf.mog2_is_detected(
            contour=contour,
            scene=frame_scene,
            dynamic_bbx_thresh=dynamic_bbx_thresh,
            variance=variance,
            history_variance=is_variance_activated,
            large_bg_movement=is_large_bg_movement_activated,
            dynamic_bbx=is_dynamic_bbx_activated,
        ):
            pass


def run_motion_filter(amf, video_frames: list, video_meta, config, flag):
    for frame in video_frames:
        global frame_count
        frame_count += 1
        motion_detection(amf=amf, frame=frame, meta=video_meta, config=config)
        if not flag():
            break


def recur_motion_filter(flag, video, config):
    video_frames, video_meta = extract_video(video)
    amf = AdvancedMotionFilter(
        "model/scene_knn_model",
        frame_width=video_meta["width"],
        frame_height=video_meta["height"],
    )
    print(f"Start running at: {datetime.fromtimestamp(time.time())}")
    while flag():
        run_motion_filter(amf, video_frames, video_meta, config, flag)
    global contour_count, frame_count
    print(f"Processed Contours Number: {contour_count}")
    print(f"Processed Frames Number: {frame_count}")


@app.command()
def evaluate_cpu(
    video: str,
    interval: int,
    count: int,
    variance: bool = typer.Option(False),
    large_bg_movement: bool = typer.Option(False),
    dynamic_bbx: bool = typer.Option(False),
):
    pid = os.getpid()
    thread_flag = True
    config = {
        "variance": variance,
        "large_bg_movement": large_bg_movement,
        "dynamic_bbx": dynamic_bbx,
    }

    def flag_trigger():
        return thread_flag

    amf_thread = threading.Thread(
        target=recur_motion_filter, args=(flag_trigger, video, config)
    )
    amf_thread.start()
    time.sleep(5)
    # run the pidstat
    pidstat_command = f"pidstat -u -p {pid} {interval} {count} | tail -n 1"
    pidstat_process = os.popen(pidstat_command, "r")
    print(pidstat_process.read())
    pidstat_process.close()
    # kill thread
    thread_flag = False


@app.command()
def motion_counts(
    video: str,
    bounding_box_threshold: int,
    history_variance: bool,
    variance_threshold: int,
    variance_sample_amount: int,
    large_bg_movement: bool,
    dynamic_bbx: bool,
):
    print("Extracting")
    video_frames, video_meta = extract_video(video)
    amf = AdvancedMotionFilter(
        "model/scene_knn_model",
        frame_width=video_meta["width"],
        frame_height=video_meta["height"],
    )
    count = 0
    print("Detecting")
    total = len(video_frames)
    i = 1
    for frame in video_frames:
        print(f"{int((i+1)/total*100)} %", end="\r")
        mask = amf.mog2_mf.apply(frame.copy())
        contours = amf.calculate_contours(mask)
        mask_area = MotionMaskMetadata(contours)
        frame_scene = amf.ssc.predict(
            mask_area.avg, mask_area.std, video_meta["width"], video_meta["height"]
        )
        dynamic_bbx_thresh = mask_area.avg + mask_area.std
        variance = amf.calculate_variance(mask_area.std)
        for contour in contours:
            if amf.mog2_is_detected(
                contour=contour,
                scene=frame_scene,
                dynamic_bbx_thresh=dynamic_bbx_thresh,
                variance=variance,
                bounding_box_threshold=bounding_box_threshold,
                history_variance=history_variance,
                variance_threshold=variance_threshold,
                variance_sample_amount=variance_sample_amount,
                large_bg_movement=large_bg_movement,
                dynamic_bbx=dynamic_bbx,
            ):
                count += 1
                break
        i += 1
    print("\n")
    print(count)


@app.command()
def run():
    print("dummy command")


def main():
    """Main program"""
    app()


if __name__ == "__main__":
    main()
