import cv2
import numpy as np
from motion_marmot.simple_scene_classifier import SimpleSceneClassifier


class AdvancedMotionFilter():
    """
    Automation of the motion filter to improve the FP rate.
    """
    DEFAULT_CONFIG = {
        'bounding_box_thresh': 200,
        'variance_thresh': 10
    }

    def __init__(self, ssc_model: str, variance_sample_amount=5):
        self.ssc = SimpleSceneClassifier("For Advanced Motion Filter", ssc_model)
        self.mog2_mf = cv2.createBackgroundSubtractorMOG2()
        self.config = self.DEFAULT_CONFIG
        self.variance_sample_amount = variance_sample_amount
        self.prev_frame_storage = []
        self.prev_frame_storage = [self.calculate_variance(0)]

    def __str__(self):
        return f"AdvancedMotionFilter(ssc={self.ssc},config={self.config})"

    def calculate_contours(self, mask):
        return cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]

    def mog2_is_detected(
        self,
        contour,
        scene,
        dynamic_bbx_thresh,
        variance,
        history_variance=False,
        large_bg_movement=False,
        dynamic_bbx=False
    ):
        area = cv2.contourArea(contour)

        bounding_box_bool = \
            ((not dynamic_bbx or scene != 2) and area > self.config.get('bounding_box_thresh')) or \
            (dynamic_bbx and scene == 2 and area > dynamic_bbx_thresh)
        large_bg_movement_bool = \
            (not large_bg_movement) or \
            (large_bg_movement and scene != 3)
        history_variance_bool = \
            (not history_variance) or \
            (history_variance and variance < self.config.get('variance_thresh'))

        return bounding_box_bool and large_bg_movement_bool and history_variance_bool

    def draw_detection_box(self, box, frame):
        cv2.rectangle(frame, (box.x, box.y), (box.x + box.w, box.y + box.h), (255, 0, 0), 2)

    def calculate_variance(self, std):
        self.prev_frame_storage.append(std)
        if len(self.prev_frame_storage) > self.variance_sample_amount:
            self.prev_frame_storage.pop(0)
        variance = 0
        if len(self.prev_frame_storage) > 0:
            variance = np.var(self.prev_frame_storage)
        return variance


class BoundingBox():
    """A rectangle bounding box that bounding motion mask detection contour"""

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class MaskArea():
    """
    The class contains related features of motion masks which is generated by MOG2 in OpenCV
    """

    def __init__(self, contours):
        """Init function for MaskArea"""
        self.contours = contours
        contour_area_list = list(
            map(cv2.contourArea, self.contours)
        ) if len(self.contours) > 0 else list()
        self.count = len(contour_area_list)
        self.total = sum(contour_area_list) if self.count > 0 else 0
        self.avg = self.total / \
            self.count if self.count > 0 else 0
        self.std = np.std(np.array(contour_area_list))