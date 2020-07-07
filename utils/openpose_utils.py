import sys

import numpy as np
from openpose import pyopenpose as op


def init_openpose(openpose_path):
    params = dict()
    params["model_folder"] = openpose_path + "/models"
    params["hand"] = True
    params["hand_detector"] = 2
    params["hand_net_resolution"] = "384x384"
    params["hand_scale_number"] = 4
    params["hand_scale_range"] = 0.4
    params["body"] = 0

    try:
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        return opWrapper
    except Exception as e:
        # self.print_log(e)
        sys.exit(-1)


def box2oprectangle(box):
    left, right, top, bottom = box
    width = np.abs(right - left)
    height = np.abs(top - bottom)
    max_length = int(max(width, height))
    center = (int(left + (width / 2)), int(bottom - (height / 2)))
    new_top = (int(center[0] - max_length / 1.3), int(center[1] - max_length / 1.3))
    max_length = int(max_length * 1.6)
    hand_rectangle = op.Rectangle(new_top[0], new_top[1], max_length, max_length)
    return hand_rectangle


def detect_keypoints(image, opWrapper_, hand_boxes=None):
    if hand_boxes is None:
        hand_boxes = [[0, image.shape[1] - 1, 0, image.shape[0] - 1]]
    # Left hand only
    hands_rectangles = [[box2oprectangle(box), op.Rectangle(0., 0., 0., 0.)] for box in hand_boxes]

    datum = op.Datum()
    datum.cvInputData = image
    datum.handRectangles = hands_rectangles
    opWrapper_.emplaceAndPop([datum])

    if datum.handKeypoints[0].shape == ():
        hand_keypoints = []
    else:
        hand_keypoints = datum.handKeypoints[0]
    return hand_keypoints, datum.cvOutputData


def read_coordinates(keypoints, frame_width, frame_height, pad=0, train=False):
    keypoints_np = np.array(keypoints[0]).astype(np.float)
    if train:
        key_range_max = np.array([frame_width, frame_height])
        key_range_min = np.array([0, 0])
    else:
        key_range_max = np.max(keypoints_np, axis=0)[:2] + pad
        key_range_min = np.min(keypoints_np, axis=0)[:2] - pad

    coordinates = list(((keypoints_np[:, :2] - key_range_min) / (key_range_max - key_range_min)).flatten())
    score = list(keypoints_np[:, 2])
    return coordinates, score
