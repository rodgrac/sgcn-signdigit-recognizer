import os
import sys

import cv2
import numpy as np
from openpose import pyopenpose as op

from .preprocessor import Preprocessor


class Skeleton_Generator(Preprocessor):
    def __init__(self, argv=None):
        super().__init__('skeleton', argv)
        self.input_dir = self.dataset_dir
        self.openpose_path = self.arg.skeleton['openpose']
        self.opWrap = self.init_openpose()

    def start(self):
        label_map_path = '{}/label.json'.format(self.output_dir)
        self.print_log("Source directory: '{}'".format(self.input_dir))
        self.print_log("Estimating poses to '{}'...".format(self.output_dir))

        self.process_frames(self.input_dir, self.output_dir, label_map_path, self.opWrap)

        self.print_log("Estimation done")

    def init_openpose(self):
        params = dict()
        params["model_folder"] = self.openpose_path + "/models"
        params["hand"] = True
        params["hand_detector"] = 2
        params["hand_net_resolution"] = "384x384"
        params["hand_scale_number"] = 6
        params["hand_scale_range"] = 0.4
        params["body"] = 0

        try:
            # Starting OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            return opWrapper
        except Exception as e:
            # self.print_log(e)
            sys.exit(-1)

    def process_frames(self, input_dir, output_dir, label_path, opWrapper):
        label_map = self.load_label_map(label_path)
        for subdir in os.listdir(input_dir):
            subdir_path = os.path.join(input_dir, subdir)
            num_files = len(os.listdir(subdir_path))
            for filename in os.listdir(subdir_path):
                self.print_log("* {} [{} / {}] \t{} ...".format(subdir, len(label_map) + 1, num_files * 10, filename))
                img = cv2.imread(os.path.join(subdir_path, filename))
                # img = cv2.resize(img, (368, 368), interpolation=cv2.INTER_CUBIC)
                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                (im_height, im_width) = img.shape[:2]
                keypoints, _ = self.detect_keypoints(img, opWrapper)
                # draw_util.draw_hand_keypoints(keypoints, img)
                # cv2.imshow("Hand Pose", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(0)

                output_sequence_path = '{}/{}.json'.format(output_dir, os.path.splitext(filename)[0])
                frame_info = self.json_pack(im_width, im_height, keypoints, label=subdir, label_index=int(subdir))
                self.save_json(frame_info, output_sequence_path)

                label_info = dict()
                label_info['has_skeleton'] = len(frame_info['data']) > 0
                label_info['label'] = frame_info['label']
                label_info['label_index'] = frame_info['label_index']
                label_map[os.path.splitext(filename)[0]] = label_info

                # save label map:
                self.save_json(label_map, label_path)

        return label_map

    # Converts a hand bounding box into a OpenPose Rectangle
    def box2oprectangle(self, box):
        left, right, top, bottom = box
        width = np.abs(right - left)
        height = np.abs(top - bottom)
        max_length = int(max(width, height))
        center = (int(left + (width / 2)), int(bottom - (height / 2)))
        # Openpose hand detector needs the bounding box to be quite big , so we make it bigger
        # Top point for rectangle
        new_top = (int(center[0] - max_length / 1.3), int(center[1] - max_length / 1.3))
        max_length = int(max_length * 1.6)
        hand_rectangle = op.Rectangle(new_top[0], new_top[1], max_length, max_length)
        return hand_rectangle

    def detect_keypoints(self, image, opWrapper_, hand_boxes=None):
        if hand_boxes is None:
            hand_boxes = [[0, image.shape[1] - 1, 0, image.shape[0] - 1]]
        # We are considering every seen hand is a left hand
        hands_rectangles = [[self.box2oprectangle(box), op.Rectangle(0., 0., 0., 0.)] for box in hand_boxes]
        # hands_rectangles.append([hand_rectangle,op.Rectangle(0., 0., 0., 0.)])

        # Create new datum
        datum = op.Datum()
        datum.cvInputData = image
        datum.handRectangles = hands_rectangles
        # Process and display image
        opWrapper_.emplaceAndPop([datum])

        if datum.handKeypoints[0].shape == ():
            # if there were no detections
            hand_keypoints = []
        else:
            hand_keypoints = datum.handKeypoints[0]
        return hand_keypoints, datum.cvOutputData

    def json_pack(self, frame_width, frame_height, keypoints, label='unknown', label_index=-1):
        frame_data = {}
        skeleton = {}
        score, coordinates = self.read_coordinates(keypoints, frame_width, frame_height)
        skeleton['pose'] = coordinates
        skeleton['score'] = score

        frame_data['skeleton'] = [skeleton]

        video_info = dict()
        video_info['data'] = [frame_data]
        video_info['label'] = label
        video_info['label_index'] = label_index
        return video_info

    def read_coordinates(self, keypoints, frame_width, frame_height):
        score, coordinates = [], []
        for key in keypoints[0]:
            coordinates += [key[0] / frame_width,
                            key[1] / frame_height]
            score += [float(key[2])]
        return score, coordinates

    def load_label_map(self, label_map_path):
        label_map = dict()

        if os.path.isfile(label_map_path):
            label_map = self.read_json(label_map_path)
        return label_map
