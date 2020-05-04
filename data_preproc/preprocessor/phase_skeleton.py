import os

import cv2

from hand_utils.hand_draw_util import draw_hand_keypoints
from utils import openpose_utils as op_ap
from .preprocessor import Preprocessor


class Skeleton_Generator(Preprocessor):
    def __init__(self, argv=None):
        super().__init__('skeleton', argv)
        self.input_dir = self.dataset_dir
        self.display_keypoints = self.arg.skeleton['display_keypoints']
        self.openpose_path = self.arg.skeleton['openpose']
        self.resize = self.arg.skeleton['resize']
        self.im_width = self.arg.skeleton['im_width']
        self.im_height = self.arg.skeleton['im_height']
        self.opWrap = op_ap.init_openpose(self.openpose_path)

    def start(self):
        label_map_path = '{}/label.json'.format(self.output_dir)
        self.print_log("Source directory: '{}'".format(self.input_dir))
        self.print_log("Estimating poses to '{}'...".format(self.output_dir))

        self.process_frames(self.input_dir, self.output_dir, label_map_path, self.opWrap)

        self.print_log("Estimation done")

    def process_frames(self, input_dir, output_dir, label_path, opWrapper):
        label_map = self.load_label_map(label_path)
        for subdir in os.listdir(input_dir):
            subdir_path = os.path.join(input_dir, subdir)
            num_files = len(os.listdir(subdir_path))
            for filename in os.listdir(subdir_path):
                if os.path.splitext(filename)[0] not in label_map:
                    self.print_log(
                        "* {} [{} / {}] \t{} ...".format(subdir, len(label_map) + 1, num_files * 10, filename))
                    img = cv2.imread(os.path.join(subdir_path, filename))
                    if self.resize is True:
                        img = cv2.resize(img, (self.im_width, self.im_height), interpolation=cv2.INTER_CUBIC)
                    img = cv2.flip(img, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    (im_height, im_width) = img.shape[:2]
                    keypoints, _ = op_ap.detect_keypoints(img, opWrapper)
                    if self.display_keypoints:
                        draw_hand_keypoints(keypoints, img)
                        cv2.imshow("Hand Pose", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        key = cv2.waitKey(0)

                        if key & 0xFF == ord('q'):
                            exit()

                        break

                    output_sequence_path = '{}/{}.json'.format(output_dir, os.path.splitext(filename)[0])

                    frame_info = self.json_pack(im_width, im_height, keypoints, label=subdir,
                                                label_index=self.labels_list.index(subdir))
                    self.save_json(frame_info, output_sequence_path)

                    label_info = dict()
                    label_info['has_skeleton'] = len(frame_info['data']) > 0
                    label_info['label'] = frame_info['label']
                    label_info['label_index'] = frame_info['label_index']
                    label_map[os.path.splitext(filename)[0]] = label_info

                    # save label map:
                    self.save_json(label_map, label_path)

        if self.display_keypoints:
            exit()

        return label_map

    def json_pack(self, frame_width, frame_height, keypoints, label='unknown', label_index=-1):
        frame_data = {}
        skeleton = {}
        coordinates, score = op_ap.read_coordinates(keypoints, frame_width, frame_height)
        skeleton['pose'] = coordinates
        skeleton['score'] = score

        frame_data['skeleton'] = [skeleton]

        video_info = dict()
        video_info['data'] = [frame_data]
        video_info['label'] = label
        video_info['label_index'] = label_index
        return video_info

    def load_label_map(self, label_map_path):
        label_map = dict()

        if os.path.isfile(label_map_path):
            label_map = self.read_json(label_map_path)
        return label_map
