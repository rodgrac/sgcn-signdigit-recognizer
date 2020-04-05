import argparse
from distutils.util import strtobool

import cv2
import numpy as np
import tensorflow as tf
import yaml

from hand_utils import hand_detector
from hand_utils import hand_draw_util
from model.sgcn import Model
from utils import openpose_utils as op_ap
from utils.io_utils import IO


class SignDigit_Webcam(IO):
    def __init__(self, argv=None):
        self.load_arg(argv)
        super().__init__(self.arg)
        self.save_arg(self.arg, 'testing')

        self.num_classes = self.arg.num_classes
        self.handscore_thresh = self.arg.hand_thresh
        self.digitscore_thresh = self.arg.sign_thresh

        self.video_source = self.arg.video_src
        self.im_width, self.im_height = (self.arg.im_width, self.arg.im_height)

        self.openpose_path = self.arg.openpose
        self.checkpoint_path = self.arg.ckpt_dir
        self.weights = self.arg.weights
        self.gpus = self.arg.gpus

        self.cap = cv2.VideoCapture(self.video_source)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        # im_width, im_height = (cap.get(3), cap.get(4))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.det_graph, self.sess = hand_detector.load_inference_graph()

        self.opWrapper = op_ap.init_openpose(self.openpose_path)

        self.model_ = Model(num_classes=self.num_classes)
        self.model_.load_weights('{}/{}'.format(self.checkpoint_path, self.weights))

        self.handbox = []
        self.keypoints = []
        self.pred_class = -1

        self.recorder = None

    def start(self):
        save_video = False

        while True:
            ret, image_np = self.cap.read()
            image_np = cv2.resize(image_np, (self.im_width, self.im_height), interpolation=cv2.INTER_CUBIC)
            try:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            except:
                self.print_log("Error converting to RGB")

            relative_boxes, scores, classes = hand_detector.detect_objects(
                image_np, self.det_graph, self.sess)

            box_maxc = np.argmax(scores)

            box_relative2absolute = lambda box: (
                box[1] * self.im_width, box[3] * self.im_width, box[0] * self.im_height, box[2] * self.im_height)

            if scores[box_maxc] > self.handscore_thresh:
                hand_box = [box_relative2absolute(relative_boxes[box_maxc])]

                self.keypoints, _ = op_ap.detect_keypoints(image_np, self.opWrapper, hand_box)

                if len(self.keypoints):
                    pose, score = op_ap.read_coordinates(self.keypoints, self.im_width, self.im_height)
                    feature = feeder_kin(pose, score)

                    pred = self.test_step(feature).numpy()
                    if np.max(pred) > self.digitscore_thresh:
                        self.pred_class = np.argmax(pred)
                    else:
                        self.pred_class = -1
                else:
                    self.pred_class = -1

                # draw_util.draw_box_on_image(hand_box, image_np)
                hand_draw_util.draw_hand_keypoints(self.keypoints, image_np)
            else:
                self.pred_class = -1

            # num_frames += 1
            # elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            # fps = num_frames / elapsed_time
            hand_draw_util.draw_text_on_image("Signed digit is : " + str(int(self.pred_class)), image_np)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            if save_video:
                self.recorder.write(image_np)

            cv2.imshow('Hand Pose', image_np)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                if not save_video:
                    print("Recording Video")
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                    self.recorder = cv2.VideoWriter('recorder/Demo_{}.mp4'.format(self.cur_time), fourcc, self.fps,
                                                    (self.im_width, self.im_height))
                else:
                    print("Saving Video")
                    self.recorder.release()
                save_video = not save_video
            if key & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def load_arg(self, argv=None):
        parser = self.get_parser()
        p = parser.parse_args(argv)

        if p.config:
            with open(p.config, 'r') as f:
                darg = yaml.load(f)

            key = vars(p).keys()
            for k in darg.keys():
                if k not in key:
                    self.print_log('Unknown Arguments: {}'.format(k))
                    assert k in key
            parser.set_defaults(**darg)

        self.arg = parser.parse_args(argv)

    @tf.function
    def test_step(self, features):
        logits = self.model_(features, training=False)
        return tf.nn.softmax(logits)

    @staticmethod
    def get_parser(add_help=False):
        parser = argparse.ArgumentParser(add_help=add_help, description="SGCN Training")
        parser.add_argument('-c', '--config', type=str, default=None)
        parser.add_argument('-dd', '--home_dir', type=str, default=None)

        parser.add_argument('--openpose', type=str, default=None)
        parser.add_argument('--num_classes', type=int, default=10)

        parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
        parser.add_argument('--weights', type=str, default='final_checkpoint')
        parser.add_argument('--gpus', default=None, nargs='+')

        parser.add_argument('--hand_thresh', type=float, default=0.5)
        parser.add_argument('--sign_thresh', type=float, default=0.5)

        parser.add_argument('--video_src', type=int, default=1)
        parser.add_argument('--im_width', type=int, default=640)
        parser.add_argument('--im_height', type=int, default=480)

        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--model', default=None)

        parser.add_argument('--log_dir', type=str, default='logs')
        parser.add_argument('--save_log', type=strtobool, default=True)
        parser.add_argument('--print_log', type=strtobool, default=True)

        return parser


def feeder_kin(pose_, score_):
    data_numpy = np.zeros((3, 1, 21))
    data_numpy[0, :] = pose_[0::2]
    data_numpy[1, :] = pose_[1::2]
    data_numpy[2, :] = score_

    data_numpy[0:2] = data_numpy[0:2] - 0.5
    data_numpy[0][data_numpy[2] == 0] = 0
    data_numpy[1][data_numpy[2] == 0] = 0

    data_numpy = tf.reshape(data_numpy, [1, 3, 1, 21])

    return data_numpy
