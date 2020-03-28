import os

import cv2
import numpy as np
import tensorflow as tf

from data_gen.preprocessor.skeleton_gen import detect_keypoints
from data_gen.preprocessor.skeleton_gen import init_openpose
from hand_utils import detector_utils
from hand_utils import draw_util
from model.stgcn import Model

PATH = "/home/rodneygracian/Desktop/Rod/research/projects/asl/GCN/asl_digits_recog/dataset/sldd/Dataset/5"


def load_img_from_folder(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    return images


@tf.function
def test_step(features):
    logits = model_(features, training=False)
    return tf.nn.softmax(logits)


def read_coordinates(keypoints_, frame_width, frame_height):
    score_, coordinates = [], []
    for key in keypoints_[0]:
        coordinates += [key[0] / frame_width,
                        key[1] / frame_height]
        score_ += [float(key[2])]
    return coordinates, score_


def feeder_kin(pose_, score_):
    data_numpy = np.zeros((3, 1, 21))
    data_numpy[0, :] = pose_[0::2]
    data_numpy[1, :] = pose_[1::2]
    data_numpy[2, :] = score_

    # centralization
    data_numpy[0:2] = data_numpy[0:2] - 0.5
    data_numpy[0][data_numpy[2] == 0] = 0
    data_numpy[1][data_numpy[2] == 0] = 0

    data_numpy = tf.reshape(data_numpy, [1, 3, 1, 21])

    return data_numpy


def init_Model(num_classes):
    model = Model(num_classes=num_classes)
    model.load_weights('./checkpoints/my_checkpoint')
    return model


if __name__ == '__main__':
    score_thresh = 0.5
    video_source = 0

    images = load_img_from_folder(PATH)
    im_width, im_height = (100, 100)

    # detection_graph, sess = detection_rectangles.load_inference_graph()
    detection_graph, sess = detector_utils.load_inference_graph()

    opWrapper = init_openpose()
    model_ = init_Model(10)

    for id, img_o in enumerate(images):
        #img = cv2.resize(img_o, (328, 328), interpolation=cv2.INTER_CUBIC)
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2 - Detect keypoints for those bounding boxes
        keypoints, _ = detect_keypoints(img, opWrapper)

        if len(keypoints):
            pose, score = read_coordinates(keypoints, im_width, im_height)
            feature = feeder_kin(pose, score)

            pred = test_step(feature).numpy()
            if np.max(pred) > 0.5:
                pred_class = np.argmax(pred)
                print(pred_class, np.max(pred))

            # # 3 - Draw!
            # # Draw bounding boxes
            # draw_util.draw_box_on_image(hand_box, image_np)
            # # Draw hand keypoints
            draw_util.draw_hand_keypoints(keypoints, img)

        # Calculate & Draw Frames per second (FPS)
        # num_frames += 1
        # elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        # fps = num_frames / elapsed_time
        # draw_util.draw_fps_on_image("FPS : " + str(int(fps)), image_np)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Pose', img)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
