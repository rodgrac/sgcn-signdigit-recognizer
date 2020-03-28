import cv2
import numpy as np
import tensorflow as tf

from data_gen.preprocessor.skeleton_gen import detect_keypoints
from data_gen.preprocessor.skeleton_gen import init_openpose
from hand_utils import detector_utils
from hand_utils import draw_util
from model.stgcn import Model

save_video = False

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
    video_source = 2

    cap = cv2.VideoCapture(video_source)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # im_width, im_height = (cap.get(3), cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    im_width, im_height = (640, 480)

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('signdigit_demo.avi', fourcc, fps, (im_width, im_height))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    recorder = cv2.VideoWriter('gcn_signdigit_demo.mp4', fourcc, fps, (640, 480))

    handbox = []

    # detection_graph, sess = detection_rectangles.load_inference_graph()
    detection_graph, sess = detector_utils.load_inference_graph()

    opWrapper = init_openpose()
    model_ = init_Model(10)

    pred_class = -1

    while True:
        ret, image_np = cap.read()
        image_np = cv2.resize(image_np, (im_width, im_height), interpolation=cv2.INTER_CUBIC)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # 1 - Get bounding boxes for seen hands
        # relative_boxes, scores, classes = detection_rectangles.detect_objects(image_np, detection_graph, sess)
        relative_boxes, scores, classes = detector_utils.detect_objects(
            image_np, detection_graph, sess)

        box_maxc = np.argmax(scores)

        box_relative2absolute = lambda box: (
            box[1] * im_width, box[3] * im_width, box[0] * im_height, box[2] * im_height)

        # hand_boxes = [ box_relative2absolute(box)  for box,score in zip(relative_boxes,scores) if score > args.score_thresh]
        if scores[box_maxc] > score_thresh:
            hand_box = [box_relative2absolute(relative_boxes[box_maxc])]

            # 2 - Detect keypoints for those bounding boxes
            keypoints, _ = detect_keypoints(image_np, opWrapper, hand_box)

            if len(keypoints):
                pose, score = read_coordinates(keypoints, im_width, im_height)
                feature = feeder_kin(pose, score)

                pred = test_step(feature).numpy()
                if np.max(pred) > 0.5:
                    pred_class = np.argmax(pred)
                else:
                    pred_class = -1
            else:
                pred_class = -1
            # # 3 - Draw!
            # # Draw bounding boxes
            # draw_util.draw_box_on_image(hand_box, image_np)
            # # Draw hand keypoints
            draw_util.draw_hand_keypoints(keypoints, image_np)
        else:
            pred_class = -1

        # Calculate & Draw Frames per second (FPS)
        # num_frames += 1
        # elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        # fps = num_frames / elapsed_time
        draw_util.draw_text_on_image("Signed digit is : " + str(int(pred_class)), image_np)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        if save_video:
            recorder.write(image_np)

        cv2.imshow('Hand Pose', image_np)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            print("Saving Video")
            save_video = not save_video
        if key & 0xFF == ord('q'):
            break

    cap.release()
    recorder.release()
    cv2.destroyAllWindows()
