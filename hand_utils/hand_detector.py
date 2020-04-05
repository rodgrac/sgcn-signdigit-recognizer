# Utilities for object detector.

import logging
import os

import cv2
import numpy as np
import tensorflow as tf

from hand_utils import label_map_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import GraphDef

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

TRAINED_MODEL_DIR = 'frozen_graphs'
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/ssd5_optimized_inference_graph.pb'
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/hand_label_map.pbtxt'

NUM_CLASSES = 1
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_inference_graph():
    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = Session(graph=detection_graph, config=config)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess


def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np):
    focalLength = 875
    avg_width = 4.0
    color0 = (255, 0, 0)
    color1 = (0, 50, 255)
    for i in range(num_hands_detect):
        if scores[i] > score_thresh:
            if classes[i] == 1: id = 'open'
            if classes[i] == 2:
                id = 'closed'
                avg_width = 3.0

            if i == 0:
                color = color0
            else:
                color = color1

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            dist = distance_to_camera(avg_width, focalLength, int(right - left))

            cv2.rectangle(image_np, p1, p2, color, 3, 1)

            cv2.putText(image_np, 'hand ' + str(i) + ': ' + id, (int(left), int(top) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(image_np, 'confidence: ' + str("{0:.2f}".format(scores[i])),
                        (int(left), int(top) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(image_np, 'distance: ' + str("{0:.2f}".format(dist) + ' inches'),
                        (int(im_width * 0.7), int(im_height * 0.9 + 30 * i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth


def detect_objects(image_np, detection_graph, sess):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
