import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path

DATA_PATH = "/home/rodneygracian/Desktop/Rod/research/projects/asl/GCN/asl_digits_recog/src/data/normalized"
OUTPUT_PATH = "/home/rodneygracian/Desktop/Rod/research/projects/asl/GCN/asl_digits_recog/src/data/tfrecord"

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(features, label):
    feature = {
        'features': _bytes_feature(tf.io.serialize_tensor(features.astype(np.float32))),
        'label': _int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def gen_tfrecord_data(num_shards, label_path, data_path, dest_folder, shuffle):
    label_path = Path(label_path)
    if not (label_path.exists()):
        print('Label file does not exist')
        return

    data_path = Path(data_path)
    if not (data_path.exists()):
        print('Data file does not exist')
        return

    try:
        with open(label_path) as f:
            _, labels = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            _, labels = pickle.load(f, encoding='latin1')

    # Datashape: Total_samples, 3, 1, 21
    data = np.load(data_path, mmap_mode='r')
    labels = np.array(labels)

    if len(labels) != len(data):
        print("Data and label lengths didn't match!")
        print("Data size: {} | Label Size: {}".format(data.shape, labels.shape))
        return -1

    print("Data shape:", data.shape)
    if shuffle:
        p = np.random.permutation(len(labels))
        labels = labels[p]
        data = data[p]

    dest_folder = Path(dest_folder)
    if not (dest_folder.exists()):
        os.mkdir(dest_folder)

    step = len(labels) // num_shards
    for shard in tqdm(range(num_shards)):
        tfrecord_data_path = os.path.join(dest_folder, data_path.name.split(".")[0] + "-" + str(shard) + ".tfrecord")
        with tf.io.TFRecordWriter(tfrecord_data_path) as writer:
            for i in range(shard * step, (shard * step) + step if shard < num_shards - 1 else len(labels)):
                writer.write(serialize_example(data[i], labels[i]))


if __name__ == '__main__':
    num_shards = 40

    gen_tfrecord_data(num_shards, DATA_PATH + "/train_label.pkl", DATA_PATH + "/train_data.npy",
                      OUTPUT_PATH, True)
    gen_tfrecord_data(num_shards, DATA_PATH + "/val_label.pkl", DATA_PATH + "/val_data.npy",
                      OUTPUT_PATH, True)
