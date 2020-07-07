import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .preprocessor import Preprocessor


class Tfrecord_Generator(Preprocessor):
    def __init__(self, argv=None):
        super().__init__('tfrecord', argv)
        self.num_shards = 40

    def start(self):
        self.gen_tfrecord_data(self.num_shards, self.input_dir + "/train_label.pkl", self.input_dir + "/train_data.npy",
                               os.path.join(self.output_dir, 'train'), True)
        self.gen_tfrecord_data(self.num_shards, self.input_dir + "/val_label.pkl", self.input_dir + "/val_data.npy",
                               os.path.join(self.output_dir, 'test'), True)

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_example(self, features, label):
        feature = {
            'features': self._bytes_feature(tf.io.serialize_tensor(features.astype(np.float32))),
            'label': self._int64_feature(label)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    def gen_tfrecord_data(self, num_shards, label_path, data_path, dest_folder, shuffle):
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
            tfrecord_data_path = os.path.join(dest_folder,
                                              data_path.name.split(".")[0] + "-" + str(shard) + ".tfrecord")
            with tf.io.TFRecordWriter(tfrecord_data_path) as writer:
                for i in range(shard * step, (shard * step) + step if shard < num_shards - 1 else len(labels)):
                    writer.write(self.serialize_example(data[i], labels[i]))
