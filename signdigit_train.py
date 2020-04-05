import argparse
import os
from distutils.util import strtobool

import tensorflow as tf
import yaml
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm

from model.sgcn import Model
from utils.io_utils import IO


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
# logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

class SignDigit_Training(IO):
    def __init__(self, argv=None):
        self.load_arg(argv)
        super().__init__(self.arg)
        self.save_arg(self.arg, 'training')

        self.config = ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.session = InteractiveSession(config=self.config)

        self.base_lr = self.arg.base_lr
        self.num_classes = self.arg.num_classes
        self.epochs = self.arg.num_epochs

        self.result_name = self.arg.result_name
        self.save_result = self.arg.save_result
        self.checkpoint_path = self.arg.ckpt_dir
        self.clear_ckpt_dir = self.arg.clear_ckpt
        self.train_data_path = self.arg.train_dir
        self.test_data_path = self.arg.test_dir
        self.save_freq = self.arg.ckpt_freq

        self.steps = self.arg.steps
        self.train_batch_size = self.arg.train_batch_size
        self.test_batch_size = self.arg.test_batch_size
        self.gpus = self.arg.gpus
        self.strategy = tf.distribute.MirroredStrategy(self.gpus)
        self.global_batch_size = self.train_batch_size * self.strategy.num_replicas_in_sync
        self.gpus = self.strategy.num_replicas_in_sync

        self.model = Model(num_classes=self.num_classes)

        boundaries = [(step * 40000) // self.train_batch_size for step in self.steps]
        values = [self.base_lr] * (len(self.steps) + 1)

        for i in range(1, len(self.steps) + 1):
            values[i] *= 0.1 ** i
        self.learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)

        with self.strategy.scope():
            ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            self.ckpt_mngr = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=5)

            self.cross_entropy_loss = tf.keras.metrics.Mean(name='cross_entropy_loss')
            self.train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
            self.train_acc_top5 = tf.keras.metrics.TopKCategoricalAccuracy(name='train_acc_top_5')

        self.epoch_test_acc = tf.keras.metrics.CategoricalAccuracy(name='epoch_test_acc')
        self.epoch_test_acc_top_5 = tf.keras.metrics.TopKCategoricalAccuracy(name='epoch_test_acc_top_5')
        self.test_acc_top_5 = tf.keras.metrics.TopKCategoricalAccuracy(name='test_acc_top_5')
        self.test_acc = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

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

    def start(self):

        train_data = self.get_dataset(self.train_data_path, num_classes=self.num_classes,
                                      batch_size=self.global_batch_size,
                                      drop_rem=True, shuffle=True)
        train_data = self.strategy.experimental_distribute_dataset(train_data)

        test_data = self.get_dataset(self.test_data_path, num_classes=self.num_classes, batch_size=self.test_batch_size,
                                     drop_rem=False, shuffle=False)

        if self.clear_ckpt_dir:
            self.remove_dir(os.path.join(self.home_dir, self.checkpoint_path))
            self.create_dir(os.path.join(self.home_dir, self.checkpoint_path))

        for data in test_data:
            features, labels = data
            break

        tf.summary.trace_on(graph=True)
        self.train_step(features, labels)
        with self.summary_writer.as_default():
            tf.summary.trace_export(name='training_trace', step=0)
        tf.summary.trace_off()

        tf.summary.trace_on(graph=True)
        self.test_step(features)
        with self.summary_writer.as_default():
            tf.summary.trace_export(name="testing_trace", step=0)
        tf.summary.trace_off()

        train_iter = 0
        test_iter = 0
        for epoch in range(self.epochs):
            self.print_log("Epoch: {}".format(epoch + 1))
            self.print_log("Training: ")
            with self.strategy.scope():
                for features, labels in tqdm(train_data):
                    self.train_step(features, labels)
                    with self.summary_writer.as_default():
                        tf.summary.scalar("cross_entropy_loss",
                                          self.cross_entropy_loss.result(),
                                          step=train_iter)
                        tf.summary.scalar("train_acc",
                                          self.train_acc.result(),
                                          step=train_iter)
                        tf.summary.scalar("train_acc_top_5",
                                          self.train_acc_top5.result(),
                                          step=train_iter)

                    self.cross_entropy_loss.reset_states()
                    self.train_acc.reset_states()
                    self.train_acc_top5.reset_states()
                    train_iter += 1

            self.print_log("Testing: ")
            for features, labels in tqdm(test_data):
                y_pred = self.test_step(features)
                self.test_acc(labels, y_pred)
                self.epoch_test_acc(labels, y_pred)
                self.test_acc_top_5(labels, y_pred)
                self.epoch_test_acc_top_5(labels, y_pred)
                with self.summary_writer.as_default():
                    tf.summary.scalar("test_acc",
                                      self.test_acc.result(),
                                      step=test_iter)
                    tf.summary.scalar("test_acc_top_5",
                                      self.test_acc_top_5.result(),
                                      step=test_iter)
                self.test_acc.reset_states()
                self.test_acc_top_5.reset_states()
                test_iter += 1
            with self.summary_writer.as_default():
                tf.summary.scalar("epoch_test_acc",
                                  self.epoch_test_acc.result(),
                                  step=epoch)
                tf.summary.scalar("epoch_test_acc_top_5",
                                  self.epoch_test_acc_top_5.result(),
                                  step=epoch)
            self.epoch_test_acc.reset_states()
            self.epoch_test_acc_top_5.reset_states()

            if (epoch + 1) % self.save_freq == 0:
                ckpt_save_path = self.ckpt_mngr.save()
                self.print_log('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                             ckpt_save_path))
        if self.save_result:
            self.print_log('Saving final checkpoint for epoch {} at {}/{}'.format(self.epochs,
                                                                                  self.checkpoint_path,
                                                                                  self.result_name))
            self.model.save_weights('{}/{}'.format(self.checkpoint_path, self.result_name))

    def get_dataset(self, dir, num_classes=10, batch_size=32, drop_rem=False, shuffle=False, shuffle_size=100):
        feature_desc = {
            'features': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_feature_function(example_proto):
            features = tf.io.parse_single_example(example_proto, feature_desc)
            data = tf.io.parse_tensor(features['features'], tf.float32)
            label = tf.one_hot(features['label'], num_classes)
            data = tf.reshape(data, (3, 1, 21))
            return data, label

        records = [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith('tfrecord')]
        dataset = tf.data.TFRecordDataset(records, num_parallel_reads=len(records))
        dataset = dataset.map(_parse_feature_function)
        dataset = dataset.batch(batch_size, drop_remainder=drop_rem)
        dataset = dataset.prefetch(batch_size)
        if shuffle:
            dataset = dataset.shuffle(shuffle_size)
        return dataset

    @tf.function
    def test_step(self, features):
        logits = self.model(features, training=False)
        return tf.nn.softmax(logits)

    @tf.function
    def train_step(self, features, labels):
        self.strategy.experimental_run_v2(self.step_fn, args=(features, labels,))

    def step_fn(self, features, labels):
        with tf.GradientTape() as tape:
            logits = self.model(features, training=True)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_sum(cross_entropy) * (1.0 / self.global_batch_size)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
        self.train_acc(labels, logits)
        self.train_acc_top5(labels, logits)
        self.cross_entropy_loss(loss)

    @staticmethod
    def get_parser(add_help=False):
        parser = argparse.ArgumentParser(add_help=add_help, description="SGCN Training")
        parser.add_argument('-c', '--config', type=str, default=None)
        parser.add_argument('-dd', '--home_dir', type=str, default=None)
        parser.add_argument('-clr', '--clear_ckpt', type=strtobool, default=False)

        parser.add_argument('--train_dir', type=str, default='data/tfrecord')
        parser.add_argument('--test_dir', type=str, default='data/tfrecord')
        parser.add_argument('--num_classes', type=int, default=10)

        parser.add_argument('--result_name', type=str, default='final_checkpoint')
        parser.add_argument('--save_result', type=strtobool, default=True)
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--num_epochs', type=int, default=80)
        parser.add_argument('--gpus', default=None, nargs='+')

        parser.add_argument('--train_batch_size', type=int, default=32)
        parser.add_argument('--test_batch_size', type=int, default=32)

        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+')
        parser.add_argument('--base_lr', type=float, default=0.01)
        parser.add_argument('--steps', type=int, default=[], nargs='+')
        # parser.add_argument('--optimizer', default='SGD')
        parser.add_argument('--nesterov', type=strtobool, default=True)
        parser.add_argument('--weight_decay', type=float, default=0.0001)

        parser.add_argument('--model', default=None)
        parser.add_argument('--weights', default=None)
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+')

        parser.add_argument('--log_dir', type=str, default='logs')
        parser.add_argument('--save_log', type=strtobool, default=True)
        parser.add_argument('--print_log', type=strtobool, default=True)
        parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
        parser.add_argument('--ckpt_freq', type=int, default=10)

        return parser
