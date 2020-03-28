import os

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm

from model.stgcn import Model

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
# logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = InteractiveSession(config=config)


def get_dataset(dir, num_classes=10, batch_size=32, drop_rem=False, shuffle=False, shuffle_size=100):
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
def test_step(features):
    logits = model(features, training=False)
    return tf.nn.softmax(logits)


@tf.function
def train_step(features, labels):
    def step_fn(features, labels):
        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
        train_acc(labels, logits)
        train_acc_top5(labels, logits)
        cross_entropy_loss(loss)

    strategy.experimental_run_v2(step_fn, args=(features, labels,))


if __name__ == '__main__':
    base_lr = 0.1
    num_classes = 10
    epochs = 200
    checkpoint_path = 'checkpoints'
    log_dir = 'logs'
    train_data_path = 'data/tfrecord'
    test_data_path = 'data/tfrecord'
    save_freq = 10
    steps = [50, 100]
    batch_size = 32
    gpus = None
    strategy = tf.distribute.MirroredStrategy(gpus)
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    gpus = strategy.num_replicas_in_sync

    train_data = get_dataset(train_data_path, num_classes=num_classes, batch_size=global_batch_size,
                             drop_rem=True, shuffle=True)
    train_data = strategy.experimental_distribute_dataset(train_data)

    test_data = get_dataset(test_data_path, num_classes=num_classes, batch_size=batch_size,
                            drop_rem=False, shuffle=False)

    boundaries = [(step * 40000) // batch_size for step in steps]
    values = [base_lr] * (len(steps) + 1)

    for i in range(1, len(steps) + 1):
        values[i] *= 0.1 ** i
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    with strategy.scope():
        model = Model(num_classes=num_classes)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_mngr = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        cross_entropy_loss = tf.keras.metrics.Mean(name='cross_entropy_loss')
        train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
        train_acc_top5 = tf.keras.metrics.TopKCategoricalAccuracy(name='train_acc_top_5')

    epoch_test_acc = tf.keras.metrics.CategoricalAccuracy(name='epoch_test_acc')
    epoch_test_acc_top_5 = tf.keras.metrics.TopKCategoricalAccuracy(name='epoch_test_acc_top_5')
    test_acc_top_5 = tf.keras.metrics.TopKCategoricalAccuracy(name='test_acc_top_5')
    test_acc = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
    summary_writer = tf.summary.create_file_writer(log_dir)

    for data in test_data:
        features, labels = data
        break

    tf.summary.trace_on(graph=True)
    train_step(features, labels)
    with summary_writer.as_default():
        tf.summary.trace_export(name='training_trace', step=0)
    tf.summary.trace_off()

    tf.summary.trace_on(graph=True)
    test_step(features)
    with summary_writer.as_default():
        tf.summary.trace_export(name="testing_trace", step=0)
    tf.summary.trace_off()

    train_iter = 0
    test_iter = 0
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch + 1))
        print("Training: ")
        with strategy.scope():
            for features, labels in tqdm(train_data):
                train_step(features, labels)
                with summary_writer.as_default():
                    tf.summary.scalar("cross_entropy_loss",
                                      cross_entropy_loss.result(),
                                      step=train_iter)
                    tf.summary.scalar("train_acc",
                                      train_acc.result(),
                                      step=train_iter)
                    tf.summary.scalar("train_acc_top_5",
                                      train_acc_top5.result(),
                                      step=train_iter)

                cross_entropy_loss.reset_states()
                train_acc.reset_states()
                train_acc_top5.reset_states()
                train_iter += 1

        print("Testing: ")
        for features, labels in tqdm(test_data):
            y_pred = test_step(features)
            test_acc(labels, y_pred)
            epoch_test_acc(labels, y_pred)
            test_acc_top_5(labels, y_pred)
            epoch_test_acc_top_5(labels, y_pred)
            with summary_writer.as_default():
                tf.summary.scalar("test_acc",
                                  test_acc.result(),
                                  step=test_iter)
                tf.summary.scalar("test_acc_top_5",
                                  test_acc_top_5.result(),
                                  step=test_iter)
            test_acc.reset_states()
            test_acc_top_5.reset_states()
            test_iter += 1
        with summary_writer.as_default():
            tf.summary.scalar("epoch_test_acc",
                              epoch_test_acc.result(),
                              step=epoch)
            tf.summary.scalar("epoch_test_acc_top_5",
                              epoch_test_acc_top_5.result(),
                              step=epoch)
        epoch_test_acc.reset_states()
        epoch_test_acc_top_5.reset_states()

        # if (epoch + 1) % save_freq == 0:
        #     ckpt_save_path = ckpt_mngr.save()
        #     print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
        #                                                         ckpt_save_path))

    ckpt_save_path = ckpt_mngr.save()
    print('Saving final checkpoint for epoch {} at {}'.format(epochs,
                                                              ckpt_save_path))
    model.save_weights('./checkpoints/my_checkpoint')