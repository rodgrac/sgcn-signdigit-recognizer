import tensorflow as tf

from graph.kinetics_skeleton import Graph

regularizer = tf.keras.regularizers.l2(l=0.0001)
initializer = tf.keras.initializers.VarianceScaling(scale=2., mode='fan_out', distribution='truncated_normal')


class SGCN(tf.keras.Model):
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(filters * kernel_size, kernel_size=1, padding='same',
                                           kernel_initializer=initializer, data_format='channels_first',
                                           kernel_regularizer=regularizer)

    def call(self, x, A, training):
        x = self.conv(x)

        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]

        x = tf.reshape(x, [N, self.kernel_size, C // self.kernel_size, T, V])
        x = tf.einsum('nkctv,kvw->nctw', x, A)
        return x, A


class SGCN_Block(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, stride=1, activation='relu',
                 residual=True, downsample=False):
        super().__init__()
        self.sgcn = SGCN(filters, kernel_size=kernel_size)

        self.bnorm = (tf.keras.layers.BatchNormalization(axis=1))

        self.act = tf.keras.layers.Activation(activation)

        if not residual:
            self.residual = lambda x, training=False: 0
        elif residual and stride == 1 and not downsample:
            self.residual = lambda x, training=False: x
        else:
            self.residual = tf.keras.Sequential()
            self.residual.add(tf.keras.layers.Conv2D(filters,
                                                     kernel_size=[1, 1],
                                                     strides=[stride, 1],
                                                     padding='same',
                                                     kernel_initializer=initializer,
                                                     data_format='channels_first',
                                                     kernel_regularizer=regularizer))
            self.residual.add(tf.keras.layers.BatchNormalization(axis=1))

    def call(self, x, A, training):
        res = self.residual(x, training=training)
        x, A = self.sgcn(x, A, training=training)
        x = self.bnorm(x, training=training)
        x += res
        x = self.act(x)
        return x, A


class Model(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()

        graph = Graph()
        self.A = tf.Variable(graph.A, dtype=tf.float32, trainable=False, name='adjacency_matrix')

        self.data_bn = tf.keras.layers.BatchNormalization(axis=1)

        self.SGCN_layers = []

        self.SGCN_layers.append(SGCN_Block(64, residual=False))
        self.SGCN_layers.append(SGCN_Block(64))
        self.SGCN_layers.append(SGCN_Block(64))
        self.SGCN_layers.append(SGCN_Block(64))
        self.SGCN_layers.append(SGCN_Block(128, stride=2, downsample=True))
        self.SGCN_layers.append(SGCN_Block(128))
        self.SGCN_layers.append(SGCN_Block(128))
        self.SGCN_layers.append(SGCN_Block(256, stride=2, downsample=True))
        self.SGCN_layers.append(SGCN_Block(256))
        self.SGCN_layers.append(SGCN_Block(256))

        self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')

        self.logits = tf.keras.layers.Conv2D(num_classes,
                                             kernel_size=1,
                                             padding='same',
                                             kernel_initializer=initializer,
                                             data_format='channels_first',
                                             kernel_regularizer=regularizer)

    def call(self, x, training):
        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]

        x = tf.transpose(x, perm=[0, 3, 1, 2])
        x = tf.reshape(x, [N, V * C, T])
        x = self.data_bn(x, training=training)
        x = tf.reshape(x, [N, V, C, T])
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        x = tf.reshape(x, [N, C, T, V])

        A = self.A
        for layer in self.SGCN_layers:
            x, A = layer(x, A, training=training)

        # N,C,T,V
        x = self.pool(x)
        x = tf.reshape(x, [N, -1, 1, 1])
        #x = tf.reduce_mean(x, axis=1)
        x = self.logits(x)
        x = tf.reshape(x, [N, -1])

        return x
