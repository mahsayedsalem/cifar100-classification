from base_model import BaseModel
import tensorflow as tf


class CifarModel(BaseModel):
    def __init__(self, config):
        super(CifarModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
        self.y = tf.placeholder(tf.float32, shape=(None, 100), name='output_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # network architecture
        conv1 = tf.layers.conv2d(self.x, kernel_size=[3, 3], filters=32, strides=2, padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1")
        conv1_relu = tf.nn.relu(conv1, name="conv1_relu")
        conv1_pool = tf.layers.max_pooling2d(inputs=conv1_relu, pool_size=[2, 2], padding='same', strides=2,
                                             name="conv1_pool")
        conv1_drop = tf.layers.dropout(conv1_pool, rate=self.keep_prob, training=self.training, name="conv1_drop")


        conv2 = tf.layers.conv2d(conv1_drop, kernel_size=[3, 3], filters=64, strides=2, padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv2")
        conv2_relu = tf.nn.relu(conv2, name="conv2_relu")

        conv3 = tf.layers.conv2d(conv2_relu, kernel_size=[3, 3], filters=64, strides=2, padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv3")
        conv3_relu = tf.nn.relu(conv3, name="conv3_relu")
        conv3_pool = tf.layers.max_pooling2d(inputs=conv3_relu, pool_size=[2, 2], strides=2, padding='same',
                                             name="conv3_pool")
        conv3_drop = tf.layers.dropout(conv3_pool, rate=self.keep_prob, training=self.training, name="conv3_drop")

        #conv4_batch = tf.layers.batch_normalization(conv4, training=self.training, name="conv4_batch")

        flatten = tf.contrib.layers.flatten(conv3_drop)

        d1 = tf.layers.dense(flatten, 512, activation=tf.nn.relu, name="dense1")
        d1_drop = tf.layers.dropout(d1, rate=self.keep_prob, training=self.training, name="d1_drop")

        logits = tf.layers.dense(d1_drop, 100, name="logits")

        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                             global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
            self.accuracy = 100.0 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
