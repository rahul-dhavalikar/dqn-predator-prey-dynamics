import tensorflow as tf


class DDQN():
    def __init__(self, n_actions, state_size):
        self.n_actions = n_actions
        self.state_size = state_size

        self.scalarInput = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)

        self.dense1 = tf.nn.relu(tf.layers.dense(inputs=self.scalarInput, units=50))

        self.dense2 = tf.nn.relu(tf.layers.dense(inputs=self.dense1, units=50))

        self.streamAC, self.streamVC = tf.split(self.dense2, 2, 1)
        #print("streamAC:", self.streamAC.shape)
        #print("streamVC:", self.streamVC.shape)

        self.streamA = self.streamAC
        self.streamV = self.streamVC
        #print("streamA:", self.streamA.shape)
        #print("streamV:", self.streamV.shape)

        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([int(self.streamA.shape[1]), n_actions]))
        self.VW = tf.Variable(xavier_init([int(self.streamV.shape[1]), 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, n_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)