import tensorflow as tf

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_dense = tf.keras.layers.Dense(channels // 8, activation='relu')
        self.dense_avg = tf.keras.layers.Dense(channels, activation='sigmoid')
        self.dense_max = tf.keras.layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        avg_out = self.dense_avg(self.shared_dense(avg_pool))
        max_out = self.dense_max(self.shared_dense(max_pool))
        weights = tf.sigmoid(avg_out + max_out)
        return inputs * weights