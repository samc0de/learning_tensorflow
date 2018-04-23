import numpy as np
import tensorflow as tf


data = np.random.randint(1000, size=10000)
x = tf.constant(data, dtype=np.float16, name='x')
y = tf.Variable(5 * x ** x - 3 * x + 15, name='y')

model = tf.global_variables_initializer()

with tf.Session() as sess:
  merged = tf.summary.merge_all()  # Unused var.
  writer = tf.summary.FileWriter('/tmp/learning_tensor_logs', graph=sess.graph)
  sess.run(model)
