import os

# Disabling the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import DataLoader

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()

print(sess.run(hello))

dataLoader = DataLoader.DataLoader()
eyes = dataLoader.loadData()
