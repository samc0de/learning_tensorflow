'''Lesson 3.

Load an image using matplotlib, transpose it using tensorflow, and display.
'''
from matplotlib import image
from matplotlib import pyplot
import tensorflow as tf
import argparse


# Assumes an image to be present at /tmp/MarshOrchid.jpg.
# Download from https://learningtensorflow.com/images/MarshOrchid.jpg.
PATH = '/tmp/MarshOrchid.jpg'

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--path', '-p', help='Path of image to be transposed.', default=PATH)
  arguments = parser.parse_args()
  path = arguments.path
  img_array = image.imread(path)
  transposed = tf.transpose(img_array, perm=[1, 0, 2])
  with tf.Session() as sess:
    transposed_img = sess.run(transposed)

  pyplot.imshow(transposed_img)
  pyplot.show()
