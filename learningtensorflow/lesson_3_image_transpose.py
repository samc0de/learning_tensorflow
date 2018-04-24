'''Lesson 3.

Load an image using matplotlib, transpose it using tensorflow, and display.
'''
from matplotlib import image  # SEE ALSO: tf.image
from matplotlib import pyplot
import tensorflow as tf
import argparse


# Assumes an image to be present at /tmp/MarshOrchid.jpg.
# Download from https://learningtensorflow.com/images/MarshOrchid.jpg.
PATH = '/tmp/MarshOrchid.jpg'


def transpose(img_array):
  '''Transpose and return input image array.'''
  transposed = tf.transpose(img_array, perm=[1, 0, 2])
  with tf.Session() as sess:
    transposed_img = sess.run(transposed)
  return transposed_img


def reverse(img_array):
  '''Reverse and return the input image.

  Allow passing in axis later.
  '''
  # Same name as funcname! Can't use reversed to avoid masking builtin.
  # reverse = tf.reverse(img_array, dims=[True, False, False])
  reverse = tf.reverse(img_array, axis=[0])
  with tf.Session() as sess:
    reverse = sess.run(reverse)
  return reverse


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--path', '-p', help='Path of image to be transposed.', default=PATH)
  parser.add_argument('--action', '-a', choices=('transpose', 'reverse'),
                      default='transpose', help='Action to take on image.')
  arguments = parser.parse_args()
  path = arguments.path
  img_array = image.imread(path)
  if arguments.action == 'reverse':
    image = reverse(img_array)
  else:
    image = transpose(img_array)

  pyplot.imshow(image)
  pyplot.show()
