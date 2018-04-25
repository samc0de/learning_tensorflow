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


def transpose(img_array, axis):
  '''Transpose and return input image array.'''
  # Find a better way for this!!!
  # Doesn't work!!! Image(x, y, z), z must be a valid colour int base.
  # perm = [1, 0, 2] if axis < 2 else [2, 1, 0]
  perm = [1, 0, 2]
  transposed = tf.transpose(img_array, perm=perm)
  with tf.Session() as sess:
    transposed_img = sess.run(transposed)
  return transposed_img


def reverse(img_array, axis):
  '''Reverse and return the input image.

  Allow passing in axis later.
  '''
  # Same name as funcname! Can't use reversed to avoid masking builtin.
  # reverse = tf.reverse(img_array, dims=[True, False, False])
  reverse = tf.reverse(img_array, axis=[axis])
  with tf.Session() as sess:
    reverse = sess.run(reverse)
  return reverse


if __name__ == '__main__':
  # Have clearer options, maybe like -r(everse), -t(ranspose).
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--path', '-p', help='Path of image to be transposed.', default=PATH)
  # Better add a subcommand.
  parser.add_argument('--action', '-a', choices=('transpose', 'reverse'),
                      default='transpose', help='Action to take on image.')
  parser.add_argument('--axis', '-x', type=int, choices=[0, 1, 2],
                      help='Axis to take action along.', default=0)
  arguments = parser.parse_args()
  path = arguments.path
  img_array = image.imread(path)
  axis = arguments.axis
  if arguments.action == 'reverse':
    image = reverse(img_array, axis)
  else:
    image = transpose(img_array, axis)

  pyplot.imshow(image)
  pyplot.show()
