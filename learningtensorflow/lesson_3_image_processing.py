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


# Defaults overlap with argparse args default system, but this is useful in
# inter function calls.
def transpose(img_array, diagonal=False):
  '''Transpose and return input image array.

  Args:
    img_array: Array representing the image to transpose.
    diagonal: Diagonal to transpose along, 0 (default) for primary, 1 for
      secondary.

  Returns:
    Array that can be displayed, for showing transpose along the given axis.
  '''
  if not diagonal:
    perm = [1, 0, 2]
    transposed = tf.transpose(img_array, perm=perm)
    with tf.Session() as sess:
      transposed_img = sess.run(transposed)
    return transposed_img

  # To flip along secondary axis, reverse image, flip along primary and again
  # reverse.
  reversed_image = reverse(img_array)
  reverse_flipped = transpose(reversed_image)
  return reverse(reverse_flipped)


def reverse(img_array, axis=0):
  '''Reverse and return the input image.

  Allow passing in axis later.
  '''
  # Same name as funcname! Can't use reversed to avoid masking builtin.
  # reverse = tf.reverse(img_array, dims=[True, False, False])
  reverse = tf.reverse(img_array, axis=[axis])
  with tf.Session() as sess:
    reverse = sess.run(reverse)
  return reverse


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--path', '-p', help='Path of image to be transposed.', default=PATH)
  # Better add a subcommand parser.
  subcommands = parser.add_subparsers(dest='image_action')

  transpose_parser = subcommands.add_parser(
      'transpose', help='Flip an image along one of its diagonals.')
  transpose_parser.add_argument('--diagonal', '-d', type=int, choices=[0, 1],
                      help='Diagonal to transpose along.', default=0)
  transpose_parser.set_defaults(func=transpose)

  reverse_parser = subcommands.add_parser('reverse', help='Flip an image.')
  # Better way to map to axis 1 and 0? Make arg -axis 1 with choices(0, 1)
  reverse_parser.add_argument(
      '--vertical', '-v', action='store_true',
      help='Whether to flip vertically.', default=False)
  reverse_parser.set_defaults(func=reverse)

  arguments = parser.parse_args()
  path = arguments.path
  img_array = image.imread(path)
  # image = arguments.func(img_array, arguments)
  if arguments.image_action == 'reverse':
    img = reverse(img_array, 0 if arguments.vertical else 1)
  else:
    img = transpose(img_array, arguments.diagonal)

  pyplot.imshow(img)
  pyplot.show()


# The main reasons to move to subcommands was the differring args. Now here we
# have only (0, 1) in both the cases. Also try moving back the implementation to
# just --action reverse --x 1.

if __name__ == '__main__':
  main()
