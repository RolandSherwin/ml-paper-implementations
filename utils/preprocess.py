import tensorflow as tf
import numpy as np


class TFDS():
    """Different preprocessing applied to a Tensorflow dataset. Needs to be applied to
    non-batched tdfs
    """

    def __init__(self) -> None:
        pass

    # adding the decorator to suppress warning: "Autograph could not transform the method"
    # caused by having variabled inside the _resizer function, if values are hard coded,
    # then there is no error.
    @tf.autograph.experimental.do_not_convert
    def _resizer(self, image, label):
        """This is the function thats mapped to a tdfs. Should not be called
        directly
        """
        if len(self._resizer_shape) == 2:
            image = tf.image.resize_with_pad(image,
                                             self._resizer_shape[0],
                                             self._resizer_shape[1],
                                             method=self._resizer_method)

        elif len(self._resizer_shape) == 3:
            image = tf.image.resize_with_pad(image,
                                             self._resizer_shape[0],
                                             self._resizer_shape[1],
                                             method=self._resizer_method)

        return image, label

    def resizer(self, shape, method='bilinear'):
        """Returns a function that can be mapped to a tdfs.
        """
        self._resizer_shape = shape
        self._resizer_method = method

        return self._resizer
