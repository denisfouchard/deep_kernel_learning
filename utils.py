import tensorflow as tf
import numpy as np


def tf_kron(a, b):
    """
    Computes the Kronecker product of two 2D tensors.

    Args:
        a (tf.Tensor): A 2D tensor with shape [m, n].
        b (tf.Tensor): A 2D tensor with shape [p, q].

    Returns:
        tf.Tensor: A 2D tensor with shape [m*p, n*q] representing the Kronecker product of `a` and `b`.

    Example:
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[0, 5], [6, 7]])
        result = tf_kron(a, b)
        # result is:
        # [[ 0,  5,  0, 10],
        #  [ 6,  7, 12, 14],
        #  [ 0, 15,  0, 20],
        #  [18, 21, 24, 28]]
    """
    a_shape = [a.shape[0], a.shape[1]]
    b_shape = [b.shape[0], b.shape[1]]
    return tf.reshape(
        tf.reshape(a, [a_shape[0], 1, a_shape[1], 1])
        * tf.reshape(b, [1, b_shape[0], 1, b_shape[1]]),
        [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]],
    )


def one_hot_encoding(y, n_classes):
    # Convert labels to one-hot encoding
    one_hot_labels = np.zeros((len(y), n_classes))
    index = [[] for i in range(n_classes)]
    for i in range(n_classes):
        index[i] = list(np.where(y == i))[0]
        one_hot_labels[index[i], i] = np.ones(len(index[i]))
