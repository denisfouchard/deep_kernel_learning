import tensorflow as tf
import numpy as np
import numpy.random as nr


def evaluate(y_true, y_pred_train, y_pred_test):
    indx = tf.math.argmax(y_pred_train, axis=1)
    indx1 = tf.math.argmax(y_true, axis=1)
    training_accuracy = 1 - tf.math.count_nonzero(indx - indx1) / len(indx)

    indx = tf.math.argmax(y_pred_test, axis=1)
    indx1 = tf.math.argmax(labeltest, axis=1)
    test_accuracy = 1 - tf.math.count_nonzero(indx - indx1) / len(indx)
    return training_accuracy, test_accuracy


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

    return one_hot_labels


def init_kernel_weights(d, n_kernel, dim):
    layer_weights = []  # Called c1 in the original code
    for weight_size in dim:
        weights_k = np.zeros((d * n_kernel, d))
        for i in range(n_kernel):
            n_blocks = int(d / weight_size)
            for j in range(n_blocks):
                weights_k[
                    i * d + j * weight_size : i * d + (j + 1) * weight_size,
                    j * weight_size : (j + 1) * weight_size,
                ] = 0.1 * nr.randn(weight_size, weight_size)
        weights_k = tf.Variable(weights_k, dtype=tf.float32)
        layer_weights.append(weights_k)
    return layer_weights
