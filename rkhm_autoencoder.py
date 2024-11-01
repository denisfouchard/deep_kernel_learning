# A demonstration script for deep RKHM autoencoder:
# "Deep Learning with Kernels through RKHM and the Perron-Frobenius Operator".

import numpy.random as nr
import tensorflow as tf
import numpy as np
import time

d = 10  # We focus on the C*-algebra of d by d matrices and its C*-subalgebras
datanum = 10  # number of training samples
tdatanum = 1000  # number of test samples
epochs = 4000  # number of epochs
c = 0.001  # parameter in the Laplacian kernel
L = 3  # number of the layers

dim = np.array([1, 2, d])  # dimension of blocks for each layer


def tf_kron(a, b):
    a_shape = [a.shape[0], a.shape[1]]
    b_shape = [b.shape[0], b.shape[1]]
    return tf.reshape(
        tf.reshape(a, [a_shape[0], 1, a_shape[1], 1])
        * tf.reshape(b, [1, b_shape[0], 1, b_shape[1]]),
        [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]],
    )


def matchange(x, m, datanum, d):
    return tf.reshape(x, [1, datanum, m * d])


@tf.function
def opti(G, Gtmp, xdata, c1, opt, Gtest, xtestdata):
    with tf.GradientTape(persistent=True) as tape:
        ydata = tf.matmul(G, c1[0])
        ytestdata = tf.matmul(Gtest, c1[0])

        for j in range(L - 1):
            tmp1 = matchange(ydata, d, datanum, d)
            tmp1test = matchange(ytestdata, d, tdatanum, d)
            for i in range(tdatanum - 1):
                tmp1 = tf.concat(
                    [tmp1, matchange(ydata, d, datanum, d)], axis=0
                )
            for i in range(datanum - 1):
                tmp1test = tf.concat(
                    [tmp1test, matchange(ytestdata, d, tdatanum, d)], axis=0
                )

            GG = tf.math.exp(
                -c
                * tf.reduce_sum(
                    abs(
                        tf.transpose(tmp1[0:datanum, :, :], (1, 0, 2))
                        - tmp1[0:datanum, :, :]
                    ),
                    axis=2,
                )
            )
            GGtmp = GG
            GG = tf_kron(GG, tf.eye(d))
            GGtest = tf.math.exp(
                -c
                * tf.reduce_sum(
                    abs(tf.transpose(tmp1test, (1, 0, 2)) - tmp1), axis=2
                )
            )
            GGtest = tf_kron(GGtest, tf.eye(d))
            ydata = tf.matmul(GG, c1[j + 1])
            ytestdata = tf.matmul(GGtest, c1[j + 1])

        loss = (
            tf.norm(
                tf.matmul(tf.transpose(ydata - xdata, (1, 0)), ydata - xdata), 2
            )
            / datanum
        )
        testloss = (
            tf.norm(
                tf.matmul(
                    tf.transpose(ytestdata - xtestdata, (1, 0)),
                    ytestdata - xtestdata,
                ),
                2,
            )
            / tdatanum
        )
        grad = tape.gradient(loss, c1)
        opt.apply_gradients(zip(grad, c1))
        return abs(testloss - loss)


if __name__ == "__main__":
    nr.seed(0)
    xdata = 0.1 * nr.randn(d, datanum)
    p = nr.randn(d * d, d)
    xdata = (
        (0.1 * p.dot(xdata)) ** 2 + 0.0001 * nr.randn(d * d, datanum)
    ).reshape([d, datanum * d])
    xdata = tf.constant(xdata.T, dtype=tf.float32)

    xtestdata = 0.1 * nr.randn(d, tdatanum)
    xtestdata = ((0.1 * p.dot(xtestdata)) ** 2).reshape([d, tdatanum * d])
    xtestdata = tf.constant(xtestdata.T, dtype=tf.float32)

    tmp1 = matchange(xdata, d, datanum, d)
    tmp1test = matchange(xtestdata, d, tdatanum, d)
    for i in range(tdatanum - 1):
        tmp1 = tf.concat([tmp1, matchange(xdata, d, datanum, d)], axis=0)
    for i in range(datanum - 1):
        tmp1test = tf.concat(
            [tmp1test, matchange(xtestdata, d, tdatanum, d)], axis=0
        )

    G = tf.math.exp(
        -c
        * tf.reduce_sum(
            abs(
                tf.transpose(tmp1[0:datanum, :, :], (1, 0, 2))
                - tmp1[0:datanum, :, :]
            ),
            axis=2,
        )
    )
    Gtmp = G
    G = tf_kron(G, tf.eye(d))
    Gtest = tf.math.exp(
        -c
        * tf.reduce_sum(abs(tf.transpose(tmp1test, (1, 0, 2)) - tmp1), axis=2)
    )
    Gtest = tf_kron(Gtest, tf.eye(d))

    nr.seed(int(time.time()))

    c1 = []
    for k in range(L):
        c1.append(np.zeros((datanum * d, d)))
        for i in range(datanum):
            for j in range(int(d / dim[k])):
                c1[k][
                    i * dim[k] : (i + 1) * dim[k], j * dim[k] : (j + 1) * dim[k]
                ] = 0.1 * np.ones((dim[k], dim[k])) + 0.05 * nr.randn(
                    dim[k], dim[k]
                )
        c1[k] = tf.Variable(c1[k], dtype=tf.float32)

    opt = tf.keras.optimizers.SGD(1e-4)

    for epoch in range(1, epochs + 1, 1):
        error = opti(G, Gtmp, xdata, c1, opt, Gtest, xtestdata)
        if epoch % 10 == 0:
            print(epoch, "Generalization Error: ", error.numpy(), flush=True)
