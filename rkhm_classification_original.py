# A demonstration script for classification with MNIST:
# "Deep Learning with Kernels through RKHM and the Perron-Frobenius Operator".
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
import numpy as np
from utils import tf_kron, one_hot_encoding, init_kernel_weights

d = 28  # We focus on the C*-algebra of d by d matrices and its C*-subalgebras
n_train = 200  # number of training samples
n_kernel = 40  # number of samples used to construct the represntation space
n_test = 1000  # number of test samples
epochs = 1000  # number of epochs
c = 0.001  # parameter in the Laplacian kernel
n_layers = 2  # number of the layers
lam1 = 1  # Perron-Frobenius regularlization parameter
lam2 = 0.001  # regularlization parameter for ||f_L||
dim = np.array([7, 4])  # dimension of blocks for each layer
dim1 = np.array(
    [2, 4]
)  # dimension of blocks of the parameter a_j for each layer j

ind = np.arange(0, n_kernel, 1, dtype=np.int32)


def C_algebra(d: int, L, dim1):
    a = []
    j = 0
    for j in range(L):
        a.append(np.zeros((d, d)))
        for i in range(int(d / dim1[j])):
            a[j][
                dim1[j] * i : dim1[j] * (i + 1), dim1[j] * i : dim1[j] * (i + 1)
            ] = np.ones((dim1[j], dim1[j]))
        a[j] = tf.constant(a[j], tf.float32)
    return a


block_matrix_algebra = C_algebra(d, n_layers, dim1)


class MLP_Classifier(Model):
    def __init__(self, n_layers: int = 2):
        super(MLP_Classifier, self).__init__()
        self.model = models.Sequential()
        for _ in range(n_layers - 1):
            self.model.add(layers.Dense(84, activation="sigmoid"))
        self.model.add(layers.Dense(10, activation="softmax"))

    def call(self, x):
        output = self.model(x)
        return output


@tf.function
def frobenius_loss(y, y_pred, c1, Gpre, GGtmp, n_train: int):
    reg1 = lam1 * (
        tf.norm(Gpre[n_layers - 1], 2)
        + tf.norm(
            tf.linalg.solve(
                0.01 * tf.eye(n_kernel) + Gpre[n_layers - 1],
                tf.eye(n_kernel),
            ),
            2,
        )
    )
    reg2 = lam2 * tf.norm(
        tf.matmul(
            tf.matmul(
                tf.transpose(c1[n_layers - 1], (1, 0)),
                tf_kron(GGtmp, tf.eye(d)),
            ),
            c1[n_layers - 1],
        ),
        2,
    )

    reg = reg1 + reg2

    loss = tf.norm(tf.matmul(tf.transpose(y - y_pred), y_pred - y), 2) / n_train

    lossreg = loss + reg
    return lossreg


@tf.function
def opti(
    y,
    c1,
    dense,
    optimizer,
    kernel_gram,
    Gtest,
    Gtmp,
    n_train: int,
    n_test: int,
    algebra_matrix,
):
    with tf.GradientTape(persistent=True) as tape:
        # Ensure dense.trainable_variables are tf.Variables
        trainable_vars = dense.trainable_variables

        ydata = tf.matmul(kernel_gram, c1[0])
        ytestdata = tf.matmul(Gtest, c1[0])
        Gpre = []
        GGtmp = tf.gather(Gtmp, indices=ind)
        Gpre.append(GGtmp)

        features = ydata[ind[0] * d : (ind[0] + 1) * d, :]
        for i in range(n_kernel - 1):
            features = tf.concat(
                [features, ydata[ind[i + 1] * d : (ind[i + 1] + 1) * d, :]],
                axis=0,
            )

        for j in range(0, n_layers - 1, 1):
            ytmp = tf.matmul(ydata, algebra_matrix[j + 1])
            ytesttmp = tf.matmul(ytestdata, algebra_matrix[j + 1])
            ftmp = tf.matmul(features, algebra_matrix[j + 1])
            tmp1 = tf.reshape(ytmp, [1, n_train, d * d])
            tmp1test = tf.reshape(ytesttmp, [1, n_test, d * d])
            tmpf = tf.reshape(ftmp, [1, n_kernel, d * d])
            for i in range(n_kernel - 1):
                tmp1 = tf.concat(
                    [tmp1, tf.reshape(ytmp, [1, n_train, d * d])], axis=0
                )
            for i in range(n_kernel - 1):
                tmp1test = tf.concat(
                    [tmp1test, tf.reshape(ytesttmp, [1, n_test, d * d])],
                    axis=0,
                )
            for i in range(n_test - 1):
                tmpf = tf.concat(
                    [tmpf, tf.reshape(ftmp, [1, n_kernel, d * d])], axis=0
                )

            gaussian_kernel_matrix = tf.math.exp(
                -c
                * tf.reduce_sum(
                    abs(tf.transpose(tmp1, (1, 0, 2)) - tmpf[0:n_train, :, :]),
                    axis=2,
                )
            )
            GGtmp = tf.gather(gaussian_kernel_matrix, indices=ind)
            gaussian_kernel_matrix = tf_kron(
                gaussian_kernel_matrix, tf.eye(d)
            ) * tf.matmul(ydata, tf.transpose(features, (1, 0)))

            Gpre.append(GGtmp)
            GGtest = tf.math.exp(
                -c
                * tf.reduce_sum(
                    abs(tf.transpose(tmp1test, (1, 0, 2)) - tmpf), axis=2
                )
            )
            GGtest = tf_kron(GGtest, tf.eye(d)) * tf.matmul(
                ytestdata, tf.transpose(features, (1, 0))
            )

            ydata = tf.matmul(gaussian_kernel_matrix, c1[j + 1])
            ytestdata = tf.matmul(GGtest, c1[j + 1])
            features = ydata[ind[0] * d : (ind[0] + 1) * d, :]
            for i in range(n_kernel - 1):
                features = tf.concat(
                    [features, ydata[ind[i + 1] * d : (ind[i + 1] + 1) * d, :]],
                    axis=0,
                )

        y_pred_train = dense.call(tf.reshape(ydata, [n_train, d * d]))
        y_pred_test = dense.call(tf.reshape(ytestdata, [n_test, d * d]))

        loss = frobenius_loss(y, y_pred_train, c1, Gpre, GGtmp, n_train)

        indx = tf.math.argmax(y_pred_train, axis=1)
        indx1 = tf.math.argmax(y, axis=1)
        acc = 1 - tf.math.count_nonzero(indx - indx1) / len(indx)

        indx = tf.math.argmax(y_pred_test, axis=1)
        indx1 = tf.math.argmax(labeltest, axis=1)
        acctest = 1 - tf.math.count_nonzero(indx - indx1) / len(indx)

        grad = tape.gradient(loss, c1 + trainable_vars)
        optimizer.apply_gradients(zip(grad, c1 + trainable_vars))

        return acc, acctest


def prepare_data(
    d: int,
    n_train: int,
    n_kernel: int,
    n_test: int,
    c,
    n_layers: int,
    dim,
    ind,
    algebra_matrix,
    X,
    y,
    n_classes=10,
):
    features = np.zeros((n_kernel, d, d))

    X_train = X[0:n_train, :, :]

    one_hot = one_hot_encoding(y, n_classes)

    ymat = one_hot[0:n_train, :]
    y = tf.constant(ymat, dtype=tf.float32)
    X_train = tf.constant(X_train, dtype=tf.float32)

    X_test = X[n_train : n_train + n_test, :, :]
    y_test = one_hot[n_train : n_train + n_test, :]

    X_test = tf.constant(X_test, dtype=tf.float32)
    y_test = tf.constant(y_test, dtype=tf.float32)

    for i in range(n_kernel):
        features[i, :, :] = X_train[ind[i], :, :]
    features = tf.constant(features, dtype=tf.float32)

    X_train = tf.reshape(X_train, [n_train * d, d])
    X_test = tf.reshape(X_test, [n_test * d, d])
    features = X[ind[0] * d : (ind[0] + 1) * d, :]
    for i in range(n_kernel - 1):
        features = tf.concat(
            [features, X[ind[i + 1] * d : (ind[i + 1] + 1) * d, :]], axis=0
        )

    kernel_gram, Gtmp, gram_test = _init_laplacian_kernel(
        d,
        n_train,
        n_kernel,
        n_test,
        c,
        ind,
        algebra_matrix,
        X,
        features,
        X_test,
    )

    c1 = init_kernel_weights(d, n_kernel, n_layers, dim)
    return y, y_test, kernel_gram, Gtmp, gram_test, c1


def _init_laplacian_kernel(
    d, n_train, n_kernel, n_test, c, ind, algebra_matrix, X, features, ytestdata
):
    # Project the data into the first input algebra matrix
    X_projected = tf.matmul(X, algebra_matrix[0])
    ytesttmp = tf.matmul(ytestdata, algebra_matrix[0])
    ftmp = tf.matmul(features, algebra_matrix[0])
    tmp1 = tf.reshape(X_projected, [1, n_train, d * d])
    tmp1test = tf.reshape(ytesttmp, [1, n_test, d * d])
    tmpf = tf.reshape(ftmp, [1, n_kernel, d * d])
    for i in range(n_kernel - 1):
        tmp1 = tf.concat(
            [tmp1, tf.reshape(X_projected, [1, n_train, d * d])], axis=0
        )
    for i in range(n_kernel - 1):
        tmp1test = tf.concat(
            [tmp1test, tf.reshape(ytesttmp, [1, n_test, d * d])], axis=0
        )
    for i in range(n_test - 1):
        tmpf = tf.concat([tmpf, tf.reshape(ftmp, [1, n_kernel, d * d])], axis=0)

    kernel_gram = tf.math.exp(
        -c
        * tf.reduce_sum(
            abs(tf.transpose(tmp1, (1, 0, 2)) - tmpf[0:n_train, :, :]), axis=2
        )
    )
    Gtmp = tf.gather(kernel_gram, indices=ind)

    gram_test = tf.math.exp(
        -c
        * tf.reduce_sum(abs(tf.transpose(tmp1test, (1, 0, 2)) - tmpf), axis=2)
    )
    kernel_gram = tf_kron(kernel_gram, tf.eye(d)) * tf.matmul(
        X, tf.transpose(features, (1, 0))
    )
    gram_test = tf_kron(gram_test, tf.eye(d)) * tf.matmul(
        ytestdata, tf.transpose(features, (1, 0))
    )

    return kernel_gram, Gtmp, gram_test


if __name__ == "__main__":

    # Import MNIST data
    mnist = tf.keras.datasets.mnist
    (ydata, y), _ = mnist.load_data()
    ydata = ydata / np.float64(255.0)

    print("Shape of the data:", ydata.shape)

    y, labeltest, kernel_gram, Gtmp, Gtest, kernel_weights = prepare_data(
        d,
        n_train,
        n_kernel,
        n_test,
        c,
        n_layers,
        dim,
        ind,
        block_matrix_algebra,
        ydata,
        y,
    )

    dense = MLP_Classifier(d)
    dense.model(tf.zeros((1, d * d), dtype=tf.float32))
    opt = tf.keras.optimizers.Adam(1e-3)
    print("Start training with model :", n_layers, "layers")

    for epoch in range(1, epochs + 1, 1):

        acc, acctest = opti(
            y=y,
            c1=kernel_weights,
            dense=dense,
            optimizer=opt,
            kernel_gram=kernel_gram,
            Gtest=Gtest,
            Gtmp=Gtmp,
            n_train=n_train,
            n_test=n_test,
            algebra_matrix=block_matrix_algebra,
        )

        print(
            epoch,
            "Training accuracy:",
            acc.numpy(),
            "Test accuracy:",
            acctest.numpy(),
            flush=True,
        )
