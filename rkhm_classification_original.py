# A demonstration script for classification with MNIST:
# "Deep Learning with Kernels through RKHM and the Perron-Frobenius Operator".
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
import numpy as np
from utils import tf_kron, one_hot_encoding, init_kernel_weights, evaluate

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
dim1 = np.array([2, 4])  # dimension of blocks of the parameter a_j for each layer j

ind = np.arange(0, n_kernel, 1, dtype=np.int32)


def initialize_block_matrix_layers(d, dim1):
    layer_block_matrices = []
    for block_size in dim1:
        block_mat = np.zeros((d, d))
        n_blocks = int(d / block_size)
        for i in range(n_blocks):
            block_mat[
                block_size * i : block_size * (i + 1),
                block_size * i : block_size * (i + 1),
            ] = np.ones((block_size, block_size))
        block_mat = tf.constant(block_mat, tf.float32)
        layer_block_matrices.append(block_mat)
    return layer_block_matrices


layer_block_matrices = initialize_block_matrix_layers(d, dim1)


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
def frobenius_loss(y, y_pred, c1, Gpre, GGtmp):
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
    n = y.shape[0]
    loss = tf.norm(tf.matmul(tf.transpose(y - y_pred), y_pred - y), 2) / n

    lossreg = loss + reg
    return lossreg


@tf.function
def opti(
    y_train,
    y_test,
    y_pred_train,
    y_pred_test,
    c1,
    dense,
    optimizer,
    n_train: int,
):
    with tf.GradientTape(persistent=True) as tape:
        # Ensure dense.trainable_variables are tf.Variables
        trainable_vars = dense.trainable_variables

        loss = frobenius_loss(y_train, y_pred_train, c1, Gpre, GGtmp)

        training_accuracy = evaluate(y_train, y_pred_train)
        test_accuracy = evaluate(y_test, y_pred_test)

        grad = tape.gradient(loss, c1 + trainable_vars)
        optimizer.apply_gradients(zip(grad, c1 + trainable_vars))

        return training_accuracy, test_accuracy


@tf.function
def forward(c1, dense, kernel_gram, Gtest, Gtmp, n_train, n_test, layer_block_matrices):
    z_train = tf.matmul(kernel_gram, c1[0])
    z_test = tf.matmul(Gtest, c1[0])
    Gpre = []
    GGtmp = tf.gather(Gtmp, indices=ind)
    Gpre.append(GGtmp)

    features = z_train[ind[0] * d : (ind[0] + 1) * d, :]
    for i in range(n_kernel - 1):
        features = tf.concat(
            [features, z_train[ind[i + 1] * d : (ind[i + 1] + 1) * d, :]],
            axis=0,
        )

    for j in range(0, n_layers - 1, 1):
        ytmp = tf.matmul(z_train, layer_block_matrices[j + 1])
        ytesttmp = tf.matmul(z_test, layer_block_matrices[j + 1])
        ftmp = tf.matmul(features, layer_block_matrices[j + 1])
        tmp1 = tf.reshape(ytmp, [1, n_train, d * d])
        tmp1test = tf.reshape(ytesttmp, [1, n_test, d * d])
        tmpf = tf.reshape(ftmp, [1, n_kernel, d * d])
        for i in range(n_kernel - 1):
            tmp1 = tf.concat([tmp1, tf.reshape(ytmp, [1, n_train, d * d])], axis=0)
        for i in range(n_kernel - 1):
            tmp1test = tf.concat(
                [tmp1test, tf.reshape(ytesttmp, [1, n_test, d * d])],
                axis=0,
            )
        for i in range(n_test - 1):
            tmpf = tf.concat([tmpf, tf.reshape(ftmp, [1, n_kernel, d * d])], axis=0)

        gaussian_kernel_matrix = tf.math.exp(
            -c
            * tf.reduce_sum(
                abs(tf.transpose(tmp1, (1, 0, 2)) - tmpf[0:n_train, :, :]),
                axis=2,
            )
        )
        GGtmp = tf.gather(gaussian_kernel_matrix, indices=ind)
        gaussian_kernel_matrix = tf_kron(gaussian_kernel_matrix, tf.eye(d)) * tf.matmul(
            z_train, tf.transpose(features, (1, 0))
        )

        Gpre.append(GGtmp)
        GGtest = tf.math.exp(
            -c * tf.reduce_sum(abs(tf.transpose(tmp1test, (1, 0, 2)) - tmpf), axis=2)
        )
        GGtest = tf_kron(GGtest, tf.eye(d)) * tf.matmul(
            z_test, tf.transpose(features, (1, 0))
        )

        z_train = tf.matmul(gaussian_kernel_matrix, c1[j + 1])
        z_test = tf.matmul(GGtest, c1[j + 1])
        features = z_train[ind[0] * d : (ind[0] + 1) * d, :]
        for i in range(n_kernel - 1):
            features = tf.concat(
                [features, z_train[ind[i + 1] * d : (ind[i + 1] + 1) * d, :]],
                axis=0,
            )

    # Train the weights of the kernel
    y_pred_train = dense(tf.reshape(z_train, [n_train, d * d]))
    y_pred_test = dense(tf.reshape(z_test, [n_test, d * d]))
    return y_pred_train, y_pred_test, Gpre, GGtmp


def train_test_split(
    d: int,
    n_train: int,
    n_kernel: int,
    n_test: int,
    ind,
    imgs,
    labels,
    n_classes=10,
):
    features = np.zeros((n_kernel, d, d))

    X_train = imgs[0:n_train, :, :]

    one_hot = one_hot_encoding(labels, n_classes)

    ymat = one_hot[0:n_train, :]
    y_train = tf.constant(ymat, dtype=tf.float32)
    X_train = tf.constant(X_train, dtype=tf.float32)

    X_test = imgs[n_train : n_train + n_test, :, :]
    y_test = one_hot[n_train : n_train + n_test, :]

    X_test = tf.constant(X_test, dtype=tf.float32)
    y_test = tf.constant(y_test, dtype=tf.float32)

    for i in range(n_kernel):
        features[i, :, :] = X_train[ind[i], :, :]
    features = tf.constant(features, dtype=tf.float32)

    X_train = tf.reshape(X_train, [n_train * d, d])
    X_test = tf.reshape(X_test, [n_test * d, d])

    return X_train, X_test, y_train, y_test


def init_laplacian_kernel(
    imgs,
    X_train,
    d,
    n_train,
    n_kernel,
    n_test,
    c,
    ind,
    algebra_matrix,
    features,
    ytestdata,
):
    features = imgs[ind[0] * d : (ind[0] + 1) * d, :]
    for i in range(n_kernel - 1):
        features = tf.concat(
            [features, imgs[ind[i + 1] * d : (ind[i + 1] + 1) * d, :]], axis=0
        )
    # Project the data into the first input algebra matrix
    X_projected = tf.matmul(X, algebra_matrix[0])
    ytesttmp = tf.matmul(ytestdata, algebra_matrix[0])
    ftmp = tf.matmul(features, algebra_matrix[0])
    tmp1 = tf.reshape(X_projected, [1, n_train, d * d])
    tmp1test = tf.reshape(ytesttmp, [1, n_test, d * d])
    tmpf = tf.reshape(ftmp, [1, n_kernel, d * d])
    for i in range(n_kernel - 1):
        tmp1 = tf.concat([tmp1, tf.reshape(X_projected, [1, n_train, d * d])], axis=0)
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
        -c * tf.reduce_sum(abs(tf.transpose(tmp1test, (1, 0, 2)) - tmpf), axis=2)
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
    (imgs, labels), _ = mnist.load_data()
    imgs = imgs / np.float64(255.0)

    print("Shape of the data:", imgs.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        d,
        n_train,
        n_kernel,
        n_test,
        c,
        n_layers,
        dim,
        ind,
        layer_block_matrices,
        imgs,
        labels,
    )

    kernel_gram, Gtmp, G_test = init_laplacian_kernel(
        imgs,
        X_train,
        d,
        n_train,
        n_kernel,
        n_test,
        c,
        ind,
        layer_block_matrices,
        X_test,
    )

    kernel_weights = init_kernel_weights(d, n_kernel, n_layers, dim)

    dense = MLP_Classifier(n_layers=d)
    optimizer = tf.keras.optimizers.Adam(1e-3)
    print("Start training with model :", n_layers, "layers")

    for epoch in range(epochs):

        y_pred_train, y_pred_test, Gpre, GGtmp = forward(
            kernel_weights,
            dense,
            kernel_gram,
            G_test,
            Gtmp,
            n_train,
            n_test,
            layer_block_matrices,
        )

        acc, acctest = opti(
            y_train=y_train,
            y_test=y_test,
            c1=kernel_weights,
            dense=dense,
            optimizer=optimizer,
            kernel_gram=kernel_gram,
            Gtest=G_test,
            Gtmp=Gtmp,
            n_train=n_train,
            n_test=n_test,
            algebra_matrix=layer_block_matrices,
        )

        print(
            epoch,
            "Training accuracy:",
            acc.numpy(),
            "Test accuracy:",
            acctest.numpy(),
            flush=True,
        )
