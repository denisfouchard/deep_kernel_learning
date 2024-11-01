# Setup

To run the code, please install the following packages with Python 3.9:
- numpy
- tensorflow 2.6
- idx2numpy


# Data

For the experiment with MNIST, we need the dataset "train-images.idx3-ubyte" and its label "train-labels.idx1-ubyte", which can be downloaded from "http://yann.lecun.com/exdb/mnist/". Download these files to the same directry as that containing codes.


# Running the code

- For deep RKHM autoencoder (the first experiment in Section 7), run "python rkhm_autoencoder.py" 
- For regression problem (the second experiment in Section 7), run "python rkhm_regress.py".
- For classification with MNIST, run "python rkhm_classification.py".
