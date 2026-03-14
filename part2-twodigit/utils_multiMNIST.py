import gzip
import pickle
import numpy as np

num_classes = 10
img_rows, img_cols = 42, 28


def get_data(path_to_data_dir, use_mini_dataset):
    """Load multi-digit MNIST dataset from gzipped pickle files."""
    exten = '_mini' if use_mini_dataset else ''

    with gzip.open(path_to_data_dir + 'train_multi_digit' + exten + '.pkl.gz', 'rb') as f:
        X_train = pickle.load(f, encoding='latin1')
    X_train = np.reshape(X_train, (len(X_train), 1, img_rows, img_cols))

    with gzip.open(path_to_data_dir + 'test_multi_digit' + exten + '.pkl.gz', 'rb') as f:
        X_test = pickle.load(f, encoding='latin1')
    X_test = np.reshape(X_test, (len(X_test), 1, img_rows, img_cols))

    with gzip.open(path_to_data_dir + 'train_labels' + exten + '.txt.gz', 'rb') as f:
        y_train = np.loadtxt(f)

    with gzip.open(path_to_data_dir + 'test_labels' + exten + '.txt.gz', 'rb') as f:
        y_test = np.loadtxt(f)

    return X_train, y_train, X_test, y_test
