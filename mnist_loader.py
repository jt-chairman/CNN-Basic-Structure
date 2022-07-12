# Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np


def load_data():

    with gzip.open('../CNN-Basic-Structure/data/27x27.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='bytes')

    return training_data, validation_data, test_data


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_data = [(x, y) for x, y in zip(tr_d[0], tr_d[1])]
    validation_data = [(x, y) for x, y in zip(va_d[0], va_d[1])]
    test_data = [(x, y) for x, y in zip(te_d[0], te_d[1])]

    return training_data, validation_data, test_data


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
