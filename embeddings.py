import tensorflow_hub as hub
import tensorflow as tf
import numpy as np


def get_elmo():
    module_url = "https://tfhub.dev/google/elmo/2"
    return hub.load(module_url)


def elmo_vectors(elmo, x):
    x = tf.constant(x)
    embeddings = elmo.signatures['default'](x)["elmo"]
    return tf.math.reduce_mean(embeddings, 1)


def split_data_into_batches(train_set_x, test_set_x, batch_size=100):
    list_train = [train_set_x[i:i + batch_size] for i in range(0, train_set_x.shape[0], batch_size)]
    list_test = [test_set_x[i:i + batch_size] for i in range(0, test_set_x.shape[0], batch_size)]
    return list_train, list_test


def get_embeddings(train_set_x, test_set_x):
    elmo = get_elmo()
    batch_train_set_x, batch_test_set_x = split_data_into_batches(train_set_x, test_set_x, batch_size=100)
    elmo_train_set_x = [elmo_vectors(elmo, x) for x in batch_train_set_x]
    elmo_test_set_x = [elmo_vectors(elmo, x) for x in batch_test_set_x]
    elmo_train_set_x_concatenated = np.concatenate(elmo_train_set_x, axis=0)
    elmo_test_set_x_concatenated = np.concatenate(elmo_test_set_x, axis=0)
    return elmo_train_set_x_concatenated, elmo_test_set_x_concatenated
