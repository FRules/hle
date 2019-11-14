import csv
import numpy as np
from copy import deepcopy
import random
from nltk import word_tokenize
import time
import datetime
import pickle
import os

save_directory = "save/"


def split_dataset(train_set_x, train_set_y, ratio: float=0.2):
    length_train_set = len(train_set_x)
    length_test_set = int(length_train_set * ratio)
    indices = random.sample(range(length_train_set), length_test_set)
    test_set_x = deepcopy(train_set_x[indices])
    test_set_y = deepcopy(train_set_y[indices])
    train_set_x = np.delete(train_set_x, indices)
    train_set_y = np.delete(train_set_y, indices)

    assert length_train_set == len(train_set_x) + len(test_set_x)

    return train_set_x, train_set_y, test_set_x, test_set_y


def get_dataset(filename: str):
    # reads the train set and splits it into the data (x)
    # and their corresponding labels (y)
    with open(filename, mode='r') as file:
        reader = csv.reader(file, delimiter=';')
        x = []
        y = []
        for row in reader:
            document = row[0]
            label = row[1]
            document = preprocess_document(document)
            x.append(document)
            y.append(label)
        assert len(x) == len(y)
        # remove header
        return np.array(x[1:]), np.array(y[1:])


def preprocess_document(document: str) -> str:
    # Calls a number of methods that do some modifications
    # to the document.
    document = remove_special_characters(document)
    document = lowercase_document(document)

    # tokenization should be the last step here
    # document = tokenize_document(document)
    return document


def tokenize_document(document: str) -> str:
    return word_tokenize(document)


def lowercase_document(document: str) -> str:
    return document.lower()


def remove_special_characters(document: str) -> str:
    # removes some pre-defined characters from the document that
    # don't give the document any more meaning and are meant just
    # for structuring the document. It won't help the classifier.
    document = document.strip()
    characters_to_remove = ['?', '!', '\'', '\"', '(', ')', '{', '}', '[', ']', '´', '`', '^']
    for character_to_remove in characters_to_remove:
        document = document.replace(character_to_remove, '')
    return document


def get_distribution_of_classes(y) -> dict:
    # Calculates how the dataset is distributed
    # The goal is that we have a similar amount
    # of samples for every class
    distribution = {}
    for class_label in y:
        if class_label in distribution:
            distribution[class_label] += 1
        else:
            distribution[class_label] = 1
    return distribution


def get_properly_distributed_train_set(x, y, threshold: int=5):
    # Some classes can be not as good represented as others.
    # For example, the class "list" has just 485 entries,
    # but the class "yesno" has 616. In cases like this, we
    # copy random X entries from the "list" class so that
    # it has almost the same ratio as the "yesno" class.
    # The aim is that the class distribution is round-about
    # the same. It should be as close as the threshold is,
    # defaulting to 5 percent. That means, the formula is
    # (int)((619 / 100) * (100 - Threshold))
    distribution = get_distribution_of_classes(y)
    max_value_of_distribution = np.max(list(distribution.values()))
    minimum_samples = int((max_value_of_distribution / 100) * (100 - threshold))
    for key, value in distribution.items():
        if value < minimum_samples:
            new_x_entries, new_y_entries = upsample_class(key, minimum_samples - value, x, y)
            x = np.append(x, new_x_entries)
            y = np.append(y, new_y_entries)
    return shuffle_x_and_y(x, y)


def upsample_class(key: str, amount: int, x, y):
    indices = np.where(y == key)[0]
    new_x_entries = deepcopy(x[np.random.choice(indices, amount)])
    new_y_entries = np.full(amount, key)

    assert len(new_x_entries) == len(new_y_entries)
    return new_x_entries, new_y_entries


def shuffle_x_and_y(x, y):
    # When we added up-sampled information to the data, we
    # appended it at the end. It's not good for the classifier
    # so we shuffle the whole dataset
    combined = list(zip(x, y))
    np.random.shuffle(combined)

    x[:], y[:] = zip(*combined)
    assert len(x) == len(y)
    return x, y


def get_preprocessed_dataset(filename: str):
    x, y = get_dataset(filename)
    train_set_x, train_set_y, test_set_x, test_set_y = split_dataset(x, y)
    train_set_x, train_set_y = get_properly_distributed_train_set(train_set_x, train_set_y, threshold=3)
    assert len(train_set_x) == len(train_set_y)
    assert len(test_set_x) == len(test_set_y)

    return train_set_x, train_set_y, test_set_x, test_set_y


def get_timestamp() -> str:
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')


def save_preprocessed_data(train_set_x_embedded, train_set_y, test_set_x_embedded,
                           test_set_y, train_set_x=None, test_set_x=None):
    timestamp = get_timestamp()
    os.mkdir(save_directory + timestamp)
    pickle_out = open(save_directory + timestamp + "/train_set_x_embedded.pickle", "wb")
    pickle.dump(train_set_x_embedded, pickle_out)
    pickle_out.close()
    pickle_out = open(save_directory + timestamp + "/test_set_x_embedded.pickle", "wb")
    pickle.dump(test_set_x_embedded, pickle_out)
    pickle_out.close()
    pickle_out = open(save_directory + timestamp + "/train_set_y.pickle", "wb")
    pickle.dump(train_set_y, pickle_out)
    pickle_out.close()
    pickle_out = open(save_directory + timestamp + "/test_set_y.pickle", "wb")
    pickle.dump(test_set_y, pickle_out)
    pickle_out.close()

    if train_set_x is not None and test_set_x is not None:
        pickle_out = open(save_directory + timestamp + "/train_set_x.pickle", "wb")
        pickle.dump(train_set_x, pickle_out)
        pickle_out.close()
        pickle_out = open(save_directory + timestamp + "/test_set_x.pickle", "wb")
        pickle.dump(test_set_x, pickle_out)
        pickle_out.close()


def load_preprocessed_data(timestamp: str, restore_not_embedded_data: bool=False):
    pickle_in = open(save_directory + timestamp + "/train_set_x_embedded.pickle", "rb")
    train_set_x_embedded = pickle.load(pickle_in)
    pickle_in = open(save_directory + timestamp + "/test_set_x_embedded.pickle", "rb")
    test_set_x_embedded = pickle.load(pickle_in)
    pickle_in = open(save_directory + timestamp + "/train_set_y.pickle", "rb")
    train_set_y = pickle.load(pickle_in)
    pickle_in = open(save_directory + timestamp + "/test_set_y.pickle", "rb")
    test_set_y = pickle.load(pickle_in)

    if restore_not_embedded_data is True:
        pickle_in = open(save_directory + timestamp + "/train_set_x.pickle", "rb")
        train_set_x = pickle.load(pickle_in)
        pickle_in = open(save_directory + timestamp + "/test_set_x.pickle", "rb")
        test_set_x = pickle.load(pickle_in)
        return train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y, train_set_x, test_set_x
    else:
        return train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y
