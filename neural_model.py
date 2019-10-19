from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

classes = {"list": 0, "factoid": 1, "summary": 2, "yesno": 3}


def get_neural_model():
    model = Sequential()
    model.add(Dense(768, input_dim=1024, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
    model.add(Dropout(0.25))
    model.add(Dense(len(classes)))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.01)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model


def to_categorical(y):
    y_encoded = np.zeros((len(y), len(classes)))
    for idx, label in np.ndenumerate(y):
        sample_index = idx[0]
        one_hot_encoding_index = classes[label]
        y_encoded[sample_index, one_hot_encoding_index] = 1
    return y_encoded


def train(model: Model, train_set_x, train_set_y, batch_size: int, epochs: int, validation_split: float=0.0):
    train_set_y = to_categorical(train_set_y)
    return model.fit(train_set_x, train_set_y,
                     batch_size=batch_size, epochs=epochs,
                     verbose=2, validation_split=validation_split)


def evaluate(model: Model, test_set_x, test_set_y, batch_size: int):
    test_set_y = to_categorical(test_set_y)
    loss, accuracy = model.evaluate(test_set_x, test_set_y, batch_size=batch_size, verbose=2)
    return loss, accuracy


def plot_history(history):
    # Accuracy plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/training_accuracy.pdf')
    plt.close()
    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/training_loss.pdf')
