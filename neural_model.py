from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

classes = ["list", "factoid", "summary", "yesno"]


def get_neural_model(compile_model=True):
    model = Sequential()
    model.add(Dense(768, input_dim=1024, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
    model.add(Dropout(0.25))
    model.add(Dense(len(classes)))
    model.add(Activation("softmax"))

    if compile_model is True:
        sgd = SGD(lr=0.01)
        model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model


def visualize_model():
    graph = tf.Graph()
    m = get_neural_model(compile=False)  # Your model implementation
    #with graph.as_default():
    sgd = SGD(lr=0.01)
    m.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
    writer = tf.summary.create_file_writer(logdir="'tensorboard", graph=graph)
    writer = tf.summary.FileWriter(logdir='tensorboard', graph=graph)
    writer.flush()


def to_categorical(y):
    y_encoded = np.zeros((len(y), len(classes)))
    for idx, label in np.ndenumerate(y):
        sample_index = idx[0]
        one_hot_encoding_index = classes.index(label)
        y_encoded[sample_index, one_hot_encoding_index] = 1
    return y_encoded


def predicted_to_label(y_encoded):
    y = np.empty(len(y_encoded), dtype=np.dtype('U100'))
    for index in range(len(y_encoded)):
        vector = y_encoded[index]
        y[index] = classes[np.argmax(vector)]
    return y


def train(model: Model, train_set_x, train_set_y, batch_size: int, epochs: int, validation_split: float=0.0,
          early_stopping=False):
    train_set_y = to_categorical(train_set_y)

    es = EarlyStopping(monitor='val_loss')
    callbacks = []

    if early_stopping:
        callbacks.append(es)

    tensorboard_callback = TensorBoard(log_dir='tensorboard')
    callbacks.append(tensorboard_callback)

    return model.fit(train_set_x, train_set_y,
                     batch_size=batch_size, epochs=epochs,
                     verbose=2, validation_split=validation_split,
                     callbacks=callbacks)


def evaluate(model: Model, test_set_x, test_set_y, batch_size: int):
    test_set_y = to_categorical(test_set_y)
    loss, accuracy = model.evaluate(test_set_x, test_set_y, batch_size=batch_size, verbose=2)
    return loss, accuracy


def predict(model: Model, test_set_x):
    return model.predict(test_set_x)


def plot_history(history):
    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/training_accuracy.pdf')
    plt.close()
    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/training_loss.pdf')
