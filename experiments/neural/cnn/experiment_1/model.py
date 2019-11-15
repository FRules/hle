from tensorflow.keras import Sequential
from tensorflow.keras.layers import MaxPooling1D, Dense, Dropout, Input, Conv1D, Activation, Flatten
from tensorflow.keras.optimizers import Adam

NAME = "neural_cnn"


def get_model():
    n_filters = 8
    kernel_size = 7

    model = Sequential()
    model.add(Conv1D(n_filters, kernel_size, activation='relu', input_shape=(1024, 1)))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(n_filters * 2, kernel_size, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(n_filters * 4, kernel_size, activation='relu'))
    model.add(MaxPooling1D(35))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4))
    model.add(Activation('softmax'))

    adam = Adam()
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model
