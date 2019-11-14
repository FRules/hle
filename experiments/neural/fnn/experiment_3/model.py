from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

NAME = "neural_2_dense_adam"


def get_model():
    model = Sequential()
    model.add(Dense(768, input_dim=1024, activation="relu"))
    model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(4))
    model.add(Activation("softmax"))

    adam = Adam()
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model
