from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

NAME = "neural_2_dense_sgd_high_lr_dropout"


def get_model():
    model = Sequential()
    model.add(Dense(64, input_dim=1024, activation="relu"))
    model.add(Dropout(0.75))
    model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
    model.add(Dropout(0.75))
    model.add(Dense(4))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.5)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model
