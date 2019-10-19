from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn
import pandas as pd

classes = ["list", "factoid", "summary", "yesno"]


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
        one_hot_encoding_index = classes.index(label)
        y_encoded[sample_index, one_hot_encoding_index] = 1
    return y_encoded


def predicted_to_label(y_encoded):
    y = np.empty(len(y_encoded), dtype=np.dtype('U100'))
    for index in range(len(y_encoded)):
        vector = y_encoded[index]
        y[index] = classes[np.argmax(vector)]
    return y


def train(model: Model, train_set_x, train_set_y, batch_size: int, epochs: int, validation_split: float=0.0):
    train_set_y = to_categorical(train_set_y)

    es = EarlyStopping(monitor='val_loss')
    callbacks = [es]
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


def plot_confusion_matrix(y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
    plot_name = "confusion_matrix"
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
            plot_name = plot_name + "_normalized.pdf"
        else:
            title = 'Confusion matrix, without normalization'
            plot_name = plot_name + "_not_normalized.pdf"

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    figure = ax.get_figure()
    figure.savefig('plots/' + plot_name, dpi=400)
