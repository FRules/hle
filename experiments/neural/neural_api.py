from importlib import import_module
import glob
import config as cfg
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train_all_models(train_set_x, train_set_y, batch_size: int, epochs: int):
    module_files = glob.glob("experiments/neural/cnn/experiment_1_*/model*")
    results = []
    for module_file in module_files:
        module_name = module_file.replace('/', '.').replace('.py', '')
        model_module = import_module(module_name)
        model = model_module.get_model()
        name = model_module.NAME
        if "cnn" in name:
            cnn = True
        else:
            cnn = False
        history = train(model, train_set_x, train_set_y, batch_size, epochs, early_stopping=False, cnn=cnn)
        results.append({"name": name, "history": history})
    return results


def plot_histories(results):
    for result in results:
        plot_history(result)


def plot_history(result):
    history = result["history"]
    name = result["name"]
    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
    plt.title(str.format('model accuracy, model {0}', name))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str.format('plots/neural/acc_{0}.pdf', name))
    plt.close()
    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
        plt.title(str.format('model loss, model {0}', name))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str.format('plots/neural/loss_{0}.pdf', name))


def train(model: Model, train_set_x, train_set_y, batch_size: int, epochs: int, validation_split: float=0.3,
          early_stopping=False, cnn=False):
    if cnn:
        train_set_x = np.expand_dims(train_set_x, axis=2)

    train_set_y = to_categorical(train_set_y)

    if validation_split > 0.0:
        es = EarlyStopping(monitor='val_loss')
    else:
        es = EarlyStopping(monitor='loss')
    callbacks = []

    if early_stopping:
        callbacks.append(es)

    tensorboard_callback = TensorBoard(log_dir='tensorboard')
    callbacks.append(tensorboard_callback)

    return model.fit(train_set_x, train_set_y,
                     batch_size=batch_size, epochs=epochs,
                     verbose=2, validation_split=validation_split,
                     callbacks=callbacks)


def to_categorical(y):
    y_encoded = np.zeros((len(y), len(cfg.CLASSES)))
    for idx, label in np.ndenumerate(y):
        sample_index = idx[0]
        one_hot_encoding_index = cfg.CLASSES.index(label)
        y_encoded[sample_index, one_hot_encoding_index] = 1
    return y_encoded
