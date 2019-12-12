from importlib import import_module
import glob
import config as cfg
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model
import evaluation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_all_models():
    models = []
    module_files = glob.glob("experiments/neural/*/experiment*/model*")
    for module_file in module_files:
        module_name = module_file.replace('/', '.').replace('.py', '')
        model_module = import_module(module_name)
        model = model_module.get_model()
        name = model_module.NAME
        plot_title = model_module.PLOT_TITLE
        if "cnn" in name:
            cnn = True
        else:
            cnn = False
        models.append({
            "model": model,
            "name": name,
            "is_cnn": cnn,
            "plot_title": plot_title
        })
    return models


def train_all_models(models, train_set_x, train_set_y, batch_size: int, epochs: int):
    results = []
    for m in models:
        is_cnn = m["is_cnn"]
        model = m["model"]
        name = m["name"]
        plot_title = m["plot_title"]
        history = train(model, train_set_x, train_set_y, batch_size, epochs, early_stopping=False, cnn=is_cnn)
        results.append({"name": name, "plot_title": plot_title, "history": history})
    return results


def plot_models(models):
    for m in models:
        model = m["model"]
        name = m["name"]
        plot_model(model, to_file=str.format('plots/architectures/{0}.png', name), show_shapes=True, expand_nested=True)


def plot_histories(results):
    for result in results:
        plot_history(result)


def plot_history(result):
    history = result["history"]
    name = result["name"]
    plot_title = result["plot_title"]
    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
    plt.title(str.format('model accuracy, model {0}', plot_title))
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
        plt.title(str.format('model loss, model {0}', plot_title))
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


def evaluate_all_models(models, test_set_x, test_set_y, batch_size: int):
    results = []
    for m in models:
        name = m["name"]
        model = m["model"]
        plot_title = m["plot_title"]
        is_cnn = m["is_cnn"]
        loss, accuracy = evaluate(model, name, plot_title, test_set_x, test_set_y, batch_size, is_cnn)
        results.append({
            "name": name,
            "loss": loss,
            "accuracy": accuracy
        })
    return results


def evaluate(model: Model, name: str, plot_title: str, test_set_x, test_set_y, batch_size: int, cnn: bool):
    test_set_y_encoded = to_categorical(test_set_y)
    if cnn:
        test_set_x = np.expand_dims(test_set_x, axis=2)
    loss, accuracy = model.evaluate(test_set_x, test_set_y_encoded, batch_size=batch_size, verbose=2)
    y_pred_encoded = model.predict(test_set_x)
    y_pred = predicted_to_label(y_pred_encoded)
    evaluation.plot_confusion_matrix(test_set_y, y_pred, name, title=plot_title)
    evaluation.plot_confusion_matrix(test_set_y, y_pred, name, title=plot_title, normalize=True)

    return loss, accuracy


def predicted_to_label(y_encoded):
    y = np.empty(len(y_encoded), dtype=np.dtype('U100'))
    for index in range(len(y_encoded)):
        vector = y_encoded[index]
        y[index] = cfg.CLASSES[np.argmax(vector)]
    return y

