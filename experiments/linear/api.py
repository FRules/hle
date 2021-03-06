import glob
from importlib import import_module

from sklearn.metrics import accuracy_score

import evaluation


def get_all_models():
    models = []
    module_files = glob.glob("experiments/linear/*/model*")
    for module_file in module_files:
        module_name = module_file.replace('/', '.').replace('.py', '')
        model_module = import_module(module_name)
        model = model_module.get_model()
        name = model_module.NAME
        models.append({
            "model": model,
            "name": name
        })
    return models


def train_all_models(models, train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y):
    results = []
    for m in models:
        name = m["name"]
        model = m["model"]
        result = train_and_test(model, train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y)
        results.append({"name": name, "results": result})
    return results


def train_and_test(model, train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y):
    model.fit(train_set_x_embedded, train_set_y)
    predictions_valid = model.predict(test_set_x_embedded)
    accuracy = accuracy_score(test_set_y, predictions_valid)
    return {"predictions_valid": predictions_valid, "accuracy": accuracy}


def visualize_results(results, test_set_y):
    for result in results:
        evaluation.plot_confusion_matrix(test_set_y, result["results"]["predictions_valid"], result["name"])
        evaluation.plot_confusion_matrix(test_set_y, result["results"]["predictions_valid"], result["name"], normalize=True)


def evaluate_all_models(models, test_set_x, test_set_y):
    results = []
    for m in models:
        name = m["name"]
        model = m["model"]
        accuracy = evaluate(model, name, test_set_x, test_set_y)
        results.append({
            "name": name,
            "accuracy": accuracy
        })
    return results


def evaluate(model, name, test_set_x, test_set_y):
    predictions_valid = model.predict(test_set_x)
    evaluation.plot_confusion_matrix(test_set_y, predictions_valid, name)
    evaluation.plot_confusion_matrix(test_set_y, predictions_valid, name, normalize=True)
    accuracy = accuracy_score(test_set_y, predictions_valid)
    return accuracy
