import sys
import preprocessor
import embeddings
from experiments.neural import api as neural_api
from experiments.linear import api as linear_api

from config import DATA_PATH


def main(arguments):
    if len(arguments) == 2:
        load_dataset = True
        dataset_timestamp = arguments[1]

    if load_dataset is False:
        train_set_x, train_set_y, test_set_x, test_set_y = preprocessor.get_preprocessed_dataset(DATA_PATH)
        train_set_x_embedded, test_set_x_embedded = embeddings.get_embeddings(train_set_x, test_set_x)
        preprocessor.save_preprocessed_data(train_set_x_embedded, train_set_y,
                                            test_set_x_embedded, test_set_y,
                                            train_set_x, test_set_x)
    else:
        train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y, train_set_x, test_set_x = \
            preprocessor.load_preprocessed_data(dataset_timestamp, restore_not_embedded_data=True)

    print("Length train set:", len(train_set_x_embedded), "\nLength test set:", len(test_set_x_embedded))

    # train_neural_models(train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y)
    train_linear_models(train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y)


def train_neural_models(train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y):
    models = neural_api.get_all_models()
    train_results = neural_api.train_all_models(models, train_set_x_embedded, train_set_y, batch_size=128, epochs=300)
    neural_api.plot_histories(train_results)
    test_results = neural_api.evaluate_all_models(models, test_set_x_embedded, test_set_y, batch_size=128)
    print(test_results)


def train_linear_models(train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y):
    models = linear_api.get_all_models()
    train_results = linear_api.train_all_models(models, train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y)
    linear_api.visualize_results(train_results, test_set_y)
    test_results = linear_api.evaluate_all_models(models, test_set_x_embedded, test_set_y)
    print(test_results)


if __name__ == '__main__':
    main(sys.argv)
