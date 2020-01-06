import sys
import argparse
import preprocessor
import ling_feature_extractor
import embeddings
from experiments.neural import api as neural_api
from experiments.linear import api as linear_api

from config import DATA_PATH


def main(dataset=None, include_pos_tags=False):
    if dataset is None:
        train_set_x, train_set_y, test_set_x, test_set_y = preprocessor.get_preprocessed_dataset(DATA_PATH, include_pos_tags)
        train_set_x_embedded, test_set_x_embedded = embeddings.get_embeddings(train_set_x, test_set_x)
        preprocessor.save_preprocessed_data(train_set_x_embedded, train_set_y,
                                            test_set_x_embedded, test_set_y,
                                            train_set_x, test_set_x)
    else:
        train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y, train_set_x, test_set_x = \
            preprocessor.load_preprocessed_data(dataset, restore_not_embedded_data=True)

    print("Length train set:", len(train_set_x_embedded), "\nLength test set:", len(test_set_x_embedded))

    neural_models_pipeline(train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y)
    # linear_models_pipeline(train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y)


def neural_models_pipeline(train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y):
    models = neural_api.get_all_models()
    neural_api.plot_models(models)
    train_results = neural_api.train_all_models(models, train_set_x_embedded, train_set_y, batch_size=128, epochs=300)
    neural_api.plot_histories(train_results)
    test_results = neural_api.evaluate_all_models(models, test_set_x_embedded, test_set_y, batch_size=128)
    print(test_results)


def linear_models_pipeline(train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y):
    models = linear_api.get_all_models()
    train_results = linear_api.train_all_models(models, train_set_x_embedded, train_set_y, test_set_x_embedded,
                                                test_set_y)
    linear_api.visualize_results(train_results, test_set_y)
    test_results = linear_api.evaluate_all_models(models, test_set_x_embedded, test_set_y)
    print(test_results)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HLE lab #1')

    parser.add_argument('--include-pos-tags', type=str2bool, default=False)
    parser.add_argument('--dataset', default=None, type=str)

    args = parser.parse_args()

    main(args.dataset, args.include_pos_tags)
