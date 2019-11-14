import sys
import preprocessor
import embeddings
from experiments.neural import neural_api


def main(arguments):
    if len(arguments) == 2:
        load_dataset = True
        dataset_timestamp = arguments[1]

    if load_dataset is False:
        train_set_x, train_set_y, test_set_x, test_set_y = preprocessor.get_preprocessed_dataset("Questions.csv")
        train_set_x_embedded, test_set_x_embedded = embeddings.get_embeddings(train_set_x, test_set_x)
        preprocessor.save_preprocessed_data(train_set_x_embedded, train_set_y,
                                            test_set_x_embedded, test_set_y,
                                            train_set_x, test_set_x)
    else:
        train_set_x_embedded, train_set_y, test_set_x_embedded, test_set_y, train_set_x, test_set_x = \
            preprocessor.load_preprocessed_data(dataset_timestamp, restore_not_embedded_data=True)

    print("Length train set:", len(train_set_x_embedded), "\nLength test set:", len(test_set_x_embedded))
    results = neural_api.train_all_models(train_set_x_embedded, train_set_y, batch_size=128, epochs=300)
    neural_api.plot_histories(results)


if __name__ == '__main__':
    main(sys.argv)
