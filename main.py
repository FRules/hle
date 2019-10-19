import preprocessor
import embeddings
import sys
import neural_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

load_dataset = False
dataset_timestamp = None

if __name__ == '__main__':
    arguments = sys.argv

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

    logistic_regression_classifier = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=10000)
    logistic_regression_classifier.fit(train_set_x_embedded, train_set_y)

    predictions_valid = logistic_regression_classifier.predict(test_set_x_embedded)
    accuracy = accuracy_score(test_set_y, predictions_valid)
    print("Accuracy:", accuracy)

    neural_model_classifier = neural_model.get_neural_model()
    history = neural_model.train(neural_model_classifier, train_set_x_embedded,
                                 train_set_y, batch_size=128, epochs=1500, validation_split=0.3)

    neural_model.plot_history(history)
    neural_model.evaluate(neural_model_classifier, test_set_x_embedded, test_set_y, batch_size=128)
