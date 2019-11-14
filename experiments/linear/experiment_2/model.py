from sklearn.svm import SVC

NAME = "svm"


def get_model():
    return SVC(kernel="rbf", gamma='scale', max_iter=1000000)
