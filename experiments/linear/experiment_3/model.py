from sklearn.neighbors import KNeighborsClassifier

NAME = "knn"


def get_model():
    return KNeighborsClassifier(n_neighbors=5)
