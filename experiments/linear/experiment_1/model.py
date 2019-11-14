from sklearn.linear_model import LogisticRegression

NAME = "log_reg"


def get_model():
    return LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=10000)
