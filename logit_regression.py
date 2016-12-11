import sys
import numpy as np
import datetime as dt
from sklearn import preprocessing
from sklearn import linear_model

import utils_logit
from utils_lib import check_data_input
from constantes import quartile_ranges
from data_extraction import get_raw_data, add_feat, add_bucket
from utils_logit import adjust_ret, norm_input, acc_pred, get_sharpe_per_bckt
utils_logit = reload(utils_logit)


def softmax(x):

    """
    :param x: array
    :return: return the softmax function

    """

    # normalize values to avoid overflow
    max_val = np.exp(x - np.max(x))
    if max_val.ndim == 1:
        return max_val / np.sum(max_val, axis=0)
    else:
        return max_val / np.array([np.sum(max_val, axis=1)]).T


class LogisticRegression(object):

    def __init__(self, input, output):

        """
        :param input: array of arrays
        :param output: array of arrays
        """

        self.x = input
        self.y = output    # The response variable will be a vector with zeros and one in the the real class K
        self.W = np.zeros((len(self.x.T), len(self.y.T)))  # The weighting vectors are initialized with 0
        self.b = np.zeros(len(self.y.T))  # The bias vector is initialized with 0

    def train(self, lr=0.1):

        """
        :param lr: float
        :return:Train the dataset using the Gradient Descent algorithm to optimize the weighting vectors and the bias

        """

        p_y_given_x = softmax(np.dot(self.x, self.W) + self.b)
        diff_y = self.y - p_y_given_x
        self.W += lr * np.dot(self.x.T, diff_y)
        self.b += lr * np.mean(diff_y, axis=0)

    def predict(self, x):

        """
        :param x: array of arrays
        :return: Calculate the predicted probabilities for each class using the softmax function

        """
        return softmax(np.dot(x, self.W) + self.b)


def test_lrn(x, y, z, learning_rate=0.01, n_iters=1000):

    # Build the Multiclass Logit Regression Classifier
    classifier = LogisticRegression(input=x, output=y)

    for _ in xrange(n_iters):
        classifier.train(lr=learning_rate)
        learning_rate *= 0.95

    return classifier.predict(z)


def calc_pred_data(classified_data, period, features):

    """
    :param classified_data: DataFrame
    :param period: integer
    :return: Calculate the predicted class for a certain observation and for a certain window period of training

    """
    tot_days = classified_data.shape[0]
    pred_table = classified_data.iloc[period:]
    pred_table['Pred_Class'] = np.nan
    dates = sorted(classified_data.index)
    ret_ranges = range(len(quartile_ranges))
    real_feat = [el for el in features if not (el == "Tmrw_Class")]
    logreg = linear_model.LogisticRegression(C=1e5)
    for i in range(period, tot_days):

        tmp_table = classified_data.iloc[i-period:i+1]
        x_table = preprocessing.scale(tmp_table[real_feat])
        x_tr_norm, x_tt_norm = x_table[:-1], x_table[-1]
        y_train = np.array(tmp_table["Tmrw_Class"].iloc[:-1])
        logreg.fit(x_tr_norm, y_train)
        y_test = logreg.predict(x_tt_norm)[0]
        pred_table.loc[dates[i], "Pred_Class"] = y_test
        # test_ipt = pred_table[features].iloc[i-period]
        # x_tr_norm, x_tt_norm = norm_input(tmp_table, test_ipt)
        # y_tr = adjust_ret(tmp_table, ret_ranges)
        # pred_arr = test_lrn(x_tr_norm, y_tr, x_tt_norm)
        # pred_table.loc[dates[i], "Pred_Class"] = np.argmax(pred_arr)

    return pred_table

if __name__ == "__main__":

    stock_name = sys.argv[1]
    start = sys.argv[2]
    stop = sys.argv[3]
    p_days = int(sys.argv[4])
    period = int(sys.argv[5])
    dist_period = int(sys.argv[6])

    check_data_input(p_days, period, dist_period, start, stop)

    # Getting Raw data
    rw_data = get_raw_data(stock_name, start, stop)

    # Formating data to add 4*p_days features
    ft_data = add_feat(rw_data, p_days)

    # New format to add bucket
    new_ft_data = add_bucket(ft_data, dist_period)

    # Getting the feat array
    X_features = new_ft_data.columns.tolist()
    y_index = X_features.index('Tmrw_return')
    del X_features[y_index]
    z_index = X_features.index('expect_ret')
    del X_features[z_index]

    new_ft_data['Ticker'] = stock_name

    new_ft_data.to_csv("Input_data.csv")
    print("Done downloading and formatting the input data, saving it...")

    pred_table = calc_pred_data(new_ft_data, period, X_features)

    # Evaluation of the accuracy and the sharpe ratio
    acc = acc_pred(pred_table)
    print("Accuracy (%) : ", acc)

    for i in range(len(quartile_ranges)+1):
        shrp = get_sharpe_per_bckt(pred_table, i)
        print("Sharpe Ratio bucket range number : "+str(i), shrp)

