import sys
import numpy as np
import pdb

import utils_logit
from constantes import min_range, max_range, step_range, main_feat
from data_extraction import frmt_raw_data, get_raw_data
from utils_logit import adjust_ret, get_expected_ret_range, acc_pred, sharpe_ratio, get_ret_ranges, norm_input
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

    for iter in xrange(n_iters):
        classifier.train(lr=learning_rate)
        learning_rate *= 0.95

    return classifier.predict(z)


def calc_pred_data(classified_data, period):

    """
    :param classified_data: DataFrame
    :param period: integer
    :return: Calculate the predicted class for a certain observation and for a certain window period of training

    """
    tot_days = classified_data.shape[0]
    pred_table = classified_data.iloc[period:]
    pred_table['Pred_Class'] = np.nan
    dates = sorted(classified_data.index)

    for i in range(period, tot_days):
        tmp_table = classified_data.iloc[i-period:i]
        test_ipt = pred_table[main_feat].iloc[i-period]
        x_tr_norm, x_tt_norm = norm_input(tmp_table, test_ipt)
        y_tr = adjust_ret(tmp_table, ret_ranges)
        pred_arr = test_lrn(x_tr_norm, y_tr, x_tt_norm)
        pred_table.loc[dates[i], "Pred_Class"] = np.argmax(pred_arr)

    return pred_table

if __name__ == "__main__":

    stock_name = sys.argv[1]
    start = sys.argv[2]
    stop = sys.argv[3]
    period = int(sys.argv[4])
    # Building the return ranges
    ret_ranges = get_ret_ranges(min_range, max_range, step_range)

    # Extraction of historical data
    raw_data = get_raw_data(stock_name, start, stop, features=main_feat)

    # Format of the historical data by adding the return of the close price
    # Classification of close return using the ret ranges
    classified_data = frmt_raw_data(stock_name, start, stop, raw_data, features=main_feat)

    # Prediction of the tomorrow class return
    pred_table = calc_pred_data(classified_data, period)

    # Evaluation of the accuracy and the sharpe ratio
    acc = acc_pred(pred_table)
    print("Accuracy (%) : ", acc)
    new_prd_df = get_expected_ret_range(pred_table, ret_ranges)
    shrp = sharpe_ratio(new_prd_df)
    print("Sharpe Ratio : ", shrp)

