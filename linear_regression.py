import sys
import numpy as np
import datetime as dt

from sklearn import linear_model
from sklearn import preprocessing

from data_extraction import get_raw_data, add_feat, add_bucket
from utils_lib import check_data_input
from constantes import yclass_label, y_reg_label
import pylab as pl


def fitting_linear(input_X, input_Y, period, pos_threshold=0.01):

    print('Fitting the linear regression for the {}'.format(set))

    tot_days = input_X.shape[0]
    pred_table = input_X.iloc[period:]
    pred_table['Expected_return'] = np.nan
    pred_table["Tmrw_return"] = input_Y
    dates = sorted(input_X.index)

    reg = linear_model.LinearRegression()
    errors = []
    accuracy = []
    for i in range(period, tot_days):

        tmp_table = input_X.iloc[i-period:i+1]
        tmp_table = preprocessing.scale(tmp_table)
        x_train = tmp_table[:-1]
        x_test = np.array([tmp_table[-1]])

        y_train = input_Y.iloc[i-period:i]
        y_test = np.array([input_Y.iloc[-1]])

        fit_regression = reg.fit(x_train, y_train)

        predicted_y = reg.predict(x_test)
        pred_table.loc[dates[i], "Expected_return"] = predicted_y

        square_error = ((predicted_y - y_test) ** 2)[0]

        date_i = dates[i]

        print('Squared Error at {} : {}'.format(date_i, square_error))
        acc = np.sign(predicted_y*y_test)[0]
        print ("Hit Ratio (1: good trend predicted) {}".format(acc))
        errors.append(square_error)
        accuracy.append(acc)

    mse = np.mean(errors)
    score = float(accuracy.count(1)) / float(len(accuracy))
    pred_table["diff"] = np.abs(pred_table["Expected_return"] - pred_table["Tmrw_return"])
    pred_table["naive_pos"] = np.sign(pred_table["Expected_return"])

    pred_table["pos"] = 0
    pred_table.loc[np.abs(pred_table["Expected_return"]) >= pos_threshold, "pos"] = \
        np.sign(pred_table.loc[np.abs(pred_table["Expected_return"]) >= pos_threshold, "Expected_return"])

    pred_table["pnl"] = pred_table["pos"] * pred_table["Tmrw_return"]

    return score, mse, pred_table


if __name__ == "__main__":
    
    stock_name = sys.argv[1]
    start = sys.argv[2]
    stop = sys.argv[3]
    p_days = int(sys.argv[4])
    period = int(sys.argv[5])
    dist_period = 0

    check_data_input(p_days, period, dist_period, start, stop)

    # Getting Raw data
    rw_data = get_raw_data(stock_name, start, stop)

    # Formating data to add 4*p_days features
    new_ft_data = add_feat(rw_data, p_days)

    # Getting the feat array
    X_features = new_ft_data.columns.tolist()
    y_index = X_features.index(y_reg_label)
    del X_features[y_index]

    new_ft_data['Ticker'] = stock_name

    new_ft_data.to_csv("Input_data.csv")
    print("Done downloading and formatting the input data, saving it...")

    # Fitting the Linear Regression
    input_X = new_ft_data[X_features]
    input_Y = new_ft_data[y_reg_label]
    score, mse, pred_table = fitting_linear(input_X, input_Y, period, pos_threshold=0.01)

    print("MSE of the linear regression is {}".format(mse))
    print("Hit ratio of the linear regression is {}".format(score))

    pl.plot(np.cumsum(pred_table["pnl"]))
    pl.show()