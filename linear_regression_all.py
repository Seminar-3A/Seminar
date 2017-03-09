import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import time
import os

from sklearn import linear_model
from sklearn import preprocessing


from data_extraction import get_raw_data, add_feat
from utils_lib import check_data_input, plot_pnl, create_dir, check_unavailable_data,plot_pnl_all,dump_df
from constantes import y_reg_label, calibration_period


def fitting_linear(input_X, input_Y, period, calibration_period, regression_type, alpha, pos_threshold=0.01):
    print('Fitting the linear regression for the {}'.format(set))
    pred_table = input_X.iloc[-calibration_period:]
    pred_table['Expected_return'] = np.nan
    pred_table["Tmrw_return"] = input_Y
    dates = sorted(input_X.index)

    if regression_type.lower() == 'ridge':
        reg = linear_model.Ridge(alpha=alpha)
    elif regression_type.lower() == 'lasso':
        reg = linear_model.Lasso(alpha=alpha)
    else:
        reg = linear_model.LinearRegression()
    errors = []
    accuracy = []

    for i in range(calibration_period):

        if i == calibration_period-1:
            tmp_table = input_X.iloc[i - (calibration_period + period):]
        else:
            tmp_table = input_X.iloc[i-(calibration_period+period):(i-calibration_period)+1]

        tmp_table = preprocessing.scale(tmp_table)
        x_train = tmp_table[:-1]
        x_test = np.array([tmp_table[-1]])

        y_train = input_Y.iloc[i - (calibration_period + period):(i - calibration_period)]
        y_test = np.array([input_Y.iloc[i - calibration_period]])

        reg.fit(x_train, y_train)

        predicted_y = reg.predict(x_test)
        pred_table.loc[dates[i-calibration_period], "Expected_return"] = predicted_y

        square_error = ((predicted_y - y_test) ** 2)[0]

        acc = np.sign(predicted_y*y_test)[0]

        errors.append(square_error)
        accuracy.append(acc)

    mse = np.mean(errors)
    score = float(accuracy.count(1)) / float(len(accuracy))
    pred_table["diff"] = pred_table["Expected_return"] - pred_table["Tmrw_return"]
    pred_table["naive_pos"] = np.sign(pred_table["Expected_return"])
    pred_table["pos"] = 1e7*pred_table["Expected_return"]
    pred_table["pnl"] = pred_table["pos"]*pred_table["Tmrw_return"]
    return score, mse, pred_table


if __name__ == "__main__":
    cwd = os.getcwd()
    universe_dir = create_dir(cwd, "stocks_hist")
    plot_dir = create_dir(cwd, "plots_bt")
    pnl_dir = create_dir(cwd, "pnl_bt")
    stocks_list_path = "universe.csv"
    stocks_df = pd.read_csv(stocks_list_path)

    regression_type = sys.argv[1]
    pnl_dir = create_dir(pnl_dir, regression_type)
    alpha = float(sys.argv[2])
    if alpha >0:
        pnl_dir = create_dir(pnl_dir, str(sys.argv[2]))
    start = sys.argv[3]
    stop = sys.argv[4]
    p_days = int(sys.argv[5])
    period = int(sys.argv[6])
    # pnl_all = DataFrame(index=list(stocks_df["Stock"]), columns=["pos","pnl"])
    pnl_all = DataFrame()
    pos_all = DataFrame()
    filename = "_".join([str(el) for el in [regression_type,alpha,start,
                                        stop,p_days,period
                                        ]])
    for stock_name in stocks_df["Stock"]:
        print("Back test for stock name : ",stock_name)
        # filename = "_".join([str(el) for el in [stock_name,regression_type,alpha,start,
        #                                 stop,p_days,period
        #                                 ]])
        plot_bt = "plt"
        dist_period = 0

        print("Backtest " + stock_name)

        # Getting Raw data
        rw_data = get_raw_data(stock_name, start, stop)

        if len(rw_data) < calibration_period:
            print("The data aren't enough to backtest for {} days!".format(calibration_period))
            continue
        check_data_input(p_days, period, dist_period, start, stop)

        # Formating data to add 4 * p_days features
        new_ft_data = add_feat(rw_data, p_days)

        if not(check_unavailable_data(new_ft_data, calibration_period, period)):
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

            score, mse, pred_table = fitting_linear(input_X, input_Y, period, calibration_period,
                                                    regression_type, alpha, pos_threshold=0)

            pred_table["Ticker"] = stock_name
            pred_table['p_days'] = str(p_days)
            pred_table['period'] = str(period)


        pnl = list(pred_table["pnl"])
        pos = list(pred_table["pos"])
        pnl_all[stock_name] = pred_table["pnl"]
        pos_all[stock_name] = pred_table["pos"]



    dump_df(pnl_all, pnl_dir, "pnl_"+filename)
    dump_df(pos_all, pnl_dir, "pos_"+filename)

    ## Plot backtest
    plot_pnl_all(pnl_dir, filename)