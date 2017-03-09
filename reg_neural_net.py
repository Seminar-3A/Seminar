import sys
import numpy as np
import pandas as pd
import time
import os
import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import preprocessing
from tqdm import tqdm

from data_extraction import get_raw_data, add_feat
from utils_lib import check_data_input, plot_pnl, create_dir, check_unavailable_data
from constantes import y_reg_label, calibration_period

seed = 7


def fitting_reg_nn(input_X, input_Y, period, calibration_period, n_layers, pos_threshold=0.01):

    print('Fitting the NN regression for the {}'.format(set))
    pred_table = input_X.iloc[-calibration_period:]
    pred_table['Expected_return'] = np.nan
    pred_table["Tmrw_return"] = input_Y
    dates = sorted(input_X.index)

    input_dim = input_X.shape[1]
    numpy.random.seed(seed)

    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(input_dim, input_dim=input_dim, init='normal', activation='relu'))
        if n_layers > 0:
            for i in range(n_layers):
                model.add(Dense(input_dim // (2 ** (n_layers)), init='normal', activation='relu'))

        model.add(Dense(1, init='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    reg = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)

    errors = []
    accuracy = []

    for i in tqdm(range(calibration_period)):

        if i == calibration_period-1:
            tmp_table = input_X.iloc[i - (calibration_period + period):]
        else:
            tmp_table = input_X.iloc[i-(calibration_period+period):(i-calibration_period)+1]

        tmp_table = preprocessing.scale(tmp_table)
        x_train = tmp_table[:-1]
        x_test = np.array([tmp_table[-1]])

        y_train = input_Y.iloc[i - (calibration_period + period):(i - calibration_period)]
        y_test = np.array([input_Y.iloc[i - calibration_period]])

        fit_regression = reg.fit(x_train, y_train)

        predicted_y = reg.predict(x_test)
        pred_table.loc[dates[i-calibration_period], "Expected_return"] = predicted_y

        square_error = ((predicted_y - y_test) ** 2)[0]

        print('Squared Error at {} : {}'.format(dates[i-calibration_period], square_error))
        acc = np.sign(predicted_y*y_test)[0]

        print ("Hit Ratio (1: good trend prediction) {}".format(acc))
        errors.append(square_error)
        accuracy.append(acc)

    mse = np.mean(errors)
    score = float(accuracy.count(1)) / float(len(accuracy))
    pred_table["diff"] = pred_table["Expected_return"] - pred_table["Tmrw_return"]
    pred_table["naive_pos"] = np.sign(pred_table["Expected_return"])

    pred_table["pos"] = 0
    pred_table.loc[np.abs(pred_table["Expected_return"]) >= pos_threshold, "pos"] = \
        np.sign(pred_table.loc[np.abs(pred_table["Expected_return"]) >= pos_threshold, "Expected_return"])

    pred_table["pnl"] = 1e6 * pred_table["pos"] * pred_table["Tmrw_return"]
    return score, mse, pred_table


if __name__ == "__main__":
    cwd = os.getcwd()
    universe_dir = create_dir(cwd, "stocks_hist")
    plot_dir = create_dir(cwd, "plots_bt")
    stocks_list_path = "universe.csv"
    stocks_df = pd.read_csv(stocks_list_path)

    n_layers = int(sys.argv[1])
    stock_name = sys.argv[2]
    start = sys.argv[3]
    stop = sys.argv[4]
    p_days = int(sys.argv[5])
    period = int(sys.argv[6])
    plot_bt = sys.argv[7] == "plt"
    dist_period = 0

    print("Backtest " + stock_name)

    # Getting Raw data
    rw_data = get_raw_data(stock_name, start, stop)

    if len(rw_data) < calibration_period:
        raise Exception("The data aren't enough to backtest for {} days!".format(calibration_period))

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

        score, mse, pred_table = fitting_reg_nn(input_X, input_Y, period, calibration_period, n_layers, pos_threshold=0)

        pred_table["Ticker"] = stock_name
        pred_table['p_days'] = str(p_days)
        pred_table['period'] = str(period)

        if plot_bt:
            plot_pnl(plot_dir, pred_table, "Neural_Net")
