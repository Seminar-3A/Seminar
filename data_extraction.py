#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime as dt
import sys

from pandas import DataFrame
from pandas.io.data import DataReader
import pandas as pd
import numpy as np

from constantes import main_feat, default_limit_classes, yclass_label
from utils_rf import subdivide_data
from random_forest import fitting_forest


pd.set_option('chained_assignment', None)


def get_raw_data(stock_name, start, stop, features=main_feat):

    """
    :param stock_name: string
    :param start: string
    :param stop: string
    :param features: array of strings
    :return: Extract History prices from start until stop and filtrate by main features

    """

    start_date = dt.datetime.strptime(start, "%Y-%m-%d")
    stop_date = dt.datetime.strptime(stop, "%Y-%m-%d")
    # add a working day to have tomorrow's return
    stop_date = stop_date + pd.tseries.offsets.BDay(1)

    dr = DataReader(stock_name, 'yahoo', start_date, stop_date)

    raw_data = dr[features]
    raw_data['Return_Close'] = 0
    dates = raw_data.index
    raw_data.loc[dates[1:], 'Return_Close'] = np.log(
        np.array(raw_data.loc[dates[1:], 'Close']) / np.array(raw_data.loc[dates[:-1], 'Close']))

    raw_data.Return_Close = raw_data.Return_Close.shift(-1)
    raw_data.columns = [features + ['Tmrw_return']]
    raw_data = raw_data.dropna()

    return raw_data


def get_data_with_past(stock_name, start, stop, features=main_feat, nb_past_days=0):
    """
    :param stock_name: string
    :param start: string
    :param stop: string
    :param features: array of strings
    :param nb_past_days : int
    :return: Extract History prices from start until stop, filtrate by main features and add past days data as variables

    """

    start_date = dt.datetime.strptime(start, "%Y-%m-%d")
    # if it's a holiday, start at the next working day
    if not bool(len(pd.bdate_range(start_date, start_date))):
        start_date = start_date + pd.tseries.offsets.BDay(1)

    stop_date = dt.datetime.strptime(stop, "%Y-%m-%d")
    # if it's a holiday, stop at the previous working day
    if not bool(len(pd.bdate_range(stop_date, stop_date))):
        stop_date = stop_date - pd.tseries.offsets.BDay(1)

    start = start_date.strftime("%Y-%m-%d")
    stop = stop_date.strftime("%Y-%m-%d")
    raw_data = get_raw_data(stock_name, start, stop, features)

    for i in range(1, nb_past_days+1):
        new_start = (dt.datetime.strptime(start, "%Y-%m-%d") - pd.tseries.offsets.BDay(i)).strftime("%Y-%m-%d")
        new_stop = (dt.datetime.strptime(stop, "%Y-%m-%d") - pd.tseries.offsets.BDay(i)).strftime("%Y-%m-%d")
        added_data = get_raw_data(stock_name, new_start, new_stop, features)
        # Checking the holidays
        k = len(added_data.index ) - len(raw_data.index)
        if k > 0:
            # enlève dernière ligne
        elif k < 0:
            # Changer start_date et retelecharger les donnees

        added_data.index = raw_data.index
        added_data.columns = [str(col) + "_" + str(i) for col in added_data.columns]
        raw_data = pd.concat([raw_data, added_data], axis=1)

    return raw_data


def get_ret_class(raw_data, ret_ranges):

    """
    :param frmt_data: DataFrame
    :param ret_ranges: array of floats
    :return: Label the returns into a certain class
    The class zero correspond to the lowest return level

    """
    raw_data[yclass_label] = np.nan
    # We label diffently the following extreme return values :
    # The returns below the lowest return range and the returns above the highest return range
    raw_data.loc[(raw_data['Tmrw_return'] < ret_ranges[0]), yclass_label] = 0
    raw_data.loc[(raw_data['Tmrw_return'] >= ret_ranges[-1]), yclass_label] = len(ret_ranges)

    for i in range(len(ret_ranges)-1):
        ret_level_min = ret_ranges[i]
        ret_level_max = ret_ranges[i+1]
        raw_data.loc[(raw_data['Tmrw_return'] >= ret_level_min)&(raw_data['Tmrw_return'] < ret_level_max), yclass_label] = i+1

    return raw_data


def frmt_raw_data(stock_name, raw_data, ret_ranges=default_limit_classes):

    """
    :param stock_name: string
    :param start: string
    :param stop: string
    :param raw_data: DataFrame
    :param features: array of strings
    :return: Format the raw history data and add to each observation the actual prediction of close return

    """

    raw_data['Ticker'] = stock_name
    frmt_data = get_ret_class(raw_data, ret_ranges)

    return frmt_data


if __name__ == "__main__":
    stock_name = sys.argv[1]
    start = sys.argv[2]
    stop = sys.argv[3]
    nb_past_days = sys.argv[4]
    raw_data = get_data_with_past(stock_name, start, stop, features=main_feat, nb_past_days=nb_past_days)
    # Getting the features names
    X_features = raw_data.columns.tolist()
    y_index = X_features.index('Tmrw_return')
    del X_features[y_index]
    # Formatting Data by labeling the returns for the classification
    frmt_data = frmt_raw_data(stock_name, raw_data, ret_ranges=default_limit_classes)

    print("Done downloading and formatting the input data, saving it...")
    frmt_data.to_csv("Input_data.csv")

    # Fitting the Random Forest
    input_X = frmt_data[X_features]
    input_Y = frmt_data[yclass_label]
    data_subdivided = subdivide_data(input_X, input_Y, test_size=0.3)

    fit_forest, score, prediction = fitting_forest(data_subdivided, n_estimators=100)
    print("score of the random forest fitting is {}".format(score))