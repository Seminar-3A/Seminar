#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime
import sys

from pandas import DataFrame
from pandas.io.data import DataReader
import pandas as pd
import numpy as np

from constantes import main_feat, default_limit_classes
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

    start, stop = [int(el) for el in start.split('-')], [int(el) for el in stop.split('-')]

    dr = DataReader(stock_name, 'yahoo', datetime(start[0], start[1], start[2]), datetime(stop[0], stop[1], stop[2]))

    raw_data = dr[features]
    raw_data['Return_Close'] = 0
    dates = raw_data.index
    raw_data.loc[dates[1:], 'Return_Close'] = np.log(
        np.array(raw_data.loc[dates[1:], 'Close']) / np.array(raw_data.loc[dates[:-1], 'Close']))

    return raw_data


def get_ret_class(raw_data, ret_ranges):

    """
    :param frmt_data: DataFrame
    :param ret_ranges: array of floats
    :return: Label the returns into a certain class
    The class zero correspond to the lowest return level

    """
    raw_data['Tmrw_Class'] = np.nan
    # We label diffently the following extreme return values :
    # The returns below the lowest return range and the returns above the highest return range
    raw_data.loc[(raw_data['Tmrw_return'] < ret_ranges[0]),'Tmrw_Class'] = 0
    raw_data.loc[(raw_data['Tmrw_return'] >= ret_ranges[-1]), 'Tmrw_Class'] = len(ret_ranges)

    for i in range(len(ret_ranges)-1):
        ret_level_min = ret_ranges[i]
        ret_level_max = ret_ranges[i+1]
        raw_data.loc[(raw_data['Tmrw_return'] >= ret_level_min)&(raw_data['Tmrw_return'] < ret_level_max), 'Tmrw_Class'] = i+1

    return raw_data


def frmt_raw_data(stock_name, start, stop, ret_ranges=default_limit_classes, raw_data=DataFrame(),features=main_feat):

    """
    :param stock_name: string
    :param start: string
    :param stop: string
    :param raw_data: DataFrame
    :param features: array of strings
    :return: Format the raw history data and add to each observation the actual prediction of close return

    """
    if raw_data.empty:
        raw_data = get_raw_data(stock_name, start, stop,features)

    raw_data.Return_Close = raw_data.Return_Close.shift(-1)
    raw_data.columns = [main_feat + ['Tmrw_return']]
    raw_data = raw_data.dropna()
    raw_data['Ticker'] = stock_name
    frmt_data = get_ret_class(raw_data, ret_ranges)

    return frmt_data


def get_multiple_inputs(stock_list, start, stop):
    """

    :param stock_list: list of stocks
    :param start:
    :param stop:
    :param features:
    :return:
    """

    frames = [frmt_raw_data(stock_list[i], start, stop,
                            ret_ranges=default_limit_classes,
                            raw_data=DataFrame(),
                            features=main_feat) for i in range(len(stock_list))]

    df_input = pd.concat(frames)

    return df_input

if __name__ == "__main__":
    stock_name = sys.argv[1]
    start = sys.argv[2]
    stop = sys.argv[3]
    raw_data = get_raw_data(stock_name, start, stop, features=main_feat)
    frmt_data = frmt_raw_data(stock_name, start, stop,
                              ret_ranges=default_limit_classes,
                              raw_data=raw_data,
                              features=main_feat)
    print("Done downloading and formatting the input data, saving it...")
    frmt_data.to_csv("Input_data.csv")

    # Fitting the Random Forest
    input_X = frmt_data[main_feat]
    input_Y = frmt_data['variation_classes']
    data_subdivided = subdivide_data(input_X, input_Y, test_size=0.3)

    fit_forest, score, prediction = fitting_forest(data_subdivided, n_estimators=100)
    print("score of the random forest fitting is {}".format(score))