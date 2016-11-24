#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime
import sys

from pandas import DataFrame
from pandas.io.data import DataReader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


main_feat = ['High', 'Low', 'Close']
pd.set_option('chained_assignment', None)
default_limit_classes = ["-inf", -5, -2, 0, 2, 5, "+inf"]


def get_raw_data(stock_name, start, stop, target="Variation_Close", features=main_feat):

    """
    :param stock_name: string
    :param start: string
    :param stop: string
    :param features: array of strings
    :param target: string "variation" to add the variation column or "log" to add the log variation column
    :return: the history data of stock_name from start until stop given for a certain features
    """

    start, stop = [int(el) for el in start.split('-')], [int(el) for el in stop.split('-')]

    dr = DataReader(stock_name, 'yahoo', datetime(start[0], start[1], start[2]), datetime(stop[0], stop[1], stop[2]))

    raw_data = dr[features]

    if target == "log":
        raw_data['Return_Close'] = 0
        dates = raw_data.index
        raw_data.loc[dates[1:], 'Return_Close'] = np.log(
            np.array(raw_data.loc[dates[1:], 'Close']) / np.array(raw_data.loc[dates[:-1], 'Close']))
    else:
        raw_data['Variation_Close'] = 0
        dates = raw_data.index
        raw_data.loc[dates[1:], 'Variation_Close'] = 100 * (
        np.array(raw_data.loc[dates[1:], 'Close']) - np.array(raw_data.loc[dates[:-1], 'Close'])) / np.array(
            raw_data.loc[dates[:-1], 'Close'])

    return raw_data


def frmt_raw_data(stock_name, start, stop, raw_data=DataFrame(),
                  features=main_feat, target="Variation_Close", limit_classes=default_limit_classes):
    """

    :param stock_name: string
    :param start: string
    :param stop: string
    :param raw_data: DataFrame
    :param features: array of strings
    :param target: string "variation" default or "log"
    :param limit_classes: array containing limits of the daily close variation classes
    :return: format the raw history data and add to each observation the actual prediction of close return
    """
    if raw_data.empty:
        raw_data = get_raw_data(stock_name, start, stop, target, features)

    if raw_data.columns[-1] == "Return_Close":

        raw_data.Return_Close = raw_data.Return_Close.shift(-1)
        raw_data.columns = [main_feat + ['Tmrw_return']]
        raw_data = raw_data.dropna()
    else:
        raw_data.Variation_Close = raw_data.Variation_Close.shift(-1)
        raw_data.columns = [main_feat + ['Tmrw_variation']]
        raw_data = raw_data.dropna()
        raw_data["variation_classes"] = divide_in_classes(raw_data["Tmrw_variation"], limit_classes)
    raw_data['Ticker'] = stock_name

    return raw_data


def divide_in_classes(variation_vector, limit_classes=default_limit_classes):
    """
    :param variation_vector: vector of float containing daily variation ratio
    :param limit_classes: vector containing the limits of the different classes of daily close variation
    :return: vector of strings
    """
    classes = ["Between " + str(limit_classes[i]) + " and " + str(limit_classes[i + 1]) for i in range(0, len(limit_classes) - 1)]

    classes_col = np.empty(len(variation_vector), dtype=object)
    for i in range(0, len(variation_vector)):
        if variation_vector[i] <= limit_classes[1]:
            classes_col[i] = classes[0]
        elif variation_vector[i] > limit_classes[-2]:
            classes_col[i] = classes[-1]
        else:
            for y in range(1, len(limit_classes)-2):
                if limit_classes[y] < variation_vector[i] <= limit_classes[y+1]:
                    classes_col[i] = classes[y]
    return classes_col


def get_multiple_inputs(stock_list, start, stop):
    """

    :param stock_list: list of stocks
    :param start:
    :param stop:
    :param features:
    :return:
    """

    frames = [frmt_raw_data(stock_list[i], start, stop, raw_data=DataFrame()) for i in range(len(stock_list))]
    df_input = pd.concat(frames)

    return df_input


if __name__ == "__main__":
    stock_name = sys.argv[1]
    start = sys.argv[2]
    stop = sys.argv[3]
    raw_data = get_raw_data(stock_name, start, stop, target="Variation_Close", features=main_feat)
    frmt_data = frmt_raw_data(stock_name, start, stop, raw_data,
                              features=main_feat, target="Variation_Close", limit_classes=default_limit_classes)
    print("Done downloading and formatting the input data, saving it...")
    frmt_data.to_csv("Input_data.csv")