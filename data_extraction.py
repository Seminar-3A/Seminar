#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime as dt
import holidays
import os.path

from pandas.io.data import DataReader
import pandas as pd
import numpy as np

from constantes import main_feat, default_limit_classes, yclass_label, quartile_ranges

pd.set_option('chained_assignment', None)
us_holidays = holidays.UnitedStates()


def get_raw_data(stock_name, start, stop):

    """
    :param stock_name: string
    :param start: string
    :param stop: string
    :param features: array of strings
    :return: Extract History prices from start until stop and filtrate by main features from the current folder

    """
    flename = "_".join([stock_name, start, stop])

    if os.path.isfile(flename + ".csv"):
        raw_data = pd.read_csv(flename + ".csv")
        raw_data.index = raw_data["Date"]
        raw_data = raw_data.drop("Date",1)
        return raw_data

    print("The parameters are not in the database !! ")
    return pd.DataFrame()

def get_YF_raw_data(stock_name, start, stop, features=main_feat):

    """
    :param stock_name: string
    :param start: string
    :param stop: string
    :param features: array of strings
    :return: Extract History prices from start until stop and filtrate by main features using Yahoo Finance

    """
    flename = "_".join([stock_name, start, stop])

    if os.path.isfile(flename + ".csv"):
        return pd.read_csv(flename + ".csv")

    start_date = dt.datetime.strptime(start, "%Y-%m-%d")
    stop_date = dt.datetime.strptime(stop, "%Y-%m-%d")
    # add a working day to have tomorrow's return
    stop_date = stop_date + pd.tseries.offsets.BDay(1)
    # if it's a holiday, stop at the next working day
    if stop_date in us_holidays:
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
    raw_data = raw_data.drop_duplicates(['Close'], take_last=True)

    raw_data.to_csv(flename + ".csv")

    return raw_data


def add_feat(rw_data, p_days):

    """
    :param rw_data:
    :param p_days:
    :param features:
    :return:
    """

    if p_days == 0:
        return rw_data

    col_add = [feat+"-" + str(i) for i in range(1, p_days + 1) for feat in main_feat+["Tmrw_return"] if not(feat == "Close")]

    ft_data = (rw_data.iloc[p_days:]).copy()

    for feat in col_add:
        ft_data[feat] = np.nan
        feat_type = feat.split("-")[0]
        feat_day = int(feat.split("-")[-1])
        ft_data[feat] = np.array(rw_data.iloc[(p_days-feat_day):-feat_day][feat_type])

    return ft_data


def add_bucket(ft_data, dist_period, default_bucket=False):

    if default_bucket:
        return get_ret_class(ft_data, default_limit_classes)

    new_ft_data = ft_data.iloc[dist_period:].copy()
    new_ft_data[yclass_label] = np.nan
    new_ft_data["expect_ret"] = np.nan
    #new_ft_data["ret_ranges"] = np.nan

    dates = ft_data.index
    for i, date in enumerate(new_ft_data.index):
        prev_rets = ft_data.loc[dates[i:i+dist_period], "Tmrw_return"]
        ret_ranges = np.percentile(prev_rets, quartile_ranges)
        pred_ret = new_ft_data.loc[date, "Tmrw_return"]
        curr_class = len(ret_ranges[ret_ranges<=pred_ret])
        new_ft_data.loc[date, yclass_label] = curr_class

        if curr_class in [0, len(ret_ranges)]:
            new_ft_data.loc[date, "expect_ret"] = ret_ranges[min(len(ret_ranges)-1, curr_class)]

        else:
            new_ft_data.loc[date, "expect_ret"] = np.mean(ret_ranges[curr_class-1:curr_class+1])

        #new_ft_data.loc[date, "ret_ranges"] = "-".join(ret_ranges.astype(str))

    return new_ft_data


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
