#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def norm_input(tmp_table,test_ipt):

    """
    :param tmp_table: DataFrame
    :param test_ipt: array
    :return: Normalize the inputs (training and test inputs) for different features(High, Low , Close)

    """
    mean_l, std_l = np.mean(tmp_table['Low']), np.std(tmp_table['Low'])
    mean_h, std_h = np.mean(tmp_table['High']), np.std(tmp_table['High'])
    mean_c, std_c = np.mean(tmp_table['Close']), np.std(tmp_table['Close'])
    z_low, z_high, z_close = test_ipt['Low'], test_ipt['High'],test_ipt['Close']

    if min(std_l,std_h,std_c)>0:
        tmp_table['High_Norm'] = (tmp_table['High'] - mean_h) / std_h
        tmp_table['Low_Norm'] = (tmp_table['Low'] - mean_l) / std_l
        tmp_table['Close_Norm'] = (tmp_table['Close'] - mean_c) / std_c
        x_tr_norm = np.array(tmp_table[['High_Norm','Low_Norm','Close_Norm']])
        x_tt_norm = np.array([(z_high - mean_h) / std_h, (z_low - mean_l) / std_l,(z_close - mean_c) / std_c])

    else:
        x_tr_norm, x_tt_norm = np.array(tmp_table[['High', 'Low', 'Close']]) ,np.array([z_high, z_low, z_close])

    return [x_tr_norm,x_tt_norm]


def get_expected_ret_range(pred_table,ret_ranges):

    """
    :param pred_table: DataFrame
    :param ret_ranges: array of floats
    :return: This function add the expected return level in order to calculate the Sharpe Ratio
    The expected return is approximated by the mean value of the return range

    """
    for el in ['Pred_Class','Tmrw_Class']:
        pred_table["Exp_"+el] = np.nan
        pred_table.loc[(pred_table[el] == 0), "Exp_"+el] = ret_ranges[0]
        pred_table.loc[(pred_table[el] == len(ret_ranges)), "Exp_"+el] = ret_ranges[-1]
        for i in range(1,len(ret_ranges)):
            pred_table.loc[(pred_table[el] == i), "Exp_"+el] = np.mean(ret_ranges[i-1:i+1])

    return pred_table


def acc_pred(pred_table):

    """
    :param pred_table: DataFrame
    :return: Calculate the accuracy of the predictions for the test period

    """
    corr_vals = (pred_table[pred_table['Tmrw_Class'] == pred_table['Pred_Class']]).shape[0]
    tot_val = float(pred_table.shape[0])

    return 100*corr_vals/ tot_val


def sharpe_ratio(new_prd_df):

    """
    :param new_prd_df: DataFrame
    :return: Calculate the Sharpe Ratio for the all period

    """
    stats_df = new_prd_df[['Exp_'+el for el in ['Pred_Class','Tmrw_Class']]]
    stats_df['diff'] = stats_df['Exp_Pred_Class']-stats_df['Exp_Tmrw_Class']
    mean_ = np.mean(stats_df['diff'])
    std_ = np.std(stats_df['diff'])

    if std_ > 0:
        return np.sqrt(stats_df.shape[0])*(np.abs(mean_)/std_)

    return "No elements predicted in this bucket"


def get_ret_ranges(min_range, max_range, step_range):

    """
    :param min_range: float
    :param max_range: float
    :param step_range: float
    :return: Return a sorted array of return levels that will help us build the return classes

    """
    a = min_range
    b= max_range
    n = int((b-a)/float(step_range))
    ret_ranges_arr = []

    for i in range(n+1):
        ret_ranges_arr.append(round(((b-a)*i/float(n))+a,3))

    return ret_ranges_arr


def adjust_ret(tmp_table,ret_ranges):

    """
    :param tmp_table:
    :param ret_ranges:
    :return: Convert each return class k into an array with zeros and one in the kth index

    """
    return np.array([[int(tmp_table['Tmrw_Class'].iloc[obs] == cls) for cls in range(len(ret_ranges))] for obs in range(tmp_table.shape[0])])


def get_sharpe_per_bckt(pred_table,bucket_pos):

    stats_df = pred_table[(pred_table["Pred_Class"] == bucket_pos)][["Tmrw_return", "expect_ret"]]
    stats_df['diff'] = stats_df['Tmrw_return'] - stats_df['expect_ret']
    mean_ = np.mean(stats_df['diff'])
    std_ = np.std(stats_df['diff'])

    if std_ > 0:
        return np.sqrt(stats_df.shape[0]) * (np.abs(mean_) / std_)

    return "No elements predicted from this range"
