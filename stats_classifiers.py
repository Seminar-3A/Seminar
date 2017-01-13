import sys
import numpy as np
import pylab as pl

from data_extraction import get_raw_data, add_feat, add_bucket
from logit_regression import calc_pred_data
from utils_logit import acc_pred
from random_forest import fitting_forest
from constantes import yclass_label


def get_accuracy(model_type,stock_name,start,stop,p_days,period,dist_period):
    """

    :param model_type:
    :param stock_name:
    :param start:
    :param stop:
    :param p_days:
    :param period:
    :param dist_period:
    :return:
    """
    rw_data = get_raw_data(stock_name, start, stop)
    ft_data = add_feat(rw_data, p_days)
    new_ft_data = add_bucket(ft_data, dist_period)
    X_features = new_ft_data.columns.tolist()
    y_index = X_features.index('Tmrw_return')
    del X_features[y_index]
    z_index = X_features.index('expect_ret')
    del X_features[z_index]
    new_ft_data['Ticker'] = stock_name
    if model_type == "rf":
        t_index = X_features.index(yclass_label)
        del X_features[t_index]
        input_X = new_ft_data[X_features]
        input_Y = new_ft_data[yclass_label]
        accuracy, pred_table = fitting_forest(input_X, input_Y, period, n_estimators=100)
        return 100*accuracy

    pred_table = calc_pred_data(new_ft_data, period, X_features)
    acc = acc_pred(pred_table)
    return acc


if __name__ == "__main__":
    stock_name = sys.argv[1]
    start = sys.argv[2]
    stop = sys.argv[3]
    acc_logit_arr,acc_rf_arr = [], []


    #
    # ##Access the candlestick period
    #
    # # p_days = int(sys.argv[4])
    # period = int(sys.argv[4])
    # dist_period = int(sys.argv[5])
    # params_range = range(4,31,2)
    # for p_days in params_range:
    #     acc_rf = get_accuracy("rf", stock_name, start, stop, p_days, period, dist_period)
    #     acc_logit = get_accuracy("logit", stock_name, start, stop, p_days, period, dist_period)
    #     acc_logit_arr.append(acc_logit)
    #     acc_rf_arr.append(acc_rf)
    #
    # pl.plot(params_range,acc_logit_arr,color="blue",label="logit")
    # pl.scatter(params_range, acc_logit_arr, color="blue")
    # pl.plot(params_range, acc_rf_arr, color="green", label="rf")
    # pl.scatter(params_range, acc_rf_arr,color="green")
    # pl.legend(bbox_to_anchor=(1,1))
    # pl.ylabel("Accuracy (%)")
    # pl.xlabel("Candlestick period (days)")
    # pl.title("Accuracy depending on the Candlestick period  "+"\n"
    #          +"Training period = "+str(period)
    #          +", Distribution period = " + str(dist_period)
    #          )
    # pl.xlim([params_range[0], params_range[-1] + 1])
    # pl.show()

    # ##Access the training period
    #
    # p_days = int(sys.argv[4])
    # dist_period = int(sys.argv[5])
    # params_range = range(100, 301, 10)
    # for period in params_range:
    #     acc_rf = get_accuracy("rf", stock_name, start, stop, p_days, period, dist_period)
    #     acc_logit = get_accuracy("logit", stock_name, start, stop, p_days, period, dist_period)
    #     acc_logit_arr.append(acc_logit)
    #     acc_rf_arr.append(acc_rf)
    #
    # pl.plot(params_range, acc_logit_arr, color="blue", label="logit")
    # pl.scatter(params_range, acc_logit_arr, color="blue")
    # pl.plot(params_range, acc_rf_arr, color="green", label="rf")
    # pl.scatter(params_range, acc_rf_arr, color="green")
    # pl.legend(bbox_to_anchor=(1, 0.7))
    # pl.ylabel("Accuracy (%)")
    # pl.xlabel("Training period (days)")
    # pl.title("Accuracy depending on the training period " + "\n"
    #          + "candlestick period = " + str(p_days)
    #          + ", distribution period = " + str(dist_period)
    #          )
    # pl.xlim([params_range[0], params_range[-1] + 1])
    # pl.show()



    ##Access the distribution period
    p_days = int(sys.argv[4])
    period = int(sys.argv[5])
    params_range = range(100,210,10)
    for dist_period in params_range:
        acc_rf = get_accuracy("rf", stock_name, start, stop, p_days, period, dist_period)
        acc_logit = get_accuracy("logit", stock_name, start, stop, p_days, period, dist_period)
        acc_logit_arr.append(acc_logit)
        acc_rf_arr.append(acc_rf)

    pl.plot(params_range,acc_logit_arr,color="blue",label="logit")
    pl.scatter(params_range, acc_logit_arr, color="blue")
    pl.plot(params_range, acc_rf_arr, color="green", label="rf")
    pl.scatter(params_range, acc_rf_arr,color="green")
    pl.legend(bbox_to_anchor=(1,0.8))
    pl.ylabel("Accuracy (%)")
    pl.xlabel("Distribution period (days)")
    pl.title("Accuracy depending on the distribution period "+"\n"
             +"candlestick period = "+str(p_days)
             +", training period = " + str(period)
             )
    pl.xlim([params_range[0],params_range[-1]+1])
    pl.show()

