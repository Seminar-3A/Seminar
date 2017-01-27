import datetime as dt
import numpy as np
import pylab as pl
import pandas as pd
import os


def create_dir(path, filename):
    directory = path+"/"+filename
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def check_unavailable_data(new_ft_data,calibration_period,period):

    if len(new_ft_data)<(calibration_period+period):

        print("The history price is not sufficient to do the backtest"+"\n"
              +"Please change the training period or the number of features")

        return True
    return False

def check_data_input(p_days, period, dist_period, start, stop):

    if p_days > period:

        raise Exception('Please enter a number of p_day inferior to the training period length \n '
                        '{} > {} !'.format(p_days, period))

    py_start = dt.datetime.strptime(start,"%Y-%m-%d")
    py_stop = dt.datetime.strptime(stop, "%Y-%m-%d")

    if (py_stop-py_start).days <=max(period, p_days, dist_period):

        raise Exception('Please enter a date range higher than period parameters: \n '
                        'feature period= {} \n '
                        'training period = {} \n'
                        'distribution buckets period = {} \n'.format(p_days, period, dist_period))


def get_stats_strat(trade_hist):

    stats_strat = {}
    bp_per_trade = round(1e4*(np.mean(trade_hist["pnl"])/(2.*1e6)), 2)
    hit_ratio = round(100*np.mean(trade_hist["pnl"]> 0), 2)
    mean_ = np.mean(trade_hist['pnl'])
    std_ = np.std(trade_hist['pnl'])
    if std_ > 0:
        sharpe_ratio = np.sqrt(trade_hist.shape[0]) * (mean_ / std_)
        sharpe_ratio = round(sharpe_ratio, 2)
    else:
        sharpe_ratio = np.nan

    stats_strat["bp_per_trade"] = str(bp_per_trade)
    stats_strat["hit_ratio"] = str(hit_ratio)
    stats_strat["sharpe_ratio"] = str(sharpe_ratio)

    return stats_strat


def plot_pnl(stock_dir, pred_table):

    pl.close('all')
    trade_hist = pred_table[~(pred_table["pnl"] == 0)]
    stats_strat = get_stats_strat(trade_hist)
    trade_hist["pnl"].cumsum().plot(color="blue")
    pl.scatter(range(len(trade_hist)), np.cumsum(trade_hist["pnl"]), color="blue")
    pl.xlim([0, len(trade_hist)])
    pl.ylabel("Cumulated P&L", labelpad=0.01)
    # pl.gca().yaxis.set_label_coords(-0.1, np.mean(trade_hist["pnl"].cumsum()))
    hr = stats_strat["hit_ratio"]
    sharpe = stats_strat["sharpe_ratio"]
    bp = stats_strat["bp_per_trade"]
    start_date = trade_hist["pnl"].index[0]
    end_date = trade_hist["pnl"].index[-1]
    if type(trade_hist["pnl"].index[0]) == pd.tslib.Timestamp:
        start_date, end_date  = str(start_date.date()), str(end_date.date())
    stock_name = trade_hist["Ticker"][0]
    p_days = trade_hist["p_days"][0]
    period = trade_hist["period"][0]
    filename = "_".join([stock_name, start_date, end_date,p_days,period])+".png"
    pl.title(stock_name + " "+start_date+ " / "+end_date+" FEAT "+p_days
            +" TRAIN "+period+"\n"+" BP " +bp+ " HR "+hr+"% "+" SR "+sharpe)


    pl.savefig(stock_dir+"/"+filename)

    #pl.show()