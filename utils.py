import pandas as pd
from matplotlib.finance import fetch_historical_yahoo

from yahoo_finance import Share


ticker = 'AAPL'
features = ['Symbol', 'Date', 'Low', 'High', 'Close', 'Volume']
start_date = '2014-04-25'
end_date = '2014-04-29'


def get_input_data(ticker, features, start_date, end_date):
    """
    Return DataFrame with the features input as columns
    :param ticker: String
    :param features: list of string
    :param start_date: string in the following format '2014-05-23'
    :param end_date: string in the same format as above
    :return: DataFrame
    """
    stock = Share(ticker)
    stock_data = stock.get_historical(start_date, end_date)
    frames = [pd.DataFrame(stock_data[i], index=[i]) for i in range(len(stock_data) - 1, -1, -1)]
    df_stock_data = pd.concat(frames)
    df_stock_data = df_stock_data[features]

    return df_stock_data


tickers = ['AAPL', 'YHOO', 'BABA']


def get_multiple_inputs(tickers, features, start_date, end_date):
    """

    :param tickers: list of string
    :param features: list of string
    :param start_date: string in the following format '2014-05-23'
    :param end_date: string in the same format as above
    :return:
    """
    dict_stock = {}

    for i in range(len(tickers)):

        dict_stock[tickers[i]] = get_input_data(ticker, features, start_date, end_date)

    return dict_stock