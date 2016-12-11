import datetime as dt


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
