import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

from data_extraction import get_raw_data, add_feat, add_bucket
from utils_lib import check_data_input
from constantes import yclass_label


def fitting_forest(input_X, input_Y, period, n_estimators=100):

    print('Fitting the random forest for the {}'.format(set))

    tot_days = input_X.shape[0]
    pred_table = input_X.iloc[period:]
    pred_table['Pred_Class'] = np.nan
    dates = sorted(input_X.index)

    forest = RandomForestClassifier(n_estimators=n_estimators, max_features="auto")
    scores = []
    for i in range(period, tot_days):

        tmp_table = input_X.iloc[i-period:i+1]
        tmp_table = preprocessing.scale(tmp_table)
        x_train = tmp_table[:-1]
        x_test = np.array([tmp_table[-1]])

        y_train = input_Y.iloc[i-period:i]
        y_test = np.array([input_Y.iloc[-1]])

        fit_forest = forest.fit(x_train, y_train)

        pred_table.loc[dates[i], "Pred_Class"] = forest.predict(x_test)

        score = forest.score(x_test, y_test)
        print('Hit Ratio (1: good trend prediction) {}'.format(score))
        scores.append(score)

    accuracy = np.array(scores).mean()

    return accuracy, pred_table

if __name__ == "__main__":
    stock_name = sys.argv[1]
    start = sys.argv[2]
    stop = sys.argv[3]
    p_days = int(sys.argv[4])
    period = int(sys.argv[5])
    dist_period = int(sys.argv[6])

    check_data_input(p_days, period, dist_period, start, stop)

    # Getting Raw data
    rw_data = get_raw_data(stock_name, start, stop)

    # Formating data to add 4*p_days features
    ft_data = add_feat(rw_data, p_days)

    # New format to add bucket
    new_ft_data = add_bucket(ft_data, dist_period)

    # Getting the feat array
    X_features = new_ft_data.columns.tolist()
    y_index = X_features.index('Tmrw_return')
    del X_features[y_index]
    z_index = X_features.index('expect_ret')
    del X_features[z_index]
    t_index = X_features.index(yclass_label)
    del X_features[t_index]

    new_ft_data['Ticker'] = stock_name

    new_ft_data.to_csv("Input_data.csv")
    print("Done downloading and formatting the input data, saving it...")

    # Fitting the Random Forest
    input_X = new_ft_data[X_features]
    input_Y = new_ft_data[yclass_label]
    accuracy, pred_table = fitting_forest(input_X, input_Y, period, n_estimators=100)

    print("score of the random forest fitting is {}".format(accuracy))