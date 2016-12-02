def subdivide_data(input_X, input_Y, test_size=0.3):
    """

    :param input_X: DataFrame
    :param input_Y: DataFrame
    :param test_size:
    :return: Dictionary with Data subdivided into Train and Test
    """
    shape = input_X.shape
    data_subdivided = {}
    limit = int(shape[0] * (1 - test_size))
    data_subdivided['X_train'] = input_X.iloc[:limit, ]
    data_subdivided['X_test'] = input_X.iloc[limit:, ]
    data_subdivided['Y_train'] = input_Y.iloc[:limit, ]
    data_subdivided['Y_test'] = input_Y.iloc[limit:, ]

    return data_subdivided
