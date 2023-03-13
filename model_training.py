from pprint import pprint

import sklearn.metrics

import autosklearn.regression
from settings import PRED_MODELS, LINREG_LIST, CORR_GROUP, DATA_DIR
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
import pickle
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


#   This function is used to define the split between the amount of training data and testing data. This value is parsed, and it defaults to 90 if nothing
# is defined when launching the script. Additionally, the value must be between 50 and 95
def value_with_threshold(value):
    try:
        ivalue = int(value)
    except:
        raise argparse.ArgumentTypeError(
            value + ' is an invalid integer, the split division must be an integer between 50\% and 95%')
    if ivalue < 50 or ivalue > 95:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


#   This function is responsible for creating an appropriate dataset to then be seperated in training and testing data
def create_supervised_dataset(df, target, feats, n_in=1, n_out=1):
    # LEGEND: df - data that we retrieve from the csv / that we supply to train the model
    # target - variable to predict (given by the args.model parse)
    # feats - list with all the features from which our variable to predict depends
    # n_in - input size; number of entries needed to make a forecast (given by args.input)
    # n_out - output size; number of entries for the model to forecast (given by the args.output)
    cols, names = list(), list()
    n_vars = len(feats)
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df[feats].shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df[target].shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(1)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(1)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    # It is important to mention that this function automatically puts the objective variable in the last column(s)
    # so that when we do the train and test splitting we can just say our train and test_Y are the last columns
    return agg.values

    #   This function results in a dataframe where depending on the n_in (number of entries needed to make a forecast), we'll have n_in x feats columns of values
    # For a certain timestamp (first column of the csv files), every row will have the values of every variable from the previous 15 mins. That said, if we have
    # a variable x1, we'll have n_in columns, that go from the timestamp (t_now-t_n_in, t_now-t_(n_in-1), ... , t_now)


if __name__ == '__main__':

    #   The parsers are used to define a bunch of parameters to define when we launch the script. However, in this case, only the 'model' parameter is
    # mandatory, as it is the only one that is not defined with a '-'. Every other parameter (--time, --input, etc) is optional, and it has a default value
    # associated
    parser = argparse.ArgumentParser(
        prog='ModelTraining', description='Program to train models as requested')
    parser.add_argument('model', choices=CORR_GROUP.keys(),
                        help='Pick which model to train')
    parser.add_argument('-t', '--time', type=int, default=60,
                        help='Time in minutes for how much time to spend on training, defaults at one hour')
    parser.add_argument('-f', '--file', type=str, default=None,
                        help='Training file, if specified, if not the program retrieves the file from the default location')
    parser.add_argument('-i', '--input', type=int, default=15,
                        help='Input size, or History window, number of entries needed to make a forecast, defaults at 15mins')
    parser.add_argument('-o', '--output', type=int, default=1,
                        help='Output size, or Prediction Window, number of entries to forecast, defaults at 1')
    parser.add_argument('-s', '--split', default=90, type=value_with_threshold,
                        help='The split between training and test data for the training of the models')

    args = parser.parse_args()

    # If we specify a certain file in the '-f' or '--file' parser argument
    if args.file is not None:

        if not os.path.exists(args.file):
            raise Exception("Input File does not exist")

        # Regarding which model we choose (args.model), we'll have different degrees of freedom / independent variables, that are put in this 'input_df' variable
        input_df = pd.read_csv(args.file, usecols=CORR_GROUP[args.model])

        # Update sclaers. The MinMaxScaler() function normalizes the data so that it is compressed between 0 and 1
        scaler = MinMaxScaler()
        print('Print1 \n\n\n')
        print(scaler)
        print('\n\n')

        # Depending the model we choose, we'll have a series of different variables to predict (see 'settings.py' file)
        for c in CORR_GROUP[args.model]:
            # fit_transform calculates mean and st. deviation for each feature and transforms all points of that feature
            input_df[c] = scaler.fit_transform(
                np.array(input_df[c]).reshape(-1, 1))
            # it persists an arbitrary Python object into one file
            joblib.dump(scaler, f'new_scalers/{c}.scale')
            print('for cycle \n\n')
            print(input_df)

    #   If we don't specify a certain training file, this just goes to the directory '.data/'args.model'.csv' and uses that data. Worth noting that
    # the data in this csv files is already normalized and pre-processed, so ti's just basically a matter of loading it to the input_df variable
    else:
        input_df = pd.read_csv(
            f'{DATA_DIR}{args.model}.csv', index_col='Unnamed: 0')
        print('else \n\n')
        print(input_df)

    #   Now we'll create the dataset, already pre-processed, to be used in the training and testing phases. This is made by the 'create_supervised_dataset'
    # funtion, which will take our pre-processed data (input_df), and according to the model chosen (args.model) and it's variables to predict (CORR_GROUP[args.model]),
    # will create a dataset
    values = create_supervised_dataset(df=input_df,
                                       target=args.model,
                                       feats=CORR_GROUP[args.model],
                                       n_in=args.input,
                                       n_out=args.output)
    len_values = values.shape[0]

    # Splitting the processed data in training and testing splits, according to the split given in the args.parse (default = 90% testing)
    n_train_seconds = int(args.split/100*len_values)
    n_test_seconds = int(len_values)
    # the training dataset will contain the values until row number 'n_train_seconds-1'
    train = values[:n_train_seconds, :]
    # the test dataset will contain the reamining, from 'n_train_seconds' until the end
    test = values[n_train_seconds:n_test_seconds, :]

    # Split into inputs (features) and outputs (labels)
    train_X, train_y = train[:, :-args.output], train[:, -args.output:]
    test_X, test_y = test[:, :-args.output], test[:, -args.output:]
    test_predictions = None

    # If we are running a Linear Regression model ('RealE_SUM', 'ReacEc_L1', 'ReacEc_L3'), it will just use a Linear Regression from the sklearn library
    if args.model in LINREG_LIST:
        linreg = LinearRegression().fit(train_X, train_y)
        # pickle.dump(linreg, open(f'models/{PRED_MODELS[args.model]}', 'wb'))
        test_predictions = linreg.predict(test_X)

    # If we are running one of the other models, we'll have AutoML from the Auto SKLearn library
    else:
        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=60*args.time,
            per_run_time_limit=6*args.time,
            tmp_folder='./tmp/autosklearn_regression_'+args.model+'_tmp',
        )
        try:
            automl.fit(train_X, train_y, dataset_name=args.model)
            # pickle.dump(automl, open('models/'+PRED_MODELS[args.model], 'wb'))
            test_predictions = automl.predict(test_X)
        except Exception:
            raise Exception

    results = pd.DataFrame(test_y)
    results.rename(columns = {'0': 'test_y'})
    results['test_predictions'] = test_predictions

    print(results.head(10))

    rmse = sklearn.metrics.mean_squared_error(
        test_y, test_predictions, squared=False)
    print(f'RMSE = {rmse}')

    # this is giving out an error that I'm not managing to understand... "IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed"
    # with open(f'results/{args.model}.csv', 'w', newline = '') as writer:
    #     for i in range(args.output):
    #         print(f'y{i+1},{str(sklearn.metrics.mean_squared_error(test_y[:,i], test_predictions[:,i], squared=False))}')
    #         writer.write(f'y{i+1},{str(sklearn.metrics.mean_squared_error(test_y[:,i], test_predictions[:,i], squared=False))}')
