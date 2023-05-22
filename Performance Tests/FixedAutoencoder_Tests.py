import pandas as pd
import numpy as np
import tensorflow as tf
from test_settings import TRAIN_SPLIT, PREVIOUS_STEPS, VARIABLES, DATA_DIR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

# Generating an empty dataframe
performance_df = pd.DataFrame()
# Sorting the varibales in alphabetical order
VARIABLES = sorted(VARIABLES)


for var_to_predict in VARIABLES:
        
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 1. DATASET LOADING + PROCESSING
    # 1.1. LOADING THE DATASET
    input_df = pd.read_csv(f"{DATA_DIR}{var_to_predict}.csv")
    input_df.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
    # retrieving the variables in which we are interested
    df = input_df[['Date', var_to_predict]]

    # 1.2. PRE-PROCESS THE DATA
    # Smoothing the data with the aid of Exponential Moving Average
    df['EMA'] = df.loc[:, var_to_predict].ewm(span = 8, adjust = True).mean()
    # Normalizing the data
    scaler = MinMaxScaler()
    df[var_to_predict] = scaler.fit_transform(np.array(df[var_to_predict]).reshape(-1, 1))
    # Smoothing the data with the aid of Exponential Moving Average
    df['EMA_Normalized'] = df.loc[:, var_to_predict].ewm(span = 8, adjust = True).mean()

    # 1.3. TRAINING AND TEST SPLITS
    # Splitting data between training and testing
    train_data_size = int(TRAIN_SPLIT * len(df)) 
    train_data = df[:train_data_size]
    test_data = df[train_data_size:len(df)]


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 2. FUNCTION USED TO DEFINE THE WAY WE DIVIDE OUR TIME SERIES TO MAKE PREDICTIONS

    # Function that divides our dataset according to the previous_steps number
    # Watch this - https://www.youtube.com/watch?v=6S2v7G-OupA&t=888s&ab_channel=DigitalSreeni
    # This function will take our data and for each 'PREVIOUS_STEPS' timestamps, the next one is saved in the y_values
    def divide_time_series(x, y, prev_steps):
        x_values = []
        y_values = []

        for i in range(len(x)-prev_steps):
            x_values.append(x.iloc[i:(i+prev_steps)].values)
            y_values.append(y.iloc[i+prev_steps])

        return np.array(x_values), np.array(y_values)

    # Defining our train and test datasets based on the previous function
    train_X, train_y = divide_time_series(x = train_data[var_to_predict],
                                    y = train_data[var_to_predict],
                                    prev_steps = PREVIOUS_STEPS)
    test_X, test_y = divide_time_series(x = test_data[var_to_predict],
                                    y = test_data[var_to_predict],
                                    prev_steps = PREVIOUS_STEPS)


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 3. BUILDING THE FIXED MODEL

    model = tf.keras.Sequential()
    initializer = tf.keras.initializers.GlorotNormal(seed = 13)

    model.add(tf.keras.layers.LSTM(48, activation = 'swish', kernel_initializer=initializer, input_shape = (PREVIOUS_STEPS, 1), return_sequences = True))
    model.add(tf.keras.layers.LSTM(32, activation = 'swish', kernel_initializer=initializer, return_sequences = False))
    model.add(tf.keras.layers.Dense(32, activation = 'swish', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(24, activation = 'swish', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(18, activation = 'swish', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(14, activation = 'swish', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(10, activation = 'swish', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(14, activation = 'swish', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(18, activation = 'swish', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(24, activation = 'swish', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(30, activation = 'swish', kernel_initializer=initializer))

    # Compiling the model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2), loss = 'mean_squared_error', metrics = [tf.keras.metrics.RootMeanSquaredError()])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0.0001,
        patience = 5,
        verbose = 1, 
        mode = 'min',
        restore_best_weights = True)

    model.summary()

    # Fitting the model
    history = model.fit(train_X, train_y, epochs = 200, batch_size = 100, validation_split = 0.2, callbacks = [early_stop]).history



    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 4. PREDICTIONG ON OUR TEST DATA
    test_predict = model.predict(test_X)
    # Generating a vector with the y predictions
    test_predict_y = []
    for i in range(len(test_predict)):
        test_predict_y.append(test_predict[i][PREVIOUS_STEPS-1])


    # -----------------------------------------------------------------------------
    # 5. ANALYSING OUR DATA
    # Some important indicators
    mae = mean_absolute_error(test_y, test_predict_y)
    mse = mean_squared_error(test_y, test_predict_y)
    rmse = math.sqrt(mse)
    r2 = r2_score(test_y, test_predict_y)

    # Dictionary with the performance of the regression
    performance = {'Variable': f'{var_to_predict}', 'MAE': {np.float32(mae)}, 'MSE': {np.float32(mse)}, 'RMSE': {np.float32(rmse)}, 'R2': {np.float32(r2)}}

    # Storing the variable's indicators in the DataFrame
    performance_df = performance_df.append(performance, ignore_index = True)


# Saving the results of the tests in a CSV file
performance_df.to_csv('CSVs/FixedAutoencoder.csv')

