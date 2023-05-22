import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
from test_settings import TRAIN_SPLIT, PREVIOUS_STEPS, VARIABLES, DATA_DIR, MODEL_DIR, SCALER_DIR, LIN_REG_VARS
from keras.models import load_model
import joblib


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. LOADING ALL THE MODELS + SCALERS
for var in VARIABLES:
    if var in LIN_REG_VARS:
        globals()[f'model_{var}'] = joblib.load(f'{MODEL_DIR}{var}.h5')
    else:
        # globals()[f'model_{var}'] = joblib.load(f'Models/{var}.h5')
        globals()[f'model_{var}'] = load_model(f'{MODEL_DIR}{var}.h5')

    # loading the scalers with joblib works just fine...
    globals()[f'scaler_{var}'] = joblib.load(f'{SCALER_DIR}{var}.scale')


# Auxiliary variables to store all the performance indicators
performance_df = pd.DataFrame()
# Soriting the variables alphabetically
VARIABLES = sorted(VARIABLES)


for var_to_predict in VARIABLES:

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 2. DATASET LOADING + PROCESSING
    # 2.1. LOADING THE DATASET
    input_df = pd.read_csv(f"{DATA_DIR}{var_to_predict}.csv")
    input_df.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
    # retrieving the variables in which we are interested
    df = input_df[['Date', var_to_predict]]
    # Smootinhg the real data
    df['EMA'] = df.loc[:, var_to_predict].ewm(span = 8, adjust = True).mean()


    # 2.2. PRE-PROCESS THE DATA
    # Normalizing the data
    scaler = joblib.load(f'{SCALER_DIR}{var_to_predict}.scale')
    df[var_to_predict] = scaler.fit_transform(np.array(df[var_to_predict]).reshape(-1, 1))
    # Smoothing the data with the aid of Exponential Moving Average
    df['EMA_Normalized'] = df.loc[:, var_to_predict].ewm(span = 8, adjust = True).mean()
    
    # 2.3. TRAINING AND TEST SPLITS
    # Splitting data between training and testing
    train_data_size = int(TRAIN_SPLIT * len(df)) 
    train_data = df[:train_data_size]
    test_data = df[train_data_size:len(df)]


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 3. FUNCTION USED TO DEFINE THE WAY WE DIVIDE OUR TIME SERIES TO MAKE PREDICTIONS

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
    train_X, train_y = divide_time_series(x = train_data['EMA_Normalized'],
                                    y = train_data['EMA_Normalized'],
                                    prev_steps = PREVIOUS_STEPS)
    test_X, test_y = divide_time_series(x = test_data['EMA_Normalized'],
                                    y = test_data['EMA_Normalized'],
                                    prev_steps = PREVIOUS_STEPS)


    # -----------------------------------------------------------------------------
    # 4. FITTING THE MODELS TO THE TRAINING AND TESTING DATA
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0.0001,
        patience = 20,
        verbose = 1, 
        mode = 'min',
        restore_best_weights = True)
    
    # Model fitting
    if var_to_predict in LIN_REG_VARS:
        globals()[f'model_{var_to_predict}'].fit(train_X, train_y)
    else:
        # fitting the model
        history = globals()[f'model_{var_to_predict}'].fit(train_X, train_y, epochs = 100, batch_size = 100, validation_split = 0.2, callbacks = [early_stop]).history


    # -----------------------------------------------------------------------------
    # 5. PREDICTING ON OUR TEST DATA
    if var_to_predict in LIN_REG_VARS:
        test_predict_y = globals()[f'model_{var_to_predict}'].predict(test_X)
    else:
        test_predict = globals()[f'model_{var_to_predict}'].predict(test_X)
        # Generating a vector with the y predictions
        test_predict_y = []
        for k in range(len(test_predict)):
            test_predict_y.append(test_predict[k][PREVIOUS_STEPS-1])

    # -----------------------------------------------------------------------------
    # 6. CALCULATING SOME INDICATORS
    mae = mean_absolute_error(test_y, test_predict_y)
    mse = mean_squared_error(test_y, test_predict_y)
    rmse = math.sqrt(mse)
    r2 = r2_score(test_y, test_predict_y)
    performance = {'Variable': f'{var_to_predict}', 'MAE': {np.float32(mae)}, 'MSE': {np.float32(mse)}, 'RMSE': {np.float32(rmse)}, 'R2': {np.float32(r2)}}

    performance_df = performance_df.append(performance, ignore_index = True)

    # Saving the dataframe in a csv file
    performance_df.to_csv('CSVs/PredictionsPerformance.csv')
    
