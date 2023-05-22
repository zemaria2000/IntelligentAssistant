
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from test_settings import TRAIN_SPLIT, PREVIOUS_STEPS, VARIABLES, DATA_DIR, MODEL_DIR, SCALER_DIR, LIN_REG_VARS
from keras.models import load_model
import joblib 
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import confusion_matrix as cm
import random

# %%
# Defining a global variable
global thresholds
thresholds = dict()

# ----------------------------------------------------------------------
# 1. MODEL LOADING + SCALERS + THRESHOLDS (RMSE)

for var_to_predict in VARIABLES:
    if var_to_predict in LIN_REG_VARS:
        globals()[f'model_{var_to_predict}'] = joblib.load(f'{MODEL_DIR}{var_to_predict}.h5')
    else:
        globals()[f'model_{var_to_predict}'] = load_model(f'{MODEL_DIR}{var_to_predict}.h5')
    # Loading the model history
    globals()[f'{var_to_predict}_history'] = np.load(f'{MODEL_DIR}{var_to_predict}.npy',allow_pickle='TRUE').item()
    globals()[f'scaler_{var_to_predict}'] = joblib.load(f'{SCALER_DIR}{var_to_predict}.scale')
    # Storing the thresholds in the global dictionary
    thresholds[var_to_predict] = globals()[f'{var_to_predict}_history']['val_root_mean_squared_error'] 


# Creating a DataFrame to store the global results
final_results_df = pd.DataFrame()

# CHOOSING WHETHER WE WANT TO TEST POINT OR CONTINUOUS ANOMALIES
anomaly_type = input('Choose if you want point (p) or continuous (c) anomalies: ')

# sorting the variables alphabetically
VARIABLES = sorted(VARIABLES)


for var_to_predict in VARIABLES:

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 2. DATASET PRE-PROCESSING
    # Loading + splitting it + normalizing it...

    # 2.1. LOADING THE DATASET
    input_df = pd.read_csv(f"{DATA_DIR}{var_to_predict}.csv")
    # retrieving the variables in which we are interested
    df = input_df[['Date', var_to_predict]]

    # 2.2. PRE-PROCESS THE DATA
    # Normalizing the data
    scaler = joblib.load(f'{SCALER_DIR}{var_to_predict}.scale')
    df[var_to_predict] = scaler.fit_transform(np.array(df[var_to_predict]).reshape(-1, 1))

    # 2.3. TRAINING AND TEST SPLITS
    # Splitting data between training and testing
    train_data_size = int(TRAIN_SPLIT * len(df)) 
    train_data = df[:train_data_size]
    test_data = df[train_data_size:len(df)]

    # 2.4. APPLYING EXPONENTIAL MOVING AVERAGE TO SMOOTH THE TRAINING AND TESTING DATA
    # Smoothing the data with the aid of Exponential Moving Average
    train_data['EMA_Normalized'] = train_data.loc[:, var_to_predict].ewm(span = 8, adjust = True).mean()
    test_data['EMA_Normalized'] = test_data.loc[:, var_to_predict].ewm(span = 8, adjust = True).mean()

    # 3.5. FUNCTION USED TO DEFINE THE WAY WE DIVIDE OUR TIME SERIES TO MAKE PREDICTIONS
    def divide_time_series(x, y, prev_steps):
        x_values = []
        y_values = []

        for i in range(len(x)-prev_steps):
            x_values.append(x.iloc[i:(i+prev_steps)].values)
            y_values.append(y.iloc[i+prev_steps])

        return np.array(x_values), np.array(y_values)

    # Defining our train datasets based on the previous function
    train_X, train_y = divide_time_series(x = train_data['EMA_Normalized'],
                                    y = train_data['EMA_Normalized'],
                                    prev_steps = PREVIOUS_STEPS)



    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 3. MODEL FITTING - TRAINING THE MODEL AND VALIDATING IT

    # Defining a callback
    early_stop = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            min_delta = 0.0001,
            patience = 20,
            verbose = 1, 
            mode = 'min',
            restore_best_weights = True)

    # Fitting the models    
    if var_to_predict in LIN_REG_VARS:
        globals()[f'model_{var_to_predict}'].fit(train_X, train_y)
    else:
        history = globals()[f'model_{var_to_predict}'].fit(train_X, train_y, epochs = 100, batch_size = 100, validation_split = 0.2, callbacks = [early_stop]).history



    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 4. INPUTING ANOMALIES IN THE DATASET

    # alpha values to iterate for
    iteration_values = np.linspace(0.01, 15, 2001)

    # auxiliary dataframe
    aux_df = pd.DataFrame()

    # Doing 10 iterations - 10 random inputs of anomalies
    for i in range(10):

        # Reseting the indexes of the test_data DataFrame
        test_data.reset_index(drop = True, inplace = True)

        # 4.1. DEFINING RANDOM POINTS AS ANOMALIES
        # storing the index of the manually inputed anomalies
        anomalies_index = []
        # getting the mean of the values in the dataset
        average_value = np.mean(test_data[var_to_predict])
        # how many point anomalies we want 
        point_anomaly_counter = 100
        
        # POINT ANOMALIES
        if anomaly_type == 'p':
            # getting the length of the testing dataset
            length = len(test_data)
            # Generating 100 different points
            anomalies_index = random.sample(range(0, len(test_data) - PREVIOUS_STEPS), point_anomaly_counter)    
            for i in range(len(test_data)):
                x = random.random()
                if i in anomalies_index:
                    if x < 0.5:
                        test_data.loc[i + PREVIOUS_STEPS, var_to_predict] = 0
                    if x >= 0.5:
                        test_data.loc[i + PREVIOUS_STEPS, var_to_predict] = 2 * random.uniform(0.9*average_value, 1.1*average_value)

        # SEQUENTIAL ANOMALIES
        if anomaly_type == 'c':
            # choosing 4 random numbers to start the iteration
            anomalies_index_start = random.sample(range(0, len(test_data) - 25 - PREVIOUS_STEPS), 4)  
            for number in anomalies_index_start:
                x = random.random()
                if x < 0.5: 
                    test_data.loc[number+PREVIOUS_STEPS:number+PREVIOUS_STEPS+25, var_to_predict] = 0
                if x >= 0.5: 
                    for j in range(number + PREVIOUS_STEPS, number + PREVIOUS_STEPS + 25):
                        test_data.loc[j, var_to_predict] = 2 * random.uniform(0.9*average_value, 1.1*average_value)
            # putting the remaining anomalies in a list
            anomalies_index = []
            for number in anomalies_index_start:
                for j in range(25):
                    anomalies_index.append(j + number)

        # 4.2. RE-NORMALIZING OUR TEST DATA, FOR IT TO CONSIDER THE ANOMALIES
        test_data['EMA_Normalized'] = test_data.loc[:, var_to_predict].ewm(span = 8, adjust = True).mean()
        # Defining the test datasets with the anomalies
        test_X, test_y = divide_time_series(x = test_data['EMA_Normalized'],
                                    y = test_data['EMA_Normalized'],
                                    prev_steps = PREVIOUS_STEPS)


        # 4.3. MAKING THE PREDICTIONS IN OUR "ANOMALOUS" TEST SPLIT
        if var_to_predict in LIN_REG_VARS:
            test_predict_y = globals()[f'model_{var_to_predict}'].predict(test_X)
            rmse = min(history['val_root_mean_squared_error'])
        else:
            test_predict = globals()[f'model_{var_to_predict}'].predict(test_X)
            # Generating a vector with the y predictions
            test_predict_y = []
            for k in range(len(test_predict)):
                test_predict_y.append(test_predict[k][PREVIOUS_STEPS-1]) 
            mse = mean_squared_error(test_y, test_predict_y)
            rmse = np.sqrt(mse)

        # 4.4. GETTING THE DIFFERENCE BETWEEN REAL AND PREDICTED VALUES
        difference = abs(test_y - test_predict_y)

        # 4.5. CREATING A LIST WITH THE TRUE ANOMALIES
        true_anomalies = []
        for l in range(len(test_y)):
            if l in anomalies_index:
                true_anomalies.append(1)
            else:
                true_anomalies.append(0)

        # Creating an auxiliary dict and DataFrame
        aux = dict()
        aux['MCC'] = 0

        # 4.6. ITERATIVE CYCLE IN WHICH WE'LL TEST A BUNCH OF THRESHOLDS
        for alpha in iteration_values:

            # Getting the "predicted anomalies"
            predicted_anomalies = []
            for m in range(len(test_y)):
                # if difference[m] > alpha * thresholds[var_to_predict]:
                if difference[m] > alpha * rmse:
                    predicted_anomalies.append(1)
                else:
                    predicted_anomalies.append(0)

            # Getting the TP, TN, FP, FN
            confusion_matrix = cm(true_anomalies, predicted_anomalies)

            # Getting the MCC
            MCC = mcc(true_anomalies, predicted_anomalies)

            # In case the MCC is better than the last one, we'll store new values in the aux dict
            if MCC > aux['MCC']:
                aux['TP'] = round(confusion_matrix[1][1], 4)
                aux['TN'] = round(confusion_matrix[0][0], 4)
                aux['FP'] = round(confusion_matrix[0][1], 4)
                aux['FN'] = round(confusion_matrix[1][0], 4)
                if aux['TP'] + aux['FP'] == 0:
                    aux['Precision'] = 0
                else:
                    aux['Precision'] = round(aux['TP']/(aux['TP'] + aux['FP']), 4)
                if (aux['TP'] + aux['FN']) == 0:
                    aux['Recall'] = 0
                else:
                    aux['Recall'] = round(aux['TP']/(aux['TP'] + aux['FN']), 4)
                if (aux['Precision'] + aux['Recall']) == 0:
                    aux['F1'] = 0
                else:
                    aux['F1'] = round(2 * aux['Precision'] * aux['Recall'] / (aux['Precision'] + aux['Recall']), 4)
                aux['alpha'] = round(alpha, 4)
                aux['MCC'] = round(MCC, 4)

        # Storing the best result from the iteration
        aux_df = pd.concat([aux_df, pd.DataFrame(aux, index = [i])])

        # 4.7. RELOADING THE DATASET
        input_df = pd.read_csv(f"{DATA_DIR}{var_to_predict}.csv")
        # retrieving the variables in which we are interested
        df = input_df[['Date', var_to_predict]]
        # normalizing
        df[var_to_predict] = scaler.fit_transform(np.array(df[var_to_predict]).reshape(-1, 1))
        # Splitting data between training and testing
        train_data_size = int(TRAIN_SPLIT * len(df)) 
        train_data = df[:train_data_size]
        test_data = df[train_data_size:len(df)]

    # 5. GETTING THE BEST RESULTS FROM THE DATAFRAME AND STORING THEM IN THE GLOBAL RESULTS ONE
    best_mcc = aux_df.loc[aux_df['MCC'] == aux_df['MCC'].max()]
    best_mcc['Variable'] = var_to_predict

    final_results_df = pd.concat([final_results_df, best_mcc])

# 6. STORING THE RESULTS IN A CSV FILE
final_results_df.to_csv(f'AD_Results_{anomaly_type}.csv')

print(final_results_df)


