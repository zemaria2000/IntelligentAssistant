from random import randint, choice
import pandas as pd
import tensorflow as tf

from settings import CORR_GROUP, PRED_MODELS, INFLUXDB_DATABASE, INFLUXDB_HOST, INFLUXDB_PASSWORD, INFLUXDB_PORT, INFLUXDB_USERNAME, WORKING_DIR
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from influxdb import InfluxDBClient
import joblib
import iso8601
from datetime import datetime, timedelta
import time
import daemon

import logging


# This is the function that allows us to make the anomaly detection. We send a query to the DB asking
# for the current values for each value to predict. Then, we apply the scalers that resulted from the 
# training process and make our predictions for the next timestamp values. We finally send the difference
# between the two values through an InfluxDB query and plot the respective difference
def ad_program():

    # Instantiating the influxDB client, located in a certain host, running on a certain port, 
    # with all the authentication credentials that are imported from the 'settings.py' file
    db_client = InfluxDBClient(
        host=INFLUXDB_HOST,
        port=INFLUXDB_PORT, 
        username=INFLUXDB_USERNAME, 
        password=INFLUXDB_PASSWORD, 
        database=INFLUXDB_DATABASE
    )

    # for each var to predict, we have different scalers that translate the influence of the different features regarding our dependent variable
    scalers = {var:joblib.load(f'scalers/{var}.scale') for var in CORR_GROUP}
    # getting the last timestamp registered in our influxDB database
    last_ts = list(db_client.query('select last(*) from P_SUM'))[0][0]['time']
    # getting the last values for each one of the variables to predict, from the database
    current_values = db_client.query(f'select last(value) from {",".join(CORR_GROUP.keys())}')
    # formating the last_ts to something like "2023-03-09T14:51.56"
    last_ts = datetime.strptime(last_ts, "%Y-%m-%dT%H:%M:%S.%fZ")
    rounded_last_ts = last_ts - timedelta(minutes=last_ts.minute % 10,
                                            seconds=last_ts.second,
                                            microseconds=last_ts.microsecond)
    # the next time stamp will be the actual timestamp plus 1 minute (basically 1 minute ahead)
    forward_ts = rounded_last_ts + timedelta(minutes=1)
    # this will retrieve the forecasted values that result from our predictive models
    forecasted_values = db_client.query(f'select last(value) from \"{",".join(CORR_GROUP.keys())}\" where time <= \'{forward_ts}\' and time >= \'{rounded_last_ts}\'')
    
    
    points = []
    for var in CORR_GROUP:
        # getting the current values for each variable and their respective predictions
        current_var_list = list(current_values.get_points(var))
        fc_var_list = list(forecasted_values.get_points(var))

        if len(current_var_list) > 0 and len(fc_var_list) > 0:
            # In here we'll get the scalers that were created during our model training process
            # and scale them (normalize them    )
            curr_var_val = np.array(current_var_list[0]['last']).reshape(1,1)
            fc_var_val = np.array(current_var_list[0]['last']).reshape(1,1)
            scaled_curr_val = scalers[var].transform(curr_var_val)
            scaled_fr_val = scalers[var].transform(fc_var_val)
            # Then we'll calculate the difference between the predicted values and the real values,
            # and we'll send a query to the InfluxDB with the respective difference, to then create
            # a graph where we can compare both values
            diff = np.abs(scaled_curr_val - scaled_fr_val)[0][0]
            points.append({
                "measurement": f"{var} Difference",
                "time": last_ts,
                "fields": {"value": diff}
            })
    logging.info(db_client.write_points(points))



def prediction_program():
    db_client = InfluxDBClient(
        host=INFLUXDB_HOST,
        port=INFLUXDB_PORT, 
        username=INFLUXDB_USERNAME, 
        password=INFLUXDB_PASSWORD, 
        database=INFLUXDB_DATABASE
    )
    # Get Last Timestamp, remove when deploying
    last_ts = list(db_client.query('select * from P_SUM GROUP BY * ORDER BY desc LIMIT 1'))[0][0]['time']

    #uncomment when deploying
    #last_ts = 'now()'

    #read models
    points = []
    
    # for each var to predict...
    for var in PRED_MODELS:
        # the forecasting timestamp will be 1 minute ahead of our current timestamp
        forecast_ts = iso8601.parse_date(last_ts) + timedelta(minutes=1)

        input_vector = []

        # for each feature that our variable to predict (var) depends on
        for needed_var in CORR_GROUP[var]:
            
            # --------------------------------------------
            # these next 4 lines of code I'll need to debug when this script is fully working... can't understand the last 2 lines
            # (this is probebly just getting the values and putting them in an numpy array...)

            # this will select during the last 15 mins (that's why we have 'last_ts - 14m') the mean value of each feature measurements
            rs = db_client.query(f"select mean(value) from {needed_var} WHERE time > '{last_ts}' - 14m and time < '{last_ts}'group by time(1m)")
            # loading the scaler for each feeature
            var_scaler = joblib.load(f'scalers/{needed_var}.scale')
            x = np.array([i['mean'] for i in rs.get_points(needed_var)]).reshape(-1, 1)
            input_vector.append(var_scaler.transform(x))
            # ---------------------------------------------------------------------

        # creating a tensor/vector 
        tensor = np.array(input_vector).transpose().reshape(-1, 15*len(CORR_GROUP[var]))
        
        # loading the model for each one of the variables to predict
        model = pickle.load(open('models/'+PRED_MODELS[var], 'rb'))
        result = model.predict(tensor)
        var_scaler = joblib.load(f'scalers/{var}.scale')
        result = var_scaler.inverse_transform(result.reshape(-1,1))
        
        # influxDB query to send the predictions to our DB
        for r in result:
            points.append({
                "measurement": f"{var} Forecast",
                "time": forecast_ts.isoformat(),
                "fields": {"value": r[0]}
            })
            forecast_ts = forecast_ts + timedelta(minutes=1)
        
    logging.info(db_client.write_points(points))

def run():
    logging.basicConfig(filename='forecaster.log', format='%(asctime)s %(message)s', level=logging.DEBUG)

    logging.info('Forecasting agent started')
    while True:
        prediction_program()
        ad_program()
        time.sleep(50)

# for what I've seen, a daemon process is a process that can run parallel to the rest of our scripts
# in the background
if __name__ == '__main__':
    with daemon.DaemonContext(chroot_directory=None, working_directory=WORKING_DIR) as context:
        run()


# daemon==1.2