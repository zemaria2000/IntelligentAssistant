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

# Instantiating the InfluxDB client
db_client = InfluxDBClient(
        host=INFLUXDB_HOST,
        port=INFLUXDB_PORT, 
        username=INFLUXDB_USERNAME, 
        password=INFLUXDB_PASSWORD, 
        database=INFLUXDB_DATABASE
    )

def main_program():
    
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
        main_program()
        time.sleep(50)

# for what I've seen, a daemon process is a process that can run parallel to the rest of our scripts
# in the background
if __name__ == '__main__':
    with daemon.DaemonContext(chroot_directory=None, working_directory=WORKING_DIR) as context:
        run()