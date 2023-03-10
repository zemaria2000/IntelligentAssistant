from settings import CORR_GROUP, INFLUXDB_PASSWORD, INFLUXDB_DATABASE, INFLUXDB_HOST, INFLUXDB_PORT, INFLUXDB_USERNAME, DATA_DIR
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from influxdb import InfluxDBClient
import joblib


# Instantiating an InfluxDB client
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

# I think this will get us all the values our DB has stored upon the P_SUM table
rs = db_client.query(f"select mean(value) from P_SUM WHERE time < '{last_ts}' group by time(1m)")
points = list(rs.get_points())

counter = 0
break_date = None

# This allows us to get 5 points?
for i in range(len(points)-1, -1, -1):
    if points[i]['mean'] is None: 
        if counter == 0: break_date = points[i+1]["time"]
        counter += 1
        if counter > 5: break

# For each variable to predict
for predictor in CORR_GROUP:
    pred_data = pd.DataFrame()
    pred_data_rs = None
    if break_date is None:
        # This will select all values that we have stored for each variable to predict
        pred_data_rs = db_client.query(f"select mean(value) from {','.join(CORR_GROUP[predictor])} WHERE time < '{last_ts}' group by time(1m)")
    else:
        # This will only get the points that are located between right now's timestamp and the breaked date that is defined above
        pred_data_rs = db_client.query(f"select mean(value) from {','.join(CORR_GROUP[predictor])} WHERE time > '{break_date}' and time < '{last_ts}' group by time(1m)")
    
    if pred_data_rs is None: raise Exception('Query failed')
    
    # for each feature from which our variable to predict depends on 
    for var in CORR_GROUP[predictor]:
        if len(pred_data.index) == 0:
            # getting the values for each feature nad putting the timestamp as the DataFrame index
            time_values = pred_data_rs.get_points(var)
            pred_data.index = [i['time'] for i in time_values]
        
        influx_values = pred_data_rs.get_points(var)
        # defining a sclaer to normalize our data
        scaler = MinMaxScaler()
        norm_influx_value = scaler.fit_transform(np.array([i['mean'] for i in influx_values]).reshape(-1, 1))
        # Storing in the scalers directory our new scalers
        joblib.dump(scaler, f'new_scalers/{var}.scale')
        pred_data[var] = norm_influx_value
    
    # storing in the data directory our new data that we can use to train our models
    pred_data.to_csv(DATA_DIR + predictor + '.csv')