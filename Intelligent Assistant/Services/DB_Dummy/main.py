import influxdb_client
from influxdb_client.client.write_api import ASYNCHRONOUS
from datetime import datetime
# from settings import INFLUXDB, DATA_DIR, VARIABLES, INJECT_TIME_INTERVAL
import pandas as pd
import time
import os
import yaml

config_yaml_path = "./Configuration/config.yaml"

with open(config_yaml_path, 'r') as f:
    Config = yaml.full_load(f)
# SOME DATABASE VARIABLES

# Database variables
db_url = Config['INFLUXDB_URL']
db_org = Config['DOCKER_INFLUXDB_INIT_ORG']
db_token = Config['DOCKER_INFLUXDB_INIT_ADMIN_TOKEN']
db_bucket = Config['DOCKER_INFLUXDB_INIT_BUCKET']

# getting the variables
VARIABLES = Config['VARIABLES']

INJECT_TIME_INTERVAL = Config['ML']['Inject_Interval']

# getting the directories
DATA_DIR = Config['Dir']['Data']


# Instantiate the InfluxDB client
client = influxdb_client.InfluxDBClient(
    url=db_url,
    token=db_token,
    org=db_org
)

def error_cb(details, data, exception): print(exception)

write_api = client.write_api(write_options = ASYNCHRONOUS, error_callback = error_cb)


# -------------------- FUNCTION THAT SENDS DATA TO INFLUXDB ---------------------- #

def send_values(measurement, equipment, value):
    
             
    to_send = influxdb_client.Point(measurement) \
        .tag("Equipment", equipment) \
        .tag("DataType", "Real Data") \
        .field("value", value) \
        .time(datetime.utcnow(), influxdb_client.WritePrecision.NS)
    
    # .strftime("%m/%d/%Y, %H:%M:%S")
    
    return write_api.write(bucket = db_bucket, org = db_org, record = to_send) 
  


# ------------------------ LOADING THE DATASETS ------------------------- #
df = pd.DataFrame()

for var in VARIABLES:
    new_df = pd.read_csv(f'{DATA_DIR}{var}.csv', index_col = 'Unnamed: 0')
    new_df = new_df[f'{var}']
    df = pd.concat([df, new_df], axis = 1)



# ------------------------ SENDING DATA TO THE DATABASE ------------------------- #

for i in range(len(df)):

    st = time.time()

    for var in VARIABLES:

        send_values(var, "Compressor", df.iloc[i][var])

        print(f"New data sent: Variable {var}, Value {df.iloc[i][var]}")
   
    print(f"\n Values succesffully sent. Waiting {INJECT_TIME_INTERVAL} seconds for the next injection \n")

    et = time.time()

    elapsed_time = et - st
    print('Execution time for the data injection:', elapsed_time, 'seconds \n')

    time.sleep(INJECT_TIME_INTERVAL)

