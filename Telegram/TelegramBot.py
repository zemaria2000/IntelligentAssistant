# This will create a Telegram bot that will send informations about what is happening with our data IN REAL TIME
# What I want to do with it is to make requests to the DB, process the data, and then send information regarding anomalies, graphs with data, etc...
# %%
# imports
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import telebot
from bot_settings import INFLUXDB, EXCEL_DIR, VARIABLES
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os.path
import numpy as np

# some fixed variables
data_bucket = INFLUXDB['bucket']
db_url = INFLUXDB['URL']
db_token = INFLUXDB['Token']
db_org = INFLUXDB['Org'] 

API_token = '6121421687:AAEZq-HQmCe9aW39dr_mHoK9e9csYMCgcF4'
single_plot_directory = './plots/plot.png'
real_pred_directory = './plots/realpred.png'

# -----------------------------------------------------------------------------
# 1. INSTANTIATE A WRITING AND QUERYING CLIENT TO OUR INFLUXDB BUCKET
client = influxdb_client.InfluxDBClient(
    url = db_url,
    token = db_token,
    org = db_org
)

# Instantiate the write api client
write_api = client.write_api(write_options = SYNCHRONOUS)
# Instantiate the query api client
query_api = client.query_api()


# ------------------------------------------------------------------------------
# 2. CREATING OUR BOT AND OUR UPDATER (FOR THE REPETITIVE MESSAGE SENDING)
bot = telebot.TeleBot(API_token)
# updater = Updater(token=API_token, use_context=True)


# ------------------------------------------------------------------------------
# List of Multipliers for the thresholds
MULTIPLIER = {
    "C_phi_L3": 0.624,
    "F": 1.771,
    "H_TDH_I_L3_N": 2.153,
    "H_TDH_U_L2_N": 1.764,
    "I_SUM": 1.989,
    "P_SUM": 1.014,
    "ReacEc_L1": 2.761,
    "ReacEc_L3": 8.577,
    "RealE_SUM": 5.796,
    "U_L1_N": 0.864
}
MODEL_DIR = "../Directories/Models/"
SCALER_DIR = "../Directories/Scalers/"

global thresholds
thresholds = dict()
for var in VARIABLES:

    # Loading the model history
    globals()[f'{var}_history'] = np.load(f'{MODEL_DIR}{var}.npy',allow_pickle='TRUE').item()
    thresholds[var] = globals()[f'{var}_history']['val_root_mean_squared_error'] * MULTIPLIER[var]




# ------------------------------------------------------------------------------
# 3. DEFINING SOME FUNCTIONS THAT OUR BOT EXECUTES WHEN A CERTAIN REQUEST IS MADE

# 3.1 - getting the last N vals for a variable

def last_n_vals(n_vals, var):

    try:
        # querying the database for the vals
        query = f'from(bucket:"{data_bucket}")\
            |> range(start: -1h)\
            |> sort(columns: ["_time"], desc: true)\
            |> limit(n: {n_vals})\
            |> filter(fn:(r) => r.DataType == "Real Data")\
            |> filter(fn:(r) => r._measurement == "{var}")'
        
        # Send the query defined above retrieving the needed data from the database
        result = query_api.query(org = INFLUXDB['Org'], query = query)

        # getting the values (it returns the values in the alphabetical order of the variables)
        results = []
        for table in result:
            for record in table.records:
                results.append((record.get_value(), record.get_time()))

        # seperating the variable and the date values
        var_vals, date_vals = list(), list()
        for i in range(len(results)):
            var_vals.append(results[i][0])
            date_vals.append(results[i][1].strftime("%m/%d/%Y %H:%M:%S"))

        df1, df2 = pd.DataFrame(date_vals), pd.DataFrame(var_vals)
        df = pd.concat([df1, df2], axis = 1)
        df.columns = ['Date', f'{var} Values']

        return df
    
    except:
        print("Couldn't preform the task. Please type 'Help' for more info!")

# I want the message to be "Last 'N' 'VAR' values"" - need at least 4 elements"
def last_n_vals_request(message):
   request = message.text.split()
   if len(request) < 3 or request[0].lower() not in "last":
      return False
   else:
      return True


# 3.2 - getting a graph for last "time" mins' behaviour of a specific variable

def plot_var_requested(var, time):

    try:
        # querying the database for the vals
        query = f'from(bucket:"{data_bucket}")\
            |> range(start: -{time}m)\
            |> filter(fn:(r) => r.DataType == "Real Data")\
            |> filter(fn:(r) => r._measurement == "{var}")'
        
        # Send the query defined above retrieving the needed data from the database
        result = query_api.query(org = "UA", query = query)

        # getting the values (it returns the values in the alphabetical order of the variables)
        results = []
        for table in result:
            for record in table.records:
                results.append((record.get_value(), record.get_time()))

        # seperating the variable and the date values
        var_vals, date_vals = list(), list()
        for i in range(len(results)):
            var_vals.append(results[i][0])
            date_vals.append(results[i][1])

        # plotting a graph of the behaviour of our variable 
        plt.style.use('ggplot')
        plt.plot(date_vals, var_vals, 'r-', linewidth = 1)
        plt.title(f"Last {time} mins {var} behaviour" )
        plt.grid(linestyle = '--', linewidth = 0.5)

        # saving the plot
        plt.savefig(single_plot_directory)

        plt.close()
    
    except:
        print("Couldn't preform the task. Please type 'Help' for more info!")
    

def plot_var_requested_request(message):
   request = message.text.split()
   if len(request) < 3 or request[0].lower() not in "plot":
      return False
   else:
      return True


# 3.3 - getting a graph for last "time" mins real vs prediction values

def plot_real_pred(var, time):

    try:
        # querying the database for the vals (last hours vals)
        query_real = f'from(bucket:"{data_bucket}")\
            |> range(start: -{time}m)\
            |> filter(fn:(r) => r.DataType == "Real Data")\
            |> filter(fn:(r) => r._measurement == "{var}")'
        query_pred = f'from(bucket:"{data_bucket}")\
            |> range(start: -{time}m)\
            |> filter(fn:(r) => r.DataType == "Prediction Data")\
            |> filter(fn:(r) => r._measurement == "{var}")'
        
        # Send the query defined above retrieving the needed data from the database
        result_real = query_api.query(org = INFLUXDB['Org'], query = query_real)
        result_pred = query_api.query(org = INFLUXDB['Org'], query = query_pred)

        # getting the values (it returns the values in the alphabetical order of the variables)
        results_real, results_pred = [], []
        for table in result_real:
            for record in table.records:
                results_real.append((record.get_value(), record.get_time()))
        for table in result_pred:
            for record in table.records:
                results_pred.append((record.get_value(), record.get_time()))
        
        # seperating the variable and the date values
        real_var_vals, pred_var_vals, real_date_vals, pred_date_vals = list(), list(), list(), list()
        for i in range(len(results_real)):
            real_var_vals.append(results_real[i][0])
            real_date_vals.append(results_real[i][1])
        for i in range(len(results_pred)):
            pred_var_vals.append(results_pred[i][0])
            pred_date_vals.append(results_pred[i][1])

        # plotting a graph of the real vs prediction behaviour of our variable
        plt.style.use('ggplot')
        plt.plot(real_date_vals, real_var_vals, 'r-', linewidth = 1)
        plt.plot(pred_date_vals, pred_var_vals, 'g--', linewidth = 1)
        plt.title(f"Last {time} mins {var} real vS predictions" )
        plt.grid(linestyle = '--', linewidth = 0.5)
        plt.legend(['Real Values', 'Predicted Values'])

        # saving the plot
        plt.savefig(real_pred_directory)

        plt.close()

    except:
        print("Couldn't preform the task. Please type 'Help' for more info!")    


def plot_real_pred_request(message):
   request = message.text.split()
   if len(request) < 3 or request[0].lower() not in "real_pred":
      return False
   else:
      return True


# 3.4 - getting the last hours' excel report - NOT WORKING!!!

def get_last_excel():

    list_of_files = glob.glob(f'{EXCEL_DIR}*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# need to write - Last Excel Report
def get_last_excel_request(message):
    request = message.text.split()
    if len(request) < 2 or request[1].lower() not in "excel":
        return False
    else:
        return True
    


# 3.5 - getting an histogram with last "time" mins errors

def error_hist(var, time):
    
    try:
        # querying the database for the vals
        query = f'from(bucket:"{data_bucket}")\
            |> range(start: -{time}m)\
            |> filter(fn:(r) => r.DataType == "Difference")\
            |> filter(fn:(r) => r._measurement == "{var}")'
        
        # Send the query defined above retrieving the needed data from the database
        result = query_api.query(org = INFLUXDB['Org'], query = query)

        # getting the values (it returns the values in the alphabetical order of the variables)
        results = []
        for table in result:
            for record in table.records:
                results.append((record.get_value(), record.get_time()))

        # seperating the variable and the date values
        var_vals, date_vals = list(), list()
        for i in range(len(results)):
            var_vals.append(results[i][0])
            date_vals.append(results[i][1])

        # Loading the threshold for the variable

        # # plotting a graph of the behaviour of our variable 
        plt.style.use('ggplot')

        plt.title(f"Last {time} mins {var} errors VS threshold" )
        plt.hist(var_vals, bins=30)
        plt.axvline(x = thresholds[var], color='green')
        # saving the plot
        plt.savefig(single_plot_directory)

        plt.close()  

    except:
        print("Couldn't preform the task. Please type 'Help' for more info!")    
        

def error_hist_request(message):    # need to write Error_hist CAR TIME
   request = message.text.split()
   if len(request) < 3 or request[0].lower() not in "error_hist":
      return False
   else:
      return True


# 3.6 - getting a list of the variables being monitored

def get_vars_request(message): # Get Variables
   request = message.text.split()
   if len(request) < 2 or request[1].lower() not in "variables":
      return False
   else:
      return True
   

# 3.7 - generating an help statement
def help_message():
    msg_to_send = "Here are the different executable commands to which I can respond: \n\n"
    # Adding the different commands 
    msg_to_send += "- 'Get Variables' - will show a list of the variables currently being monitored; \n"
    msg_to_send += "- 'Last {N} {VAR} Values' - will give back the {N} latest values from the {VAR} that we defined; \n"
    msg_to_send += "- 'Plot {VAR} {TIME}' - will send back a plot of the {VAR} values in the last {TIME} minutes; \n"
    msg_to_send += "- 'Real_Pred {VAR} {TIME}' - will reply with a plot where we can see the predicted VS real values from the {VAR}, within the last {TIME} minutes; \n"
    msg_to_send += "- 'Error_Hist {VAR} {TIME}' - returns an histogram where we can see the distribution of the {VAR} errors/differences from the last {TIME} minutes. \n"
    # msg_to_send += "- 'Last Excel Report' - sends back the latest excel report with the anomalies."

    return msg_to_send


def help_message_request(message):    # need to write help
   request = message.text.split()
   if request[0].lower() not in "help":
      return False
   else:
      return True


# j = updater.job_queue
# # Auto message sending
# def send_msg_on_telegram(message, context: CallbackContext):
#     msg = 'Hello World'
#     bot.send_message(chat_id = message.chat.id, text = msg)
    # telegram_api_url = f"https://api.telegram.org/bot{API_token}/sendMessage?text={message}"
    # tel_resp = requests.get(telegram_api_url)


# -----------------------------------------------------------------------------------------
# BOT MESSAGE HANDLERS (REGARDING THE FUNCTIONS CREATED ABOVE)

# returning last N vals
@bot.message_handler(func = last_n_vals_request)
def send_last_n_data(message):
    number_readings = message.text.split()[1]
    var = message.text.split()[2]
    # executing the function to retrieve last N values from the specific variable from the DB
    df = last_n_vals(n_vals = number_readings, var = var)
    df.set_index('Date', inplace = True)
    bot.send_message(message.chat.id, df[f'{var} Values'].to_string(header = False))

# retunring a graph of a variable's behaviour
@bot.message_handler(func = plot_var_requested_request)
def send_var_graph(message):
    var = message.text.split()[1]
    time = message.text.split()[2]
    # plotting the graph
    plot_var_requested(var, time)
    # sending the plot
    bot.send_photo(message.chat.id, photo = open(single_plot_directory, 'rb'))

# retunring a graph of a variable's real vs prediction behaviour
@bot.message_handler(func = plot_real_pred_request)
def send_real_pred(message):
    var = message.text.split()[1]
    time = message.text.split()[2]
    # plotting the graph
    plot_real_pred(var, time)
    # sending the plot
    bot.send_photo(message.chat.id, photo = open(real_pred_directory, 'rb'))

# retunring a graph of a variable's real vs prediction behaviour
@bot.message_handler(func = get_last_excel_request)
def send_last_excel(message):
    excel_file = get_last_excel()
    # generating the correct directory of the file
    aux = excel_file.split("\\")
    filepath = aux[0] + "/" + aux[1]
    filename = aux[1]
    bot.send_document(message.chat.id, document = open(filepath, 'rb'), filename = filename)

# retunring an histogram of the last TIME mins errors
@bot.message_handler(func = error_hist_request)
def send_hist(message):
    var = message.text.split()[1]
    time = message.text.split()[2]
    # plotting the graph
    error_hist(var, time)
    # sending the plot
    bot.send_photo(message.chat.id, photo = open(single_plot_directory, 'rb'))

# returning a list of the variables
@bot.message_handler(func = get_vars_request)
def send_vars(message):
    vars = list(VARIABLES)
    msg_to_send = f"Here's the list of variables being monitored: \n"
    for i in range(len(vars)):
        if i != len(vars):
            msg_to_send = msg_to_send + f"{vars[i]}, "
        else:
            msg_to_send = msg_to_send + f"{vars[i]}."
    bot.send_message(message.chat.id, msg_to_send)




# start reply
@bot.message_handler(commands = ['start'])
def greet(message):
    bot.send_message(message.chat.id, "Hello! How can I be useful to you?")



# help command
@bot.message_handler(func = help_message_request)
def help_function(message):
    msg_to_send = help_message()
    # Returning the message
    bot.send_message(message.chat.id, msg_to_send)


    # while True:
    # waiting for requests
bot.polling()
    # j.run_once(once, 10)
    # time.sleep(10)

