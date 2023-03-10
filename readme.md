
===================================================================================
CREATING OUR VIRTUAL ENVIRONMENT

1. Navigate to this specified folder:
    1.1. Start Ubuntu (Searchbar -> Ubuntu)
    1.2. Change to this specified folder (go to the start directory, than 'cd mnt', than cd until this folder)
2. Create and run the virtual environment:
    - python3 -m venv venv (creating it) (if we don't have the needed package, we'll have to do 'apt install python3.10-venv')
    - source venv/bin/activate (activating it)
3. Install all the specified requirements to run the necessary scripts
    - pip install -r requirements.txt / pip install -r my_requirements.txt
4. Run (hopefully...) the scripts we want

IMPORTANT!!!
I only managed to pip install the scikit-learn library when I set my wsl python version to 3.9. With version 3.10 I was always getting an error that wouldn't allow me to install the library

====================================================================================
SCRIPTS THAT EXIST IN THIS FOLDER        
    
    - 'train_all_models.sh' - shell script that we can run in our command line, and executes automatically a series of instructions. In this specific case, 
    it runs the model_training.py script. It is worth noting that that script requires a arg.parse argument (specifying the model to be trained). That's why 
    it does 'python3 model_training.py model': it trains model by model from the list given

    - 'pedro_requirements.txt' - this folder is the one already created by Pedro. It has all the needed libraries and respective functions that allow us to run all the python scripts in this folder 

    - 'requirements.txt' - this folder, created by me, is made from scratch and has all the requirements that I think are needed to run the files that I', experimenting

    - 'settings.py' - it contains a bunch of different settings that are constantly "pulled" by the different scripts. It has:
        + CORR_GROUP - list key with all the variables to predict (P_SUM, U_L1_N, etc) and the respective features that they depend on (some variables depend
        on up to 20 variables, others only 1 or 2);
        + PRED_MODELS - for each variable, we have a model that needs to be trained and implemented;
        + DATA_DIR - it just contains the directory where some of the csv train data files are stored (one for each variable to predict);
        + AD_THRESHOLD - for each variable to predict, it defines a FIXED threshold. Above this threshold, we consider the measurement to be an anomaly;
        + LINREG_LIST - list with the variables that only use a simple linear regression model (all the others, apparently, use autoML);
        + INFLUXDB_... - some basic settings regarding the inlfuxDB bucket (user, password, host, port...)

    - 'model_training.py' - script that allows us to train the ML models. This script is ran automatically through the 'train_all_models.sh' one. Basically we pass the model we want to train, the script looks for the training file with all the needed data (csv file in \data directory). Then, it creates the dataset based on that data, splits the data in training and testing data, and finally runs the appropriate training process. In the end, it calculates the RMSE (Root Mean Squared Error), that allows us to see the error between the predicitions our trained model did in the testing data, and the "real" data.

    - ´prediction.py' - This script has 2 main functions - one destined to make predictions based on some values it collects form an influxDB repository (e.g. from last 15 minutes), and one destined to make anomaly detection. More specifically:
        + prediction_preogram() - it starts by getting last timestamp's variable values frome the database. Then, it loads the models that were trained with 'model_training.py', and applies the trained model to make a prediction one timestamp into the future (in the case of this problem, 1 minute). It finishes off by sending the prediction value to the database, through an InfluxDB query.
        + ad_program() - this function will load the current values that are being sent from the equipment in real time, and will load the prediction made by the predictor. Then, it will compare the results of both of them, and calculate their difference (real value VS forecasted value). Finally, it will send this difference to the InfluxDB database.
        (it is worth noting that I cannot run this script until I at least have some values to put in InfluxDB. It might not be the real time values, just some fixed values to see if the program works fine)

    - 'get_training_data.py' - this script will connect to our influx database, and gather data regarding each variable to predict and each feature. Then, it will create a scaler and normalize that data (basically it will automatically pre-process our data). The scalers for each feature will be stored in the 'new_scalers/{var}.scale' direc
    
    - 'anomaly_detection.py' - this script is already written in the 'prediction.py' one. In this one, we can only do the anomaly detection, whereas in the 'prediction.py', we can make the anomaly detections and also make the predictions. This script can be omitted, because it already runs inside our 'prediction.py'.
    
    - 'forecasting_agent.py' - this script is already written in the 'prediction.py' one. In this one, we can only do the forecasting of future values, whereas in the 'prediction.py', we can make the anomaly detections and also make the predictions. This script can be omitted, because it already runs inside our 'prediction.py' - CREATED BY ME

    - 'main.sh' - shell script created by me that tries to run all the above mentioned python scripts in the order defined bellow - CREATE BY ME 

====================================================================================
RUNNING THE SCRIPTS

Analysing each script, we can identify a logical sequence to run them:
    1. ´get_training_data.py' - this is the 1st script to run, where we'll connect to our database and collect some data. That data will be stored in a specific folder and then be used to train our algorithms
    2. 'model_training.py' or 'train_all_models.sh' - we have two options:
        2.1. running the 'model_training.py', where we need to specifiy a certain model to train (basically do 'python3 model_training.py P_SUM', for instance), and we train an individual model
        2.2. running the 'train_all_models.sh', a shell script that allows us to run train all the models one by one (we do a for cycle for each model, that runs the 'model_training.py')
    3. 'prediction.py' - finally, after the models are trained, we'll run the main script. In this case, inside our 'prediction.py' file we have the 'anomaly_detection.py' script, that makes the anomaly detections, and also a 'predicting script', which makes predictions based on the models training. Inside this 'prediction.py', we probably need to first run the prediction function first, to then have forecasted values that can be compared with the real values, in the anomaly detection function.

    (instead of running the 'prediction.py, what we can do is run the 'anomaly_detection.py' and the 'forecasting_agent.py' seperatel...)

I also created a new sheel script ('main.sh'), that contains all the abovementioned sequential steps to run the scripts


====================================================================================
SOME CONCERNS/QUESTIONS THAT I HAVE...

CONCERNS
1. I need to somehow have access to data from the InfluxDB (that I think belongs to the compressor use case in Bosch's shop floor) in order to see if these scripts are working correctly
2. Don't know if my main.sh shell script will work properly
3. Obviously the order in the "RUNNING THE SCRIPTS" section is the order that makes most sense to me, after analysing all the scripts. Can't confirm 100% that it's the right order
4. Maybe we can seperate the forecasting/predicting agent and the anomaly detection one. And run them seperately and simultaneously...


QUESTIONS
1. In the 'prediction.py' script (and also in the 'forecasting_agent.py'), there's a line of code where we load a file from the directory 'models/...'. There's reference to that directory in the 'model_training.py' script, but the lines that used it were commented by himself...
2. Some doubts in some parts of some scripts... They are mentioned in the scripts