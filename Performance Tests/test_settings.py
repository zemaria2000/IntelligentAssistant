# ------------------------------------------------------------------------------------- #
# ---------------------------- MACHINE LEARNING PARAMETERS ---------------------------- #

# percentage of the dataset used for training
TRAIN_SPLIT = 0.9
# Number of timestamps to look back in order to make a prediction
PREVIOUS_STEPS = 30

# Variables to predict
VARIABLES = {
    'P_SUM',
    'U_L1_N',
    'I_SUM',
    'H_TDH_I_L3_N',
    'F',
    'ReacEc_L1',
    'C_phi_L3',
    'ReacEc_L3',
    'RealE_SUM',
    'H_TDH_U_L2_N'
}
# Variables to which a linear regression is more suitable
LIN_REG_VARS = {
    'RealE_SUM', 
    'ReacEc_L1', 
    'ReacEc_L3'
}

# ------------------------------------------------------------------------------------- #
# ------------------------------------ SOME DIRECTORIES ------------------------------- #

# Directories
DATA_DIR = '../Directories/Datasets/'
MODEL_DIR = '../Directories/Models/'
SCALER_DIR = '../Directories/Scalers/'


