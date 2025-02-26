DEBUG = False
EPS = 0.00000001
FEATURES_NUM = 2 #not for single <arg>
RANGE_N = 5000000 #<arg> #-1 2000 10000 5000000 500000
DEBUG_PLOT_RANGE = 2000
USE_RATIO=False
TEST_NUM = 1000 #<arg>
SPLIT_RATIO = 0.8 #0.7 #0.5 #<arg>
DISTURB_SCALE = 1.5 #1.0 #255 #0.25 #<arg>
DISTURB_PROBABILITY = 0.01 #0.01 #0.05 0.02 #<arg> 0.01
DISTURB_N_THRESHOLD_MIN = 0.4 #0.3 #0.2 <arg>
DISTURB_N_THRESHOLD_MAX = 1.0 #0.8 #<arg>
EXCHANGE_NUM = 5 #<arg>100 0 20
EXCHANGE_DISTURB_THRESHOULD = 0.3
ERROR_SPLIT_PROBABLITY = 0.5 #<arg>
RANDOM_SEED=42 #<arg>
SCALER_SUFFIX = ['', '_log2', '_sin', '_std', '_std_scaled', '_std_log2', '_std_sin', '_std_log2_sin', '_l1', '_l2']
SCALED_DATA = ['xc_scaled', 'xc_log2', 'xc_sin', 'xc_std', 'xc_std_scaled', 'xc_std_log2', 'xc_std_sin','xc_std_log2_sin', 'xc_l1', 'xc_l2']
SCALER_SUFFIX_SINGLE = ['']
SCALED_DATA_SINGLE = ['xc_scaled']
# main settings
EPOCHS = 15 #20#5 25 10 50
DATA_FILE_SUFFIX = "" #"_std_log2_sin"
DATA_FILE_SUFFIX_SINGLE = "_purged"
# USAD settings
USAD_ALPHA = 0.9#0.6 * l(ae1s, data) + 0.4 * l(ae2ae1s, data)#0.8 * l(ae1s, data) + 0.2 * l(ae2ae1s, data)#0.4 * l(ae1s, data) + 0.6 * l(ae2ae1s, data) #0.1 0.9 #<arg>
USAD_BETA = 0.1
# USAD_BiLSTM_VAE settings
USAD_BLV_ALPHA = 0.5
USAD_BLV_BETA = 0.5
KL_WEIGHT = 0.5
BATCH_SIZE = 128 #128