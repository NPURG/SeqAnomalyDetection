# preprocess settings
DEBUG = False #True
EPS = 0.00000001
FEATURES_NUM = 2 #not for single <arg>
RANGE_N = 97600 #<arg> #-1 2000 10000 5000000 500000
DEBUG_PLOT_RANGE = 10000#2000
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
# RANDOM_SEED=42 #<arg>
SCALER_SUFFIX = ['', '_log2', '_sin', '_std', '_std_scaled', '_std_log2', '_std_sin', '_std_log2_sin', '_l1', '_l2']
SCALED_DATA = ['xc_scaled', 'xc_log2', 'xc_sin', 'xc_std', 'xc_std_scaled', 'xc_std_log2', 'xc_std_sin','xc_std_log2_sin', 'xc_l1', 'xc_l2']
SCALER_SUFFIX_SINGLE = ['']
SCALED_DATA_SINGLE = ['xc_scaled']
USE_PURGED = False

# main settings
RANDOM_SEED = 42
EPOCHS = 10 #15 #20#5 25 10 50
BATCH_SIZE = 32 #2048*3 #128 #10000

DATA_FILE_SUFFIX = "" #"_std_log2_sin"
## DATA_FILE_SUFFIX_SINGLE = "_purged"

# USAD settings
USAD_ALPHA = 0.9#0.6 * l(ae1s, data) + 0.4 * l(ae2ae1s, data)#0.8 * l(ae1s, data) + 0.2 * l(ae2ae1s, data)#0.4 * l(ae1s, data) + 0.6 * l(ae2ae1s, data) #0.1 0.9 #<arg>
USAD_BETA = 0.1

# USAD_BiLSTM_VAE settings
USAD_BLV_ALPHA = 0.5
USAD_BLV_BETA = 0.5
KL_WEIGHT = 0.5

# LR settings
LR_STEP_SIZE = 3

# tensorboard settings
LOGPATH = "logs"
# special tokens usage: SPECIl_TOKEN_BASE + special_tokens['<TOK>']
special_tokens = {
    "<CLS>" : 0,
    "<SEP>" : 1,
    "<MASK>" : 2,
    "<PAD>" : 3,
    "<BOS>" : 4,
    "<EOS>" : 5,
}
SPECIAL_TOKEN_BASE = 128 # max(tokens.values())

# datasetcfg = {
#     'addr1394': {
#         'initargs':{  
#             'data_dir': 'data/addr1394',
#             'window_size': 1000, #12 #+
#             'train': True,
#             'truncate': 500000,#256, 50000
#             'transform': None,
#             'split': SPLIT_RATIO,
#             'mask_col': 'DST'
#         },
#     },
#     'window_generator': {
#         'initargs': {
#             # window_size = input_width + shift(offset) = input_width + label + bias
#             #                 label
#             #                  /+\
#             # 0 1 2 3 4 5 6 7 8 9
#             #\-+-+-+-/\+-+-+-+-+/
#             #  input  shift(offset)
#             'input_width': 990, # input_width + shift(offset) = window_size #+
#             'label_width': 10,  # offset - label_width = bias between input and label #+
#             'shift': 10, # NOTE: window shift(offset) >= label_width #+
#             'label_columns': ['MASKEDTOK', 'MASKED'], #['DST'],# for addr1394-single
#             'train_columns': ['MASKEDTOK', 'MASKED'], #['DST'], # for addr1394
#             'dtype': int # data type
#         },
#         'make_dataloader': {
#             'batch_size': BATCH_SIZE,
#             'shuffle': True,
#             'num_workers': 8
#         }
#     }
# }

# modelcfg = {
#     # 'TS_Embedding': {
#     #     'initargs': {
#     #         'vocab_size': 256 + len(special_tokens),
#     #         'd_model': 32, #128,
#     #         'max_len': 50,
#     #         'dpout': 0.1
#     #     }
#     # },
#     'BERT':{
#         # BERT Parameters
#         'max_len' : 2000, # NOTE: max_len >= window_size # 2000, # 50 max_len of input sequence #+
#         'vocab_size': 256 + len(special_tokens),
#         #'max_pred' : 5, # max tokens of prediction
#         'n_layers' : 6,
#         'n_heads' : 8,
#         'd_model' : 32,
#         'd_ff' : 32 * 4, # 4*d_model, FeedForward dimension
#         'd_k' : 16, 
#         'd_v' : 16,  # dimension of K(=Q), V
#         'n_segments' : 2,
#         'dp_out' : 0.1
#     }
# }


datasetcfg = {
    'addr1394': {
        'initargs':{  
            'data_dir': 'data/addr1394',
            'window_size': 1000, #12 #+
            'train': True,
            'truncate': 500000,#256, 50000
            'transform': None,
            'split': SPLIT_RATIO,
            'mask_col': 'DST'
        },
    },
    'window_generator': {
        'initargs': {
            # window_size = input_width + shift(offset) = input_width + label + bias
            #                 label
            #                  /+\
            # 0 1 2 3 4 5 6 7 8 9
            #\-+-+-+-/\+-+-+-+-+/
            #  input  shift(offset)
            'input_width': 800, #700 990 input_width + shift(offset) = window_size #+
            'label_width': 200,  #300 10 offset - label_width = bias between input and label #+
            'shift': 200, # 10 NOTE: window shift(offset) >= label_width #+
            'label_columns': ['MASKEDTOK', 'MASKED'], #['DST'],# for addr1394-single
            'train_columns': ['MASKEDTOK', 'MASKED'], #['DST'], # for addr1394
            'dtype': int # data type
        },
        'make_dataloader': {
            'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': 2
        }
    }
}

modelcfg = {
    # 'TS_Embedding': {
    #     'initargs': {
    #         'vocab_size': 256 + len(special_tokens),
    #         'd_model': 32, #128,
    #         'max_len': 50,
    #         'dpout': 0.1
    #     }
    # },
    'BERT':{
        # BERT Parameters
        'max_len' : 1008, # NOTE: max_len >= window_size # 2000, # 50 max_len of input sequence #+
        'vocab_size': 256 + len(special_tokens),
        #'max_pred' : 5, # max tokens of prediction
        'n_layers' : 6,
        'n_heads' : 6,
        'd_model' : 10,
        'd_ff' : 10 * 4, # 4*d_model, FeedForward dimension
        'd_k' : 8, 
        'd_v' : 8,  # dimension of K(=Q), V
        'n_segments' : 2,
        'dp_out' : 0.1
    }
}
