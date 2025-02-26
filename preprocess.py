import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile
from sklearn import preprocessing
import matplotlib.pyplot as plt
from settings import *
import random

DEBUG = True

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB','addr1394', 'addr1394-single']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(':')[0], line.split(':')[1].split(',')
        start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
        temp[start-1:end-1, indx] = 1
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
    if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
    x = df[df.columns[3:]].values[::10, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_data(dataset, data_hook=None):
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    if dataset == 'addr1394':
        dataset_folder = 'data/addr1394'
        features_num = FEATURES_NUM #<arg>
        #~Read channel data (address)
        # 1394 protocal
        df_dst = pd.read_csv(os.path.join(dataset_folder,"channels_1394_DST.csv"))#目的地址（*）
        df_id = pd.read_csv(os.path.join(dataset_folder,"channels_1394_ID.csv"))   
        df_comb = pd.concat([df_dst, df_id], axis=1)
        channel = df_comb.apply(lambda x: x.astype(str).map(lambda x: int(x, base=16))).astype(float)
        
        # channel = pd.read_csv(os.path.join(dataset_folder,'channel.csv'), header=None)
        # channel = channel.apply(lambda x: x.astype(str).map(lambda x: int(x, base=16)))
        # channel = channel.astype(float)
        
        #~Normalization
        range_n = RANGE_N #<arg>
        xc = channel.values[:range_n] #[0:8000] #[0:5000] #[0:3000] #[0:2000]  # [0:1500] #cut values
        if DEBUG:
            print("xc: ", xc.shape)
        #TODO: test linear/non-linear scaler
        xc_scaled = preprocessing.MinMaxScaler().fit_transform(xc) #!!
        xc_log2 = np.log2(xc_scaled + 1) #!!
        xc_sin = np.sin(xc_scaled * np.pi) #!!
        xc_std = preprocessing.StandardScaler().fit_transform(xc) #!!
        xc_std_scaled = preprocessing.MinMaxScaler().fit_transform(xc_std)
        xc_std_log2 = np.log2(np.array([n if n.all() != 0 else n + EPS for n in xc_std_scaled]) + 1)
        xc_std_sin = np.sin(xc_std_scaled * np.pi)
        xc_std_log2_sin = np.sin(xc_std_log2 * np.pi)
        xc_l1 = preprocessing.normalize(xc, norm='l1') #!!
        xc_l2 = preprocessing.normalize(xc, norm='l2') #!!

        for name, xc_scaled in zip(SCALER_SUFFIX, map(eval, SCALED_DATA)):
            if DEBUG:
                print(xc_scaled.shape)
                # plt.plot(xc[:DEBUG_PLOT_RANGE], label='xc'+name)
                plt.plot(xc_scaled[:DEBUG_PLOT_RANGE,0], label='xc_scaled_DST'+name)
                plt.plot(xc_scaled[:DEBUG_PLOT_RANGE,1], label='xc_scaled_ID'+name)
                plt.legend()
                plt.show()
            ##
            if USE_RATIO is True:
                split_ratio = SPLIT_RATIO #0.7 #0.5 #<arg>
                tc = xc_scaled[int(len(xc_scaled) * split_ratio):]
                xc = xc_scaled[:int(len(xc_scaled) * split_ratio)]
            else:
                test_num = TEST_NUM#<arg>
                #TODO: this tc is only for test
                tc = xc_scaled[:test_num]
                # tc = xc_scaled[-test_num:]
                xc = xc_scaled[:-test_num]
            print("train"+name+" shape:", xc.shape)
            print("test"+name+" shape:", tc.shape)
            if DEBUG:
                # plt.plot(xc[0:500])
                # plt.plot(tc[0:500])
                # plt.show()

                plt.plot(tc[:1000], label='tc'+name)
                # plt.plot(tc_scaled[:1000], label='tc_s')
                plt.legend()
                plt.show()
            ##

            #Label generatoin for tc
            disturb_scale = DISTURB_SCALE #255 #0.25 #<arg>
            disturb_probability = DISTURB_PROBABILITY #0.05 0.02 #<arg> 0.01
            disturb_n_threshold_min = DISTURB_N_THRESHOLD_MIN #0.2 <arg>
            disturb_n_threshold_max = DISTURB_N_THRESHOLD_MAX #0.8 #<arg>
            error_split_probablity = ERROR_SPLIT_PROBABLITY #<arg>
            # dd = 1
            disturbc = []
            labelsc = np.zeros_like(tc)
            ttc = tc.copy()
            np.random.seed(RANDOM_SEED) #<arg>

            for i, _ in enumerate(tc):
                # abnormal: disturb
                if np.random.rand(1)[0] < disturb_probability:
                # if np.random.randn(1)[0] > 2.1: # disturb_probability:
                    # if np.random.rand(1)[0] < error_split_probablity or i == 0 or i == len(tc)-1:
                    d = disturb_scale * np.random.random(1)[0]
                    while abs(d) < disturb_n_threshold_min or abs(d) > disturb_n_threshold_max or abs(d) == ttc[:,0][i]:
                        d = disturb_scale * np.random.random(1)[0] #disturb_scale * np.abs(np.random.randn(1)[0]) + 0.3 #(np.random.rand(1)[0] + 0.5) * 2 * disturb_scale
                    ttc[:,0][i] += d  #TODO: disturb_scale
                    ttc[:,0][i] = abs(ttc[:,0][i])

                    # d = disturb_scale * np.random.randn(1)[0]
                    # while d < disturb_n_threshold_min or d > disturb_n_threshold_max or d == ttc[i]:
                    #     d = disturb_scale * np.random.randn(1)[0]
                    # ttc[i] = d
                    # labelsc.append(np.array([1.0, 0.0])) #1 # abnormal
                    # print(ttc[i])
                    labelsc[i,0] = 1 #False #1 # abnormal [-TODO](+DONE:shift USAD and TrainAD output)：comment for test
                    disturbc.append(d)
                    # else:
                    #     # swap error
                    #     temp = tc[i + 1]
                    #     tc[i + 1] = tc[i]
                    #     tc[i] = temp
                    #     labelsc.append(np.array([1.0]))
                else:
                    #labelsc.append(np.array([0.0])) #0#[-TODO](+DONE:shift USAD and TrainAD output)：comment for test
                    disturbc.append(0.0)#[-TODO](+DONE:shift USAD and TrainAD output)：comment for test

            # abnormal: exchange
            ## for i in range(EXCHANGE_NUM):
            ##     index1 = np.random.randint(0, len(ttc[:,0]))
            ##     index2 = np.random.randint(0, len(ttc[:,0]))
            ##     tmp = ttc[index1, 0]
            ##     ttc[index1, 0] = ttc[index2, 0]
            ##     ttc[index2, 0] = tmp
            ##     labelsc[index1, 0] = 1
            ##     labelsc[index2, 0] = 1
            ##     disturbc[index1] = ttc[index2, 0] - ttc[index1, 0]
            ##     disturbc[index2] = -disturbc[index1]
            # print(labelsc[:200])
            # channel = pd.DataFrame(xc)
            # channel_ano = pd.DataFrame(tc)

            # print(channel.shape, channel_ano.shape)
            # channel.head(20)
            # channel_ano.head(50)
            if DEBUG:
                plt.figure(figsize=(12, 8))

                # Plot disturb
                plt.subplot(411)
                plt.plot(disturbc, label='disturb', linewidth=2, alpha=0.7)
                plt.legend()

                # Plot ttc
                plt.subplot(412)
                plt.plot(np.array(tc)[:,0], label='DST_ori'+name)
                plt.plot(np.array(ttc)[:,0], label='DST_dis'+name, linewidth=2, alpha=0.7)
                plt.legend()

                # Plot vari
                plt.subplot(413)
                plt.plot(np.array(ttc)[:,0] - np.array(tc)[:,0], label='vari', linewidth=2, alpha=0.7)
                plt.legend()

                plt.subplot(414)
                plt.plot(np.array(labelsc)[:,0], label='labels')
                plt.legend()
                plt.show()
            ###

            #################################################
            # train, min_a, max_a = normalize2(xc)
            # test, _, _ = normalize2(tc, min_a, max_a)
            train = np.array(xc).reshape((-1,features_num))#TODO：comment for test
            test = np.array(ttc).reshape((-1,features_num))#TODO：comment for test
            test_ori = np.array(ttc).reshape((-1,features_num))#TODO：comment for test
            labels = np.array(labelsc, dtype=float).reshape((-1,features_num)) #pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
            labels_nc = np.array(labelsc, dtype=float).reshape((-1,features_num)) #pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
            for file in ['train', 'test_ori', 'test', 'labels', 'labels_nc']: #TODO: 'test_ori', 'labels_nc' are only for test, not true
                np.save(os.path.join(folder, f'{file+name}.npy'), eval(file))
        if data_hook is not None:
            data_hook(locals())
    elif dataset == 'addr1394-single': #single class
        dataset_folder = 'data/addr1394'
        features_num = 1
        xc = np.load(os.path.join(dataset_folder, 'channels_1394_DST_purged_blanks.npy'))[:RANGE_N] #[0:8000] #[0:5000] #[0:3000] #[0:2000]  # [0:1500] #cut values
        if DEBUG:
            print("xc: ", xc.shape)
        #TODO: test linear/non-linear scaler
        xc_scaled = preprocessing.MinMaxScaler().fit_transform(xc.reshape(-1, features_num))#!!
        # xc_log2 = np.log2(xc_scaled + 1) #!!
        # xc_sin = np.sin(xc_scaled * np.pi) #!!
        # xc_std = preprocessing.StandardScaler().fit_transform(xc) #!!
        # xc_std_scaled = preprocessing.MinMaxScaler().fit_transform(xc_std)
        # xc_std_log2 = np.log2(np.array([n if n.all() != 0 else n + EPS for n in xc_std_scaled]) + 1)
        # xc_std_sin = np.sin(xc_std_scaled * np.pi)
        # xc_std_log2_sin = np.sin(xc_std_log2 * np.pi)
        # xc_l1 = preprocessing.normalize(xc, norm='l1') #!!
        # xc_l2 = preprocessing.normalize(xc, norm='l2') #!!


        for name, xc_scaled in zip(SCALER_SUFFIX_SINGLE, map(eval, SCALED_DATA_SINGLE)):
            name = name + "_purged"
            if DEBUG:
                print(xc_scaled.shape)
                plt.plot(xc[:DEBUG_PLOT_RANGE], label='xc'+name)
                plt.plot(xc_scaled[:DEBUG_PLOT_RANGE], label='xc_scaled'+name)
                plt.legend()
                plt.show()
            ##
            if USE_RATIO is True:
                split_ratio = SPLIT_RATIO #<arg>
                tc = xc_scaled[int(len(xc_scaled) * split_ratio):]
                xc = xc_scaled[:int(len(xc_scaled) * split_ratio)]
            else:
                test_num = TEST_NUM #<arg>
                tc = xc_scaled[-test_num:]
                xc = xc_scaled[:-test_num]
            print("train"+name+" shape:", xc.shape)
            print("test"+name+" shape:", tc.shape)
            if DEBUG:
                # plt.plot(xc[0:500])
                # plt.plot(tc[0:500])
                # plt.show()

                plt.plot(tc[:1500], label='tc'+name)
                # plt.plot(tc_scaled[:1000], label='tc_s')
                plt.legend()
                plt.show()
            ##

            #Label generatoin for tc
            disturb_scale = DISTURB_SCALE #255 #0.25 #<arg>
            disturb_probability = DISTURB_PROBABILITY #0.05 0.02 #<arg> 0.01
            disturb_n_threshold_min = DISTURB_N_THRESHOLD_MIN #0.2 <arg>
            disturb_n_threshold_max = DISTURB_N_THRESHOLD_MAX #0.8 #<arg>
            error_split_probablity = ERROR_SPLIT_PROBABLITY #<arg>
            # dd = 1
            disturbc = []
            labelsc = np.zeros_like(tc)
            labelsc_nc = np.zeros_like(tc) #labels for n_classes
            ttc = tc.copy()
            np.random.seed(RANDOM_SEED) #<arg>
            
            # # abnormal: disturb
            disturb_indices = []
            # for i, _ in enumerate(tc):
                
            #     # if np.random.rand(1)[0] < disturb_probability: # !change the distrubution of disturb errors' choices here
            #     #DISTURB_CENTERED
            #     if i in [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
            #              510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,
            #              610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630, 
            #              710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,    
            #              100,101,102,103,104,500,501,502,503,504,505,506,800,801,802,803]: # !change the distrubution of disturb errors' choices here
            #     # if np.random.randn(1)[0] > 2.1: # disturb_probability:
            #         # if np.random.rand(1)[0] < error_split_probablity or i == 0 or i == len(tc)-1:
            #         d = disturb_scale * np.random.random(1)[0]
            #         while abs(d) < disturb_n_threshold_min or abs(d) > disturb_n_threshold_max or abs(d) == ttc[:,0][i]:
            #             d = disturb_scale * np.random.random(1)[0] #disturb_scale * np.abs(np.random.randn(1)[0]) + 0.3 #(np.random.rand(1)[0] + 0.5) * 2 * disturb_scale
            #         ttc[i] += d  
            #         ttc[i] = abs(ttc[i])

            #         # d = disturb_scale * np.random.randn(1)[0]
            #         # while d < disturb_n_threshold_min or d > disturb_n_threshold_max or d == ttc[i]:
            #         #     d = disturb_scale * np.random.randn(1)[0]
            #         # ttc[i] = d    
            #         # labelsc.append(np.array([1.0, 0.0])) #1 # abnormal
            #         # print(ttc[i])
            #         labelsc[i] = 1 #False #1 # abnormal [-TODO](+DONE:shift USAD and TrainAD output)：comment for test
            #         labelsc_nc[i] = 1 # n_classes
            #         disturb_indices.append(i)
            #         disturbc.append(d)
            #         # else:
            #         #     # swap error
            #         #     temp = tc[i + 1]
            #         #     tc[i + 1] = tc[i]
            #         #     tc[i] = temp
            #         #     labelsc.append(np.array([1.0]))
            #     else:
            #         #labelsc.append(np.array([0.0])) #0#[-TODO](+DONE:shift USAD and TrainAD output)：comment for test
            #         disturbc.append(0.0)#[-TODO](+DONE:shift USAD and TrainAD output)：comment for test

            # abnormal: exchange
            
            from_index_pairs = [(10, 10+50), (210, 210+50), (410, 410+50), (710, 710+50)]
            to_index_pairs =   [(110, 110+50), (320, 320+50), (500, 500+50), (810, 810+50)]
            for from_pair, to_pair in zip(from_index_pairs, to_index_pairs):
                from_index = from_pair[0]
                to_index = to_pair[0]
                for i in range(from_pair[1] - from_pair[0]):
                    temp = ttc[from_index + i].copy()
                    ttc[from_index + i] = tc[to_index + i]
                    tc[to_index + i] = temp
                    labelsc[from_index + i] = 1
                    labelsc_nc[from_index + i] = 1
                    labelsc[to_index + i] = 1
                    labelsc_nc[to_index + i] = 2
                    disturb_indices.append(from_index + i)
                    disturbc.append(ttc[from_index + i] - tc[to_index + i])
                    disturb_indices.append(to_index + i)
                    disturbc.append(ttc[to_index + i] - tc[from_index + i])
            
            # exchange_indices = [10,110,210,310,410] #random.sample(range(len(ttc)), EXCHANGE_NUM)
            # print("exchange_indices: ", exchange_indices)
            # print("exchanged values: ", ttc[exchange_indices])
            # for i in range(EXCHANGE_NUM):
            #     index1 = exchange_indices[i]
            #     index2 = random.choice([x for x in range(len(ttc)) if ((x not in exchange_indices) 
            #                                                        and (x not in disturb_indices) 
            #                                                        and (np.abs(ttc[x][0] - ttc[index1][0]) > EXCHANGE_DISTURB_THRESHOULD))])
            #     exchange_indices.append(index2)
            #     if DEBUG:
            #         print('pre:', ttc[index1], ttc[index2])
            #     ttc[index1][0], ttc[index2][0] = ttc[index2][0], ttc[index1][0]
            #     labelsc[index1] = 1
            #     labelsc[index2] = 1
            #     labelsc_nc[index1] = 2
            #     labelsc_nc[index2] = 2
            #     disturbc[index1] = ttc[index2][0] - ttc[index1][0]
            #     disturbc[index2] = -disturbc[index1]
            #     if DEBUG:
            #         print(index1, index2)
            #         print(disturbc[index1], disturbc[index2])
            #         print(ttc[index1], ttc[index2])

            if DEBUG:
                print("----------------- shapes -----------------")
                print("train"+name+" shape:", xc.shape)
                print("test"+name+" shape:", tc.shape)
                print("labels"+name+" shape:", labelsc.shape)
                print("labels_nc"+name+" shape:", labelsc_nc.shape)
                print("disturb"+name+" shape:", np.array(disturbc).shape)
                print("------------------------------------------")
                plt.figure(figsize=(12, 8))

                # Plot disturb
                plt.subplot(411)
                plt.plot(disturbc, label='disturb', linewidth=2, alpha=0.7)
                plt.legend()

                # Plot ttc
                plt.subplot(412)
                plt.plot(np.array(tc), label='DST_ori'+name)
                plt.plot(np.array(ttc), label='DST_dis'+name, linewidth=2, alpha=0.7)
                plt.legend()

                # Plot vari
                plt.subplot(413)
                plt.plot(np.array(ttc) - np.array(tc), label='vari', linewidth=2, alpha=0.7)
                plt.legend()

                plt.subplot(414)
                plt.plot(np.array(labelsc_nc), label='labels')
                plt.legend()
                plt.show()
            ###
            plt.plot(labelsc_nc, label='labels_nc')
            # plt.plot(labelsc, label='labels')
            plt.show()
            #################################################
            # train, min_a, max_a = normalize2(xc)
            # test, _, _ = normalize2(tc, min_a, max_a)
            train = np.array(xc).reshape((-1,features_num))#TODO：comment for test
            test_ori = np.array(tc).reshape((-1,features_num))#TODO：comment for test
            test = np.array(ttc).reshape((-1,features_num))#TODO：comment for test
            labels = np.array(labelsc, dtype=float).reshape((-1,features_num)) #pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
            labels_nc= np.array(labelsc_nc, dtype=int).reshape((-1,features_num)) 
            for file in ['train', 'test_ori', 'test', 'labels', 'labels_nc']:
                np.save(os.path.join(folder, f'{file+name}.npy'), eval(file))
        if data_hook is not None:
            data_hook(locals())
    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            load_data(d)
    else:
        print("Usage: python preprocess.py <datasets>")
        print(f"where <datasets> is space separated list of {datasets}")