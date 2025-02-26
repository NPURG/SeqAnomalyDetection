# Licence: MIT
# Author: Kevin Huang
# Date: 2024-03

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import pandas as pd
import numpy as np
from settings import *

class TimeseriesDataset(Dataset):
    def __init__(self, data, window, transform=None):
        # self.factor = factor
        self.data = data
        self.window = window
        self.transform = transform
    
    def __len__(self):
        return len(self.data) - self.window + 1
    
    def __getitem__(self, index):
        # print("getitem called")
        if index < 0:
            index += len(self)
        # features = self.data[index*self.factor:index*self.factor+self.window]
        features = self.data[index:index+self.window]
        if self.transform is not None:
            return self.transform(features)
        return features

class Addr1394Dataset(TimeseriesDataset):
    '''Addr 1394 dataset'''
    def __init__(self,
                 data_dir, 
                 window_size,
                 train,
                 truncate, 
                 transform,
                 split,
                 mask_col='DST',
                 dst_file='channels_1394_DST.csv',
                 id_file='channels_1394_ID.csv'):
        # dataset_folder = 'data/addr1394'
        # 1394 protocal
        self.data_dir = data_dir
        self.window = window_size
        self.train = train
        self.transform = transform
        dst_data = pd.read_csv(os.path.join(data_dir, dst_file))
        id_data = pd.read_csv(os.path.join(data_dir, id_file))
        self.split = split
        self.data = pd.concat([dst_data, id_data], axis=1)\
                            .apply(lambda x: x.astype(str).map(lambda x: int(x, base=16)))
        # train test split
        self.data = self.data[:int(len(self.data)*self.split)] if self.train else self.data[int(len(self.data)*self.split):]
        # truncate the data
        self.data = self.data[:truncate] if truncate is not None else self.data

        self.data['DISTURBED'] = self.data['DST'].apply(lambda x: x if np.random.rand() > DISTURB_PROBABILITY else np.random.randint(0, 255))

        self.mask_id = SPECIAL_TOKEN_BASE + special_tokens['<MASK>']
        self.pad_id = SPECIAL_TOKEN_BASE + special_tokens['<PAD>']
        
        self.data['MASKED'], self.data['MASKEDTOK'] = self.get_masked_data(col=mask_col)

        super(Addr1394Dataset, self).__init__(self.data, self.window, transform=self.transform)
    def get_masked_data(self, col='DST'):
        # mask 15% of the data, candidate positions
        cand_pos = np.random.choice(self.data.index, int(len(self.data.index) * 0.15), replace=False)
        np.random.shuffle(cand_pos)
        masked_data = self.data[col].copy()
        masked_tok = np.zeros_like(self.data[col].values)
        masked_tok.fill(self.pad_id) # padding <PAD> as the default value in unmasked token thus ignored in cross_entropy_loss (ignore=pad_id)
        for pos in cand_pos:
            if np.random.rand() < 0.8: # 80% <mask>
                masked_tok[pos-masked_data.index[0]] = masked_data.loc[pos]
                masked_data.loc[pos] = self.mask_id
            elif np.random.rand() > 0.9: # 10% random replace
                masked_tok[pos-masked_data.index[0]] = masked_data.loc[pos]
                masked_data.loc[pos] = np.random.randint(0, 255)
            # else 10% keep the same
        return masked_data, masked_tok
    def get_seg_swap(self, tc):
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
            
        #     # if np.random.rand(1)[0] < disturb_probability: # !change thedistrubution of disturb errors' choices here
        #     #DISTURB_CENTERED
        #     if i in [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,2930,31,32,33,34,35,36,37,38,39,40,
        #              510,511,512,513,514,515,516,517,518,519,520,521,522,523,524525,526,527,528,529,530,
        #              610,611,612,613,614,615,616,617,618,619,620,621,622,623,624625,626,627,628,629,630, 
        #              710,711,712,713,714,715,716,717,718,719,720,721,722,723,724725,726,727,728,729,730,    
        #              100,101,102,103,104,500,501,502,503,504,505,506,800,801,802803]: # !change the distrubution of disturb errors' choices here
        #     # if np.random.randn(1)[0] > 2.1: # disturb_probability:
        #         # if np.random.rand(1)[0] < error_split_probablity or i == 0 or i== len(tc)-1:
        #         d = disturb_scale * np.random.random(1)[0]
        #         while abs(d) < disturb_n_threshold_min or abs(d) >disturb_n_threshold_max or abs(d) == ttc[:,0][i]:
        #             d = disturb_scale * np.random.random(1)[0] #disturb_scale *np.abs(np.random.randn(1)[0]) + 0.3 #(np.random.rand(1)[0] + 0.5) * 2 *disturb_scale
        #         ttc[i] += d  
        #         ttc[i] = abs(ttc[i])

        #         # d = disturb_scale * np.random.randn(1)[0]
        #         # while d < disturb_n_threshold_min or d >disturb_n_threshold_max or d == ttc[i]:
        #         #     d = disturb_scale * np.random.randn(1)[0]
        #         # ttc[i] = d    
        #         # labelsc.append(np.array([1.0, 0.0])) #1 # abnormal
        #         # print(ttc[i])
        #         labelsc[i] = 1 #False #1 # abnormal [-TODO](+DONE:shift USAD andTrainAD output)：comment for test
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
        #         #labelsc.append(np.array([0.0])) #0#[-TODO](+DONE:shift USAD andTrainAD output)：comment for test
        #         disturbc.append(0.0)#[-TODO](+DONE:shift USAD and TrainADoutput)：comment for test

        # abnormal: exchange
        
        from_index_pairs = [(10, 10+50), (210, 210+50), (410, 410+50), (710, 71+50)]
        to_index_pairs =   [(110, 110+50), (320, 320+50), (500, 500+50), (810, 81+50)]
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
        return ttc, labelsc, labelsc_nc, disturb_indices, disturbc
    def get_random_disturb(self, tc):
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
    
        

class WindowGenerator:
    def __init__(
        self, 
        input_width, 
        label_width, 
        shift, # offset
        dtype, 
        train_columns,
        label_columns,
        batch_size,
        shuffle,
        num_workers,
        dataset_initarg,
        dataset_name="Addr1394Dataset",
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dtype = dtype
        self.dataset_name = dataset_name
        self.dataset_initarg = dataset_initarg
        self._dataset : TimeseriesDataset= eval(dataset_name)(**dataset_initarg)
        # 存储原始数据.
        
        # 找出标签列的下标索引.
        self.columns = self._dataset.data.columns
        if label_columns is None:
            self.label_columns = self.columns
            self.train_columns = self.columns
        else: 
            self.label_columns = pd.Index(label_columns)
            self.train_columns = pd.Index(train_columns)
        self.label_column_indices = [
            self.columns.get_loc(name) for name in self.label_columns
        ]
        self.train_column_indices = [
            self.columns.get_loc(name) for name in self.train_columns
        ]
        
        # 计算窗口的参数.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(input_width)
        self.input_indices = np.arange(input_width)
        
        self.label_start = self.total_window_size - label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.label_start, self.total_window_size)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column names(s): {self.label_columns.to_list()}'
        ])

    def change_column(self, train_columns, label_columns):
        self.label_columns = pd.Index(label_columns)
        self.train_columns = pd.Index(train_columns)
        self.label_column_indices = [
            self.columns.get_loc(name) for name in self.label_columns
        ]
        self.train_column_indices = [
            self.columns.get_loc(name) for name in self.train_columns
        ]

    def split_window(self, features):
        inputs = features[self.input_slice, self.train_column_indices]
        labels = features[self.label_slice, self.label_column_indices]
        # print("features type: ", type(features))
        # print("size input: ", inputs.shape)
        # print("size label: ", labels.shape)
        
        return inputs, labels
      
    def make_dataloader(self,
                        train):
        
        __dataset_initarg : dict= self.dataset_initarg.copy()
        __dataset_initarg['train'] = train
        __dataset_initarg['transform'] = self.split_window # the key is replace the transform to split_window
        __dataset : TimeseriesDataset = eval(self.dataset_name)(**__dataset_initarg)
        # print("test_transform: ", __dataset.transform)
        # print("test_train: ", __dataset.train)
        # print("data type: ", type(__dataset.data))
        __dataset.data = __dataset.data.to_numpy()
        # print("data type aft: ", type(__dataset.data))
        
        dataloader = DataLoader(
            dataset=__dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
        
        return dataloader
    
    @property
    def train_dataset(self):
        return self.make_dataloader(True)

    @property
    def test_dataset(self):
        return self.make_dataloader(False)

    @property
    def example(self):
        '''获取并缓存一个批次的(inputs, labels)窗口.'''
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self._dataset))
            self._example = result
        return result
    
