# test
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tsdatasets import *
from settings import *

if __name__ == '__main__':
    # test Addr1394Dataset
    # addr1394 = Addr1394Dataset()
    # print(addr1394[0])
    # test WindowGenerator
    window = WindowGenerator(
        input_width=datasetcfg['window_generator']['initargs']['input_width'],
        label_width=datasetcfg['window_generator']['initargs']['label_width'], 
        shift=datasetcfg['window_generator']['initargs']['shift'], 
        dtype=datasetcfg['window_generator']['initargs']['dtype'], 
        train_columns=datasetcfg['window_generator']['initargs']['train_columns'],
        label_columns=['MASKED' ,'DST'],
        dataset_initarg=datasetcfg['addr1394']['initargs'],
        batch_size=datasetcfg['window_generator']['make_dataloader']['batch_size'], 
        shuffle=datasetcfg['window_generator']['make_dataloader']['shuffle'],
        num_workers=datasetcfg['window_generator']['make_dataloader']['num_workers']
	)
    print(window)
    window.change_column(['DST', 'MASKED', 'MASKEDTOK'],['DST', 'MASKED', 'MASKEDTOK'])
    dl_train = window.train_dataset
    dl_test = window.test_dataset
    # print("train:", dl_train)
    for x, y in iter(dl_train):
        # x, y = p
        print("x: " , x)
        print(x.shape)
        print("y: " , y)
        print(y.shape)
        break
    print("train:", dl_test)
    for x, y in iter(dl_test):
        # x, y = p
        print("x: " , x)
        print(x.shape)
        print("y: " , y)
        print(y.shape)
        break