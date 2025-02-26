import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from settings import *
import os

dataset_folder = 'data/addr1394'

# 假设我们有一个时间序列数据
data = np.random.randn(1000)
# data = np.load(os.path.join(dataset_folder, 'channels_1394_DST_purged_blanks.npy'))[:RANGE_N]  #comm-addr1394-single:1 #[0:8000] #[0:5000] #[0:3000] #[0:2000]  # [0:1500] #cut values
# 进行白噪声检验
ljung_box_result = acorr_ljungbox(data, lags=[10], return_df=True)

# 输出检验结果
print(ljung_box_result)
