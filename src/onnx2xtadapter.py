import numpy as np
import onnx
from onnx import helper

# NOTE: move to [onnx-modifier]

# 假设原始形状是 [batch_size, channels, height, width]
# Transpose后的形状是 [batch_size, height, width, channels]
# 新的形状张量

batch_size = 1
sequence_length = 16
data_dim = 1
embedding_dim = 1

new_shape = np.array([batch_size, sequence_length, data_dim], dtype=np.int64)
new_shape_tensor = helper.make_tensor_value_info('new_shape', onnx.TensorProto.INT64, [4])
model.graph.value_info.append(new_shape_tensor)

# 创建Reshape节点
reshape_node = helper.make_node(
    'Reshape',
    inputs=['input_tensor', 'new_shape'],
    outputs=['reshaped_tensor'],
    name='Reshape'
)