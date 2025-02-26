import os
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from settings import *
from docs import *
# Some standard imports
import numpy as np

from torch import nn

import torch.onnx
# PEND: 更换版本/替换底层算子 (无效)

class TestModel(nn.Module):
    def __init__(self, dims):
        super(TestModel, self).__init__()
        self.dims = dims
        self.lstm = nn.LSTM(dims, 5, 1, batch_first=True)
        self.fc = nn.Linear(1, 5)
        self.name = "TestModel"

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.reshape(-1, 1)
        x = self.fc(x)
        return x

def load_model_simple(modelname, shape, device=torch.device("cpu")):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double().to(device)

	fname = f'checkpoints/{args.model[0]}_{args.dataset}/model.ckpt'#TODO: path names for all models
	if os.path.exists(fname):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		print(fname)
		checkpoint = torch.load(fname, map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])
		sample_x = torch.randn(shape, dtype=torch.float32, requires_grad=True).to(device)
		return model, sample_x
	else:
		print(f"{color.GREEN}Model {model.name}{color.ENDC} does not exist")
	return None


if __name__ == '__main__':
    batch_size = 1
    dims = 5
    seq_length = 16
    opset = 11
    device = 'cpu'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, x = load_model_simple(args.model[0], shape=(batch_size, seq_length, dims), device=device)
    # model, x = model.to(device), x.to(device)
    # print(device)

    # # set the model to inference mode
    model.train(False)
    model.eval()
    # print(model)
    # print(model(x))
    shape = (batch_size, seq_length, dims)

    # model = TestModel(dims).double()
    # x = torch.randn(shape, dtype=float, requires_grad=True)
    
	
    ## convert to onnx
	# Export the model
    torch.onnx.export(model,                 # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  f"{model.name}.onnx",      # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                   opset_version=opset,
                #   custom_opsets={'Relu':6},# the ONNX version to export the model to
                  do_constant_folding=True,
                #   export_modules_as_functions=True,# whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) #, # the model's output names
                #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                #                 'output' : {0 : 'batch_size'}})
    #check model
    import onnx

    onnx_model = onnx.load(f"{model.name}.onnx")
    onnx.checker.check_model(onnx_model)

