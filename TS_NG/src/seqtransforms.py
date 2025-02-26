import torch
import torch.nn.functional as F

# All sequence transforms assume that the input tensor is of shape (batch_size, seq_len, embedding_dim)

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(input):
        for transform in transforms:
            input = transform(input)
        return input
    return func

def transform_cat2_sep(tensor_pair, sep_id):
    return torch.cat([tensor_pair[0], torch.full((tensor_pair[0].shape[0], 1, 1), 
                                                 sep_id, 
                                                 dtype=tensor_pair[0].dtype, 
                                                 device=tensor_pair[0].device), 
                      tensor_pair[1], torch.full((tensor_pair[0].shape[0], 1, 1), 
                                                 sep_id, 
                                                 dtype=tensor_pair[0].dtype, 
                                                 device=tensor_pair[0].device)], 
                      dim=1)

# tensor: (batch_size, seq_len, embedding_dim)

# function to add BOS/EOS and create tensor for input sequence indices
def transform_add_bos_eos_to_tensor(tensor: torch.Tensor, bos_id, eos_id):
    batch_size, _seq_len, _ = tensor.shape
    bos_tensor = torch.full((batch_size, 1, 1), bos_id, dtype=tensor.dtype, device=tensor.device)
    eos_tensor = torch.full((batch_size, 1, 1), eos_id, dtype=tensor.dtype, device=tensor.device)
    tensor = torch.cat([bos_tensor, tensor, eos_tensor], dim=1)
    return tensor

def transform_add_cls_to_tensor(tensor: torch.Tensor, cls_id):
    batch_size, _seq_len, _ = tensor.shape
    cls_tensor = torch.full((batch_size, 1, 1), cls_id, dtype=tensor.dtype, device=tensor.device)
    tensor = torch.cat([cls_tensor, tensor], dim=1)
    return tensor

# function to collate data samples into batch tensors
def transform_add_pad_to_tensor(tensor: torch.Tensor, max_length, pad_id):
    _batch_size, seq_len, _ = tensor.shape
    assert seq_len <= max_length, f"Sequence length exceeds maximum length: seq_len: {seq_len}, max_length: {max_length}"
    pad2d = (0, 0, 0, max_length - seq_len)
    tensor = F.pad(tensor, pad2d, 'constant', pad_id)
    return tensor

def util_get_token_mask(processed_data, tok_id):
    return (processed_data == tok_id)

# pad2d = (padding_left:0, padding_right:0, padding_top:2 <CLS><SEP>, padding_bottom:1<SEP>)
def util_expand_token_mask(tensor : torch.Tensor, shape_car, shape_cdr, pad2d=(0,0,2,1)):
    car = torch.zeros(shape_car, device=tensor.device)
    cdr = torch.ones(shape_cdr, device=tensor.device)
    cons = torch.cat([car, cdr], dim=1)
    cons = torch.nn.ReplicationPad2d(pad2d)(cons).long()
    cons = transform_add_pad_to_tensor(cons, tensor.shape[1], 1)
    assert tensor.shape == cons.shape, f"Shapes don't match: tensor shape: {tensor.shape}, cons shape: {cons.shape}"
    return cons
