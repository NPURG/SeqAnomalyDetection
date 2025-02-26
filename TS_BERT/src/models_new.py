import torch, gc
import torch.nn as nn
from torch.nn import GELU
import torch.nn.functional as F
import torch.optim as optim
from functools import partial

#from transformer.Embed import PositionalEncoding
#from torch.nn import TransformerEncoderLayer
#from torch.nn import TransformerEncoder

from tsdatasets import Addr1394Dataset, WindowGenerator
from constants import *
from settings import *
from seqtransforms import *

from tqdm import tqdm

import os
import copy
import time
from torch.utils.tensorboard import SummaryWriter

# DONE: Get this FUCK model to run and give it a fucking tensorboard !!!
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BERT_Embedding(nn.Module):
    
    def __init__(self, vocab_size, max_len, n_segments, d_model, pad_id):
        super(BERT_Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id, device=DEVICE)  # token embedding
        self.pos_embed = nn.Embedding(max_len, d_model, device=DEVICE)  # position embedding 
        self.seg_embed = nn.Embedding(n_segments, d_model, device=DEVICE)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(x).to(DEVICE)  # [seq_len] -> [batch_size, seq_len]
        # print(self.tok_embed(x).shape , self.pos_embed(pos).shape , self.seg_embed(seg).shape)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(K.size(-1))) # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, n_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_q * n_heads, device=DEVICE)
        self.W_K = nn.Linear(d_model, d_k * n_heads, device=DEVICE)
        self.W_V = nn.Linear(d_model, d_v * n_heads, device=DEVICE)
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_model = d_model
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_q).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, seq_len, n_heads, d_v]
        output = nn.Linear(self.n_heads * self.d_v, self.d_model, device=DEVICE)(context)
        return nn.LayerNorm(self.d_model, device=DEVICE)(output + residual) # output: [batch_size, seq_len, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff, d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff, device=DEVICE)
        self.fc2 = nn.Linear(d_ff, d_model, device=DEVICE)
        self.gelu = GELU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_ff, d_q, d_k, d_v, n_heads, d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_q, d_k, d_v, n_heads, d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_ff, d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs

class SeqBERT(nn.Module):
    def __init__(self, d_model, vocab_size, max_len, pad_id, # embedding
                 d_ff, d_q, d_k, d_v, n_heads, # encoder
                 n_layers, n_segments, dp_out):
        super(SeqBERT, self).__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.embedding = BERT_Embedding(vocab_size, max_len, n_segments, d_model, pad_id)
        self.layers = nn.ModuleList([EncoderLayer(d_ff, d_q, d_k, d_v, n_heads, d_model) for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model, device=DEVICE),
            nn.Dropout(dp_out), 
            nn.Tanh(),			
        )
        self.classifier = nn.Linear(d_model, 2, device=DEVICE)
        self.linear = nn.Linear(d_model, d_model, device=DEVICE)
        self.activ2 = GELU()
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False, device=DEVICE)
        self.fc2.weight = embed_weight
    
    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, seq_len = seq_q.size()
        # eq(pad_id) is PAD token
        pad_attn_mask = seq_q.data.eq(self.pad_id).unsqueeze(1)  # [batch_size, 1, seq_len]
        return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

    def forward(self, input_ids, segment_ids):
        output = self.embedding(input_ids, segment_ids) # [bach_size, seq_len, d_model]
        enc_self_attn_mask = self.get_attn_pad_mask(input_ids, input_ids) # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask) # side effects
        # it will be decided by first token(CLS)
        h_pooled = self.fc(output[:, 0]) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2] predict isNext

        # masked_pos = masked_pos[:, :, None].expand(-1, -1, self.d_model) # [batch_size, max_pred, d_model]
        # h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(output)) # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked) # [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf

def get_transform_pipeline(sep_id, cls_id, pad_id, max_length):
    transform1 = partial(transform_cat2_sep, sep_id=sep_id)
    # transform2 = partial(transform_add_bos_eos_to_tensor, bos_id=bos_id, eos_id=eos_id)
    transform2 = partial(transform_add_cls_to_tensor, cls_id=cls_id)
    transform3 = partial(transform_add_pad_to_tensor, max_length=max_length, pad_id=pad_id)
    return sequential_transforms(transform1, # sep
                                 transform2, # cls # bos eos
                                 transform3) # pad

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_name': model.__class__.__name__,
        'model_settings': copy.deepcopy(model.__dict__),
        'model_obj': model,
		'model_state_dict': model.state_dict(),}, path)

def pretrain_random_mask(model, window, id_dict, time_stamp, seg_car_length, seg_cdr_length, vocab_size, summary_step=10):    
    # optimazer
    optimizer = optim.Adadelta(model.parameters(), lr=0.02) #0.001 TODO: Adam, warmup
    criterion = nn.CrossEntropyLoss(ignore_index=id_dict['pad_id'])
    
    steplr = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.65)

    # representation model
    # input: [sequence] with size: (batch_size, seq_length)
    # output: [sequence] with size: (batch_size, seq_length)
    
    
    # Task cloze:
    # ex input: [CLS] 1 2 3 4 [mask] 6 7 8 9 10 [SEP] 1 2 3 4 5 6 7 8 [mask] 10 [SEP]
    # ex output: 5 9
    
    # Task next sentence prediction:
    # ex input: [CLS] 1 2 3 4 [SEP] 5 6 7 8 [SEP]
    # ex output: 1
    
    # Task full sequence prediction:
    # ex input: 1([CLS]) 1 2 3 4 [SEP] [mask] [mask] [mask] [mask] [SEP]
    # ex output: 5 6 7 8
    
    # Load the trainning data
    
    window.change_column(['MASKEDTOK', 'MASKED'],['MASKEDTOK', 'MASKED'])
    dl_train = window.train_dataset
    
    # turn the trainning mode on
    model.train(mode=True)
    with SummaryWriter(log_dir=os.path.join(LOGPATH, model.__class__.__name__, time_stamp), comment='SeqBERT') as writer:
        # dl_test = window.test_dataset
        for epoch in range(EPOCHS):
            it = 0
            for x, y in iter(dl_train):
                # Move tensors to the device
                masked_token_x = x[:,:,0].to(DEVICE).unsqueeze(-1) # masked token
                masked_token_y = y[:,:,0].to(DEVICE).unsqueeze(-1)
                
                x = x[:,:,1].to(DEVICE).unsqueeze(-1) # DST sequence data
                y = y[:,:,1].to(DEVICE).unsqueeze(-1)
                
                # Apply preprocessing transforms
                input_ids = ts_preprocess((x, y))
                masked_tokens = ts_masked_token_preprocess((masked_token_x, masked_token_y))
    
                # Get masks
                mask_padding = get_mask_padding(input_ids).squeeze(-1).long()
                mask_masktoken = get_mask_masktoken(input_ids).squeeze(-1).long()
                segment_ids = get_mask_segment(input_ids,
                                               shape_car=(x.size(0), seg_car_length, 1), 
                                               shape_cdr=(x.size(0), seg_cdr_length, 1)).squeeze(-1).long().to(DEVICE)
                
                # Forward
                logits_lm, logits_clsf = model(input_ids.squeeze(-1), segment_ids)
                loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
                loss_lm = (loss_lm.float()).mean()
                #loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
                loss = loss_lm #+ loss_clsf
        
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if it % summary_step == 0:
                    writer.add_scalar(f'Loss/train-epoch-{epoch}', loss, it)
                # Print progress
                if (it + 1) % 10 == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                    if loss < 1:
                        save_model(model, f"checkpoints/SeqBERT/pretrain_random_mask/model-{loss}-{epoch}-{it}-{time.time()}.ckptn")
                it += 1
            steplr.step()
            save_model(model, f"checkpoints/SeqBERT/pretrain_random_mask/model-{loss}-{epoch}-{time.time()}.ckptn")

def pretrain_mask_predict(model, window, id_dict, time_stamp, seg_car_length, seg_cdr_length, vocab_size, summary_step=10):    
    # optimazer
    optimizer = optim.Adam(model.parameters(), lr=7e-5) #0.001
    criterion = nn.CrossEntropyLoss(ignore_index=id_dict['pad_id'])
    
    steplr = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.65)
    
    window.change_column(['DST'],['DST'])
    dl_train = window.train_dataset
    
    # turn the trainning mode on
    model.train(mode=True)
    with SummaryWriter(log_dir=os.path.join(LOGPATH, model.__class__.__name__+"_mask_predict", time_stamp), comment='SeqBERT pretrain_mask_predict') as writer:
        # dl_test = window.test_dataset
        for epoch in range(EPOCHS):
            it = 0
            data_iter = tqdm(iter(dl_train), mininterval=10)
            for x, y in iter(data_iter):
                # Move tensors to the device
                masked_token_x = torch.zeros_like(x).fill_(id_dict['pad_id']).to(DEVICE) # masked token
                masked_token_y = y.to(DEVICE)

                x = x.to(DEVICE) # DST sequence data
                y = torch.zeros_like(y).fill_(id_dict['mask_id']).to(DEVICE) # mask all the ys' to predict

                # Apply preprocessing transforms
                input_ids = ts_preprocess((x, y))
                masked_tokens = ts_masked_token_preprocess((masked_token_x, masked_token_y))

                # Get masks
                mask_padding = get_mask_padding(input_ids).squeeze(-1).long()
                mask_masktoken = get_mask_masktoken(input_ids).squeeze(-1).long()
                segment_ids = get_mask_segment(input_ids,
                                               shape_car=(x.size(0), seg_car_length, 1), 
                                               shape_cdr=(x.size(0), seg_cdr_length, 1)).squeeze(-1).long().to(DEVICE)

                # Forward
                logits_lm, logits_clsf = model(input_ids.squeeze(-1), segment_ids)
                loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
                loss_lm = (loss_lm.float()).mean()
                #loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
                loss = loss_lm #+ loss_clsf

                #################
                # print("logits shape:", logits_lm.shape, "mt_shape", masked_tokens.shape)
                # o = logits_lm.detach().argmax(dim=-1).to('cpu')[:,48]
                # t = masked_tokens.to('cpu')[:,48]
                # print("TTTTTTT", o, t)
                #################

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if it % summary_step == 0:    
                    writer.add_scalar(f'Loss/train-epoch-{epoch}', loss, it)

                # Print progress and Save model
                if loss < 1.96:
                    save_model(model, f"checkpoints/SeqBERT/pretrain-predict/model-{loss}-{epoch}-{it}-{time.time()}.ckptn")
                data_iter.set_description(f'Epoch [{epoch}/{EPOCHS}]')
                data_iter.set_postfix(loss = '{:.6f}'.format(loss))

                it += 1
            steplr.step()
            save_model(model, f"checkpoints/SeqBERT/pretrain-predict/model-{loss}-{epoch}-{time.time()}.ckptn")

if __name__ == '__main__':
	## Tranning process is just like bert, but for sequence data
	## 1. Mask the data (15% of the data is masked, 80% of the time, it is replaced with a mask token, 10% of the time, it is replaced with a random token, 10% of the time, it is left as is):
	## 2. Test time serise as next sentence prediction
    ## 3. Embedding the data
	## 4. Mask the data
	## 5. Train the model
	## 6. Evaluate the model
	## 7. Save the model
	## 8. Load the model
	## 9. Predict the model
	## 10. Visualize the model
    time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))

    # Set the seed
    torch.manual_seed(RANDOM_SEED)
    # Set the device
    
    # Get basic constants
    bos_id = SPECIAL_TOKEN_BASE + special_tokens['<BOS>']
    eos_id = SPECIAL_TOKEN_BASE + special_tokens['<EOS>']
    mask_id = SPECIAL_TOKEN_BASE + special_tokens['<MASK>']
    pad_id = SPECIAL_TOKEN_BASE + special_tokens['<PAD>']
    cls_id = SPECIAL_TOKEN_BASE + special_tokens['<CLS>']
    sep_id = SPECIAL_TOKEN_BASE + special_tokens['<SEP>']
    
    id_dict = {
        'bos_id': bos_id,
        'eos_id': eos_id,
        'mask_id': mask_id,
        'pad_id': pad_id,
        'cls_id': cls_id,
        'sep_id': sep_id
    }

    batch_size = datasetcfg['window_generator']['make_dataloader']['batch_size'] # batch_size = BATCH_SIZE
    max_length = modelcfg['BERT']['max_len']
    vocab_size = modelcfg['BERT']['vocab_size']

    # Config the dataset split window
    # data = Addr1394Dataset()
    window = WindowGenerator(
        input_width=datasetcfg['window_generator']['initargs']['input_width'], 
        label_width=datasetcfg['window_generator']['initargs']['label_width'], 
        shift=datasetcfg['window_generator']['initargs']['shift'], 
        dtype=datasetcfg['window_generator']['initargs']['dtype'], 
        train_columns=datasetcfg['window_generator']['initargs']['train_columns'],
        label_columns=datasetcfg['window_generator']['initargs']['label_columns'],
        dataset_initarg=datasetcfg['addr1394']['initargs'],
        batch_size=batch_size, 
        shuffle=datasetcfg['window_generator']['make_dataloader']['shuffle'],
        num_workers=datasetcfg['window_generator']['make_dataloader']['num_workers']
	)
    
	# Config the process of data like NLP
    car_length = datasetcfg['window_generator']['initargs']['input_width'] # input_width + label_width = window_size
    cdr_length = datasetcfg['window_generator']['initargs']['label_width'] # input_width + label_width = window_size
    
    ts_preprocess = get_transform_pipeline(sep_id=sep_id, cls_id=cls_id, pad_id=pad_id, max_length=max_length)
    ts_masked_token_preprocess = get_transform_pipeline(sep_id=pad_id, cls_id=pad_id, pad_id=pad_id, max_length=max_length)
    
    get_mask_padding = partial(util_get_token_mask, tok_id=pad_id)
    get_mask_masktoken = partial(util_get_token_mask, tok_id=mask_id)
    get_mask_segment = partial(util_expand_token_mask)
    
    # model
    model = SeqBERT(
        max_len = max_length,
        n_layers = modelcfg['BERT']['n_layers'],
        n_heads = modelcfg['BERT']['n_heads'],
        d_model = modelcfg['BERT']['d_model'],
        d_ff = modelcfg['BERT']['d_ff'], # 4*d_model, FeedForward dimension
        d_q = modelcfg['BERT']['d_k'], 
        d_k = modelcfg['BERT']['d_k'],
        d_v = modelcfg['BERT']['d_v'],  # dimension of K(=Q), V
        n_segments = modelcfg['BERT']['n_segments'],
        vocab_size = vocab_size,
        dp_out = modelcfg['BERT']['dp_out'],
        pad_id = pad_id
        ).to(device=DEVICE)
    # with SummaryWriter(log_dir=os.path.join(LOGPATH, model.__class__.__name__, time_stamp), comment='SeqBERT') as writer:
    #     dummy_input = torch.zeros((batch_size, max_length), dtype=torch.long).to(DEVICE)
    #     with torch.no_grad():
    #         model.eval()
    #         writer.add_graph(model, (dummy_input, dummy_input))
    
    # Train the model
    # pretrain_random_mask(model, window, id_dict, time_stamp, car_length, cdr_length, vocab_size)

    model.load_state_dict(torch.load("checkpoints/SeqBERT/model-1.88-colab.ckptn", map_location=DEVICE)['model_state_dict'])
    pretrain_mask_predict(model, window, id_dict, time_stamp, car_length, cdr_length, vocab_size)



    