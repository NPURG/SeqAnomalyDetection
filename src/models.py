from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import pickle
# import dgl
# from dgl.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *
from settings import *
torch.manual_seed(1)

## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
	def __init__(self, feats):
		super(LSTM_Univariate, self).__init__()
		self.name = 'LSTM_Univariate'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 1
		self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for i in range(feats)])

	def forward(self, x):
		hidden = [(torch.rand(1, 1, self.n_hidden, dtype=torch.float64), 
			torch.randn(1, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
		outputs = []
		for i, g in enumerate(x):
			multivariate_output = []
			for j in range(self.n_feats):
				univariate_input = g.view(-1)[j].view(1, 1, -1)
				out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
				multivariate_output.append(2 * out.view(-1))
			output = torch.cat(multivariate_output)
			outputs.append(output)
		return torch.stack(outputs)

## Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
	def __init__(self, feats):
		super(Attention, self).__init__()
		self.name = 'Attention'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = 5 # MHA w_size = 5
		self.n = self.n_feats * self.n_window
		self.atts = [ nn.Sequential( nn.Linear(self.n, feats * feats), 
				nn.ReLU(True))	for i in range(1)]
		self.atts = nn.ModuleList(self.atts)

	def forward(self, g):
		for at in self.atts:
			ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
			g = torch.matmul(g, ats)		
		return g, ats

## LSTM_AD Model
class LSTM_AD(nn.Module):
	def __init__(self, feats):
		super(LSTM_AD, self).__init__()
		self.name = 'LSTM_AD'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 64
		self.lstm = nn.LSTM(feats, self.n_hidden)
		self.lstm2 = nn.LSTM(feats, self.n_feats)
		self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

	def forward(self, x):
		hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		hidden2 = (torch.rand(1, 1, self.n_feats, dtype=torch.float64), torch.randn(1, 1, self.n_feats, dtype=torch.float64))
		outputs = []
		for i, g in enumerate(x):
			out, hidden = self.lstm(g.view(1, 1, -1), hidden)
			out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
			out = self.fcn(out.view(-1))
			outputs.append(2 * out.view(-1))
		return torch.stack(outputs)

## DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
	def __init__(self, feats):
		super(DAGMM, self).__init__()
		self.name = 'DAGMM'
		self.lr = 0.0001
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 8
		self.n_window = 5 # DAGMM w_size = 5
		self.n = self.n_feats * self.n_window
		self.n_gmm = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.estimate = nn.Sequential(
			nn.Linear(self.n_latent+2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
			nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
		)

	def compute_reconstruction(self, x, x_hat):
		relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
		cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
		return relative_euclidean_distance, cosine_similarity

	def forward(self, x):
		## Encode Decoder
		x = x.view(1, -1)
		z_c = self.encoder(x)
		x_hat = self.decoder(z_c)
		## Compute Reconstructoin
		rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
		z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
		## Estimate
		gamma = self.estimate(z)
		return z_c, x_hat.view(-1), z, gamma.view(-1)

## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
	def __init__(self, feats):
		super(OmniAnomaly, self).__init__()
		self.name = 'OmniAnomaly'
		self.lr = 0.002
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 32
		self.n_latent = 8
		self.lstm = nn.GRU(feats, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def forward(self, x, hidden = None):
		hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x.view(1, 1, -1), hidden)
		## Encode
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		## Decoder
		x = self.decoder(x)
		return x.view(-1), mu.view(-1), logvar.view(-1), hidden

## USAD Model (KDD 20)
class USAD(nn.Module):
	def __init__(self, feats):
		super(USAD, self).__init__()
		self.name = 'USAD'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 32 #16
		self.n_latent = 8 # 5
		self.n_window = 16 # USAD w_size =16, 5, 34
		self.n = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			# nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = self.encoder(g)#.view(1,-1))
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1, self.n), ae2.view(-1, self.n), ae2ae1.view(-1, self.n)


class USAD_GELU(nn.Module):
	def __init__(self, feats):
		super(USAD_GELU, self).__init__()
		self.name = 'USAD_GELU'
		self.lr = 0.001
		self.n_feats = feats
		self.n_hidden = 32 #16
		self.n_latent = 8 # 5
		self.n_window = 16 # USAD w_size =16, 5, 34
		self.n = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			# nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.GELU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.GELU(),
			nn.Linear(self.n_hidden, self.n_latent), nn.GELU(),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.GELU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.GELU(),
			nn.Linear(self.n_hidden, self.n), nn.GELU(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.GELU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.GELU(),
			nn.Linear(self.n_hidden, self.n), nn.GELU(),
		)

	def forward(self, g):
		## Encode
		z = self.encoder(g)#.view(1,-1))
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1, self.n), ae2.view(-1, self.n), ae2ae1.view(-1, self.n)
'''
1. 创建一个新的USAD模型，该模型继承自nn.Module。
2. 在模型的初始化方法中，定义所有需要的层和参数，包括编码器、解码器和小波变换。
3. 在模型的前向传播方法中，首先对输入序列应用小波变换，然后将变换后的序列输入到编码器中。
4. 在编码器的输出上应用逆小波变换，然后将结果输入到解码器中。
5. 返回解码器的输出。

小波变换和逆小波变换是在序列长度维度上应用的。
'''

import pywt

class WaveletUSAD(nn.Module):
    def __init__(self, feats):
        super(WaveletUSAD, self).__init__()
        self.name = 'WaveletUSAD'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 5
        self.n_window = 34
        self.n = self.n_feats * self.n_window
        self.wavelet = pywt.Wavelet('db1')
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )

    def forward(self, g):
        # Apply wavelet transform
        g_transformed = pywt.wavedec(g, self.wavelet)
        g_transformed = torch.cat(g_transformed, dim=-1)
        # Encode
        z = self.encoder(g_transformed)
        # Apply inverse wavelet transform
        z_inv_transformed = pywt.waverec(z, self.wavelet)
        # Decoders (Phase 1)
        ae1 = self.decoder1(z_inv_transformed)
        ae2 = self.decoder2(z_inv_transformed)
        # Encode-Decode (Phase 2)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)

## USAD with Embeddings Model (TODO:)
class USAD_EMB(nn.Module):
	def __init__(self, feats):
		super(USAD, self).__init__()
		self.name = 'USAD_EMB'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 5
		self.n_window = 34 # USAD w_size =16, 5
		self.n = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = self.encoder(g.view(1,-1))
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)
	
# ## USAD_LSTM (New Model()
# class USAD_LSTM(nn.Module):
# 	def __init__(self, feats):
# 		super(USAD_LSTM, self).__init__()
# 		self.name = 'USAD_LSTM'
# 		self.lr = 0.0001
# 		self.n_feats = feats
# 		self.n_hidden = 32
# 		self.n_latent = 16
# 		self.n_window = 34 # USAD w_size =16, 5
# 		self.n = self.n_feats * self.n_window
# 		self.encoder = nn.LSTM(self.n, self.n_hidden, batch_first=True)
# 		self.decoder1 = nn.LSTM(self.n_hidden, self.n, batch_first=True)
# 		self.decoder2 = nn.LSTM(self.n_hidden, self.n, batch_first=True)

# 	def forward(self, g):
# 		## Encode
# 		_, (h_n, _) = self.encoder(g.view(1, self.n_window, self.n_feats))
# 		z = h_n.view(1, -1)
# 		## Decoders (Phase 1)
# 		_, (h_n_dec1, _) = self.decoder1(z.view(1, 1, -1))
# 		ae1 = h_n_dec1.view(-1, self.n)
# 		_, (h_n_dec2, _) = self.decoder2(z.view(1, 1, -1))
# 		ae2 = h_n_dec2.view(-1, self.n)
# 		## Encode-Decode (Phase 2)
# 		_, (h_n_enc, _) = self.encoder(ae1.view(1, self.n, self.n_hidden))
# 		ae2ae1, _ = self.decoder2(h_n_enc.view(1, 1, -1))
# 		ae2ae1 = ae2ae1.view(-1, self.n)
# 		return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)

##make this model more sophiscated and reasonable as a good paper's realization of unsupervised sequential anomaly detection modelAdd attention mechanism: Incorporate an attention mechanism to allow the model to focus on relevant parts of the input sequence while encoding and decoding.
'''
Use bidirectional LSTM: Replace the unidirectional LSTM with a bidirectional LSTM to capture both past and future context information.

Add residual connections: Introduce residual connections between the encoder and decoder layers to facilitate the flow of information and improve gradient propagation.

Include variational autoencoder (VAE) components: Extend the model to include VAE components, such as a latent space regularization term and a reconstruction loss based on the VAE framework.

Implement a self-attention mechanism: Introduce a self-attention mechanism to capture long-range dependencies and improve the model's ability to detect anomalies.

Incorporate a temporal attention mechanism: Implement a temporal attention mechanism to weigh the importance of different time steps in the input sequence.

Utilize a more advanced architecture: Consider using more advanced architectures, such as Transformer-based models, which have shown promising results in sequential anomaly detection tasks.

Remember to carefully review and adapt these suggestions based on the specific requirements and constraints of your project.
'''
## USAD_LSTM New Model
class USAD_LSTM(nn.Module):
	def __init__(self, feats):
		super(USAD_LSTM, self).__init__()
		self.name = 'USAD_LSTM'
		self.lr = 0.001 #0.0001
		self.n_latent = 16 #8
		self.n_hidden = 8 #8
		self.n_layers = 2 #1
		self.n_window = 16 # USAD w_size =16
		self.dpout = 0.2 #0
		self.n_feats = feats

		if DEBUG:
			print(__file__ + ":n_feats: ", self.n_feats)
			print(__file__ + ":n_hidden: ", self.n_hidden)
			print(__file__ + ":n_latent: ", self.n_latent)
			print(__file__ + ":n_window: ", self.n_window)

		self.encoder = nn.LSTM(self.n_feats, self.n_latent, num_layers=self.n_layers, dropout=self.dpout, batch_first=True)  # Use bidirectional LSTM #bi-directional -> self.n_hidden * 2
		self.decoder1 = nn.LSTM(self.n_latent, self.n_hidden, num_layers=self.n_layers, dropout=self.dpout, proj_size=self.n_feats, batch_first=True)  # Update input size for decoder1
		self.decoder2 = nn.LSTM(self.n_latent, self.n_hidden, num_layers=self.n_layers, dropout=self.dpout, proj_size=self.n_feats, batch_first=True)  # Update input size for decoder2

		# self.apply(init_weights)
	def forward(self, g):
		## Encode
		out_n, (h_n, c_n) = self.encoder(g.view(-1, self.n_window, self.n_feats))
		if DEBUG:
			print(__file__ + ":g-shape:", g.shape)
			# print(__file__ + ":z-shape:", z.shape)
			print(__file__ + ":h_n-shape:", h_n.shape)
			print(__file__ + ":c_n-shape:", np.array([c.detach().numpy() for c in c_n]).shape)

		## Decoders (Phase 1)
		ae1, (h_n_dec1, c_n_dec1) = self.decoder1(out_n) # to re-build the input, rebuild based on encoder's out and lantents(bias) (with 0 init)

		## Decoders (Phase 2)
		# if DEBUG:
			# plt.subplot(2,1,1)
			# print("out_n_shape",out_n.shape)
			# plt.plot(out_n[:,0,:].cpu().detach().numpy())
			# plt.subplot(2,1,2)
			# plt.plot(ae1[:,0,:].cpu().detach().numpy())
			# plt.show()
			# print(__file__ + ":Dec2 hidden cons shape:", 
		 			# torch.concat((h_n[0: self.n_layers],h_n[self.n_layers:]),axis=-1).shape)
			
		ae2, (h_n_dec2, c_n_dec2) = self.decoder2(out_n)#, (h_n_dec1, c_n_dec1))	

		## Encode-Decode (Phase 2)
		if DEBUG:
			print(__file__ + ":Encoder input shape:", ae1.shape)
		o_enc_ae1, (h_n_enc, c_n_enc) = self.encoder(ae1)#, (h_n, c_n))#(h_n_dec2, c_n_dec2))

		ae2ae1, (h_xx, c_yy) = self.decoder2(o_enc_ae1)#, (h_n_dec2, c_n_dec2))# to distinguish the rebuild and the original
		if DEBUG:
			print(__file__ + ":ae1 shape:", ae1.shape)
			print(__file__ + ":ae2 shape:", ae2.shape)
			print(__file__ + ":ae2ae1 shape:", ae2ae1.shape)
		return ae1.squeeze(0), ae2.squeeze(0), ae2ae1.squeeze(0)
class USAD_LSTM_NP(nn.Module):
	def __init__(self, feats):
		super(USAD_LSTM_NP, self).__init__()
		self.name = 'USAD_LSTM_NP'
		self.lr = 0.0001 #0.0001 0.001
		self.n_latent = 16 #8
		self.n_hidden = 8 #8
		self.n_layers = 2 #1
		self.n_window = 16 # USAD w_size =16
		self.dpout = 0.2 #0
		self.n_feats = feats

		if DEBUG:
			print(__file__ + ":n_feats: ", self.n_feats)
			print(__file__ + ":n_hidden: ", self.n_hidden)
			print(__file__ + ":n_latent: ", self.n_latent)
			print(__file__ + ":n_window: ", self.n_window)

		self.encoder = nn.LSTM(self.n_feats, self.n_latent, num_layers=self.n_layers, dropout=self.dpout, batch_first=True)  # Use bidirectional LSTM #bi-directional -> self.n_hidden * 2
		self.decoder1 = nn.LSTM(self.n_latent, self.n_hidden, num_layers=self.n_layers, dropout=self.dpout, batch_first=True)  # Update input size for decoder1
		self.decoder2 = nn.LSTM(self.n_latent, self.n_hidden, num_layers=self.n_layers, dropout=self.dpout, batch_first=True)  # Update input size for decoder2
		self.feedforward = nn.Sequential(
			nn.Linear(self.n_hidden,self.n_hidden), nn.Sigmoid(),
			nn.Linear(self.n_hidden, feats)
		)
		# self.apply(init_weights)
	def forward(self, g):
		## Encode
		out_n, (h_n, c_n) = self.encoder(g.view(-1, self.n_window, self.n_feats))
		if DEBUG:
			print(__file__ + ":g-shape:", g.shape)
			# print(__file__ + ":z-shape:", z.shape)
			print(__file__ + ":h_n-shape:", h_n.shape)
			print(__file__ + ":c_n-shape:", np.array([c.detach().numpy() for c in c_n]).shape)

		## Decoders (Phase 1)
		ae1, (h_n_dec1, c_n_dec1) = self.decoder1(out_n) # to re-build the input, rebuild based on encoder's out and lantents(bias) (with 0 init)
		ae1 = self.feedforward(ae1)
		## Decoders (Phase 2)
		# if DEBUG:
			# plt.subplot(2,1,1)
			# print("out_n_shape",out_n.shape)
			# plt.plot(out_n[:,0,:].cpu().detach().numpy())
			# plt.subplot(2,1,2)
			# plt.plot(ae1[:,0,:].cpu().detach().numpy())
			# plt.show()
			# print(__file__ + ":Dec2 hidden cons shape:", 
		 			# torch.concat((h_n[0: self.n_layers],h_n[self.n_layers:]),axis=-1).shape)
			
		ae2, (h_n_dec2, c_n_dec2) = self.decoder2(out_n)#, (h_n_dec1, c_n_dec1))	
		ae2 = self.feedforward(ae2)
		## Encode-Decode (Phase 2)
		if DEBUG:
			print(__file__ + ":Encoder input shape:", ae1.shape)
		o_enc_ae1, (h_n_enc, c_n_enc) = self.encoder(ae1)#, (h_n, c_n))#(h_n_dec2, c_n_dec2))

		ae2ae1, (h_xx, c_yy) = self.decoder2(o_enc_ae1)#, (h_n_dec2, c_n_dec2))# to distinguish the rebuild and the original
		ae2ae1 = self.feedforward(ae2ae1)
		if DEBUG:
			print(__file__ + ":ae1 shape:", ae1.shape)
			print(__file__ + ":ae2 shape:", ae2.shape)
			print(__file__ + ":ae2ae1 shape:", ae2ae1.shape)
		return ae1.squeeze(0), ae2.squeeze(0), ae2ae1.squeeze(0)
## USAD_BiLSTM New Model
class USAD_BiLSTM(nn.Module):
	def __init__(self, feats):
		super(USAD_BiLSTM, self).__init__()
		self.name = 'USAD_BiLSTM'
		self.lr = 0.0001 #0.0001
		self.n_feats = feats # 2
		self.n_hidden = 32
		self.n_latent = 16
		self.n_layers = 8
		self.n_window = 12 # USAD w_size =16, 5
		self.n = self.n_feats * self.n_window #=68
		if DEBUG:
			print(__file__ + ":n_feats: ", self.n_feats)
			print(__file__ + ":n_hidden: ", self.n_hidden)
			print(__file__ + ":n_latent: ", self.n_latent)
			print(__file__ + ":n_window: ", self.n_window)
			print(__file__ + ":n: ", self.n)
		self.encoder = nn.LSTM(self.n_feats, self.n_latent, num_layers=self.n_layers, batch_first=True, bidirectional=True)  # Use bidirectional LSTM #bi-directional -> self.n_hidden * 2
		self.decoder1 = nn.LSTM(self.n_latent * 2, self.n_hidden, num_layers=self.n_layers, proj_size=2, batch_first=True)  # Update input size for decoder1
		self.decoder2 = nn.LSTM(self.n_latent * 2, self.n_hidden, num_layers=self.n_layers, proj_size=2, batch_first=True)  # Update input size for decoder2
		self.fc_mu = nn.Sequential(#nn.LayerNorm(self.n_latent * 2),  # VAE component: batch normalization layer
								   nn.Linear(self.n_latent * 2, self.n_latent * 2),)  # VAE component: linear layer for mean)
		self.fc_logvar = nn.Sequential(#nn.LayerNorm(self.n_latent * 2),
								   nn.Linear(self.n_latent * 2, self.n_latent * 2),  # VAE component: linear layer for log variance
								   nn.GELU())
		self.map_h_dec2 = nn.Sequential(#nn.LayerNorm(self.n_latent * 2),
								   nn.Linear(self.n_latent * 2, self.n_feats),  # VAE component: linear layer for mapping h_dec2
								   nn.GELU())
		self.map_c_dec2 = nn.Sequential(#nn.LayerNorm(self.n_latent * 2),
								   nn.Linear(self.n_latent * 2, self.n_hidden),  # VAE component: linear layer for mapping h_dec2
								   nn.GELU())
		self.map_h_ae2ae1 = nn.Sequential(#nn.LayerNorm(self.n_latent * 2),
								   nn.Linear(self.n_latent * 2, self.n_feats),  # VAE component: linear layer for mapping h_dec2
								   nn.GELU())
		self.map_c_ae2ae1 = nn.Sequential(#nn.LayerNorm(self.n_latent * 2),
								   nn.Linear(self.n_latent * 2, self.n_hidden),  # VAE component: linear layer for mapping h_dec2
								   nn.GELU())
		self.layer_norm = nn.LayerNorm(self.n_latent * 2)
		self.apply(init_weights)
	def forward(self, g):
		## Encode
		out_n, (h_n, c_n) = self.encoder(g.view(-1, self.n_window, self.n_feats))
		if DEBUG:
			print(__file__ + ":g-shape:", g.shape)
			# print(__file__ + ":z-shape:", z.shape)
			print(__file__ + ":h_n-shape:", h_n.shape)
			print(__file__ + ":c_n-shape:", np.array([c.detach().numpy() for c in c_n]).shape)

		## Decoders (Phase 1)
		ae1, (h_n_dec1, c_n_dec1) = self.decoder1(out_n) # to re-build the input, rebuild based on encoder's out and lantents(bias) (with 0 init)
		 #= o_dec1 #torch.concat((c_n_dec1[0],c_n_dec1[1]),axis=-1)#.view(-1, self.n_hidden * 2)  # Shape: (n_hidden*2,)
		## Decoders (Phase 2)
		if DEBUG:
			# plt.subplot(2,1,1)
			# print("out_n_shape",out_n.shape)
			# plt.plot(out_n[:,0,:].cpu().detach().numpy())
			# plt.subplot(2,1,2)
			# plt.plot(ae1[:,0,:].cpu().detach().numpy())
			# plt.show()
			print(__file__ + ":Dec2 hidden cons shape:", 
		 			torch.concat((h_n[0: self.n_layers],h_n[self.n_layers:]),axis=-1).shape)
		ae2, (h_n_dec2, c_n_dec2) = self.decoder2(out_n, (h_n_dec1, c_n_dec1))	
												#(self.map_h_dec2(torch.concat((h_n[0: self.n_layers],h_n[self.n_layers:]),axis=-1)), #.unsqueeze(0)
			   									# self.map_c_dec2(torch.concat((c_n[0: self.n_layers],c_n[self.n_layers:]),axis=-1))))#.unsqueeze(0)
		 #= o_dec2 #torch.concat((c_n_dec2[0],c_n_dec2[1]),axis=-1)#.view(-1, self.n_hidden * 2)  # Shape: (n_hidden*2,)
		## Encode-Decode (Phase 2)
		if DEBUG:
			print(__file__ + ":Encoder input shape:", ae1.shape)
		o_enc_ae1, (h_n_enc, c_n_enc) = self.encoder(ae1, (h_n, c_n))#(h_n_dec2, c_n_dec2))
										    # (torch.concat((h_n_dec2[0],h_n_dec2[1]),axis=-1).unsqueeze(0), 
			  								# torch.concat((c_n_dec2[0],c_n_dec2[1]),axis=-1).unsqueeze(0)))#ae1.view(1, self.n_hidden * 2, self.n_hidden * 2), (h_n, c_n))  # Update input size for encoder
		# mu_ae1 = self.fc_mu(o_enc_ae1)  # Shape: (1, n_latent)
		# logvar_ae1 = self.fc_logvar(o_enc_ae1)  # Shape: (1, n_latent)
		# z_latent_ae1 = self.reparameterize(mu_ae1, logvar_ae1)  # Shape: (1, n_latent)

		#c_n_enc_concat =  torch.concat((c_n_enc[0], c_n_enc[1]),axis=-1)
		ae2ae1, (h_xx, c_yy) = self.decoder2(o_enc_ae1, (h_n_dec2, c_n_dec2))# to distinguish the rebuild and the original
							#(self.map_h_ae2ae1(torch.concat((h_n_enc[0: self.n_layers],h_n_enc[self.n_layers:]),axis=-1)), #.unsqueeze(0)
							# self.map_c_ae2ae1(torch.concat((c_n_enc[0: self.n_layers],c_n_enc[self.n_layers:]),axis=-1))))#.unsqueeze(0) #h_n_enc, (h_n_dec2, c_n_dec2))
		if DEBUG:
			print(__file__ + ":ae1 shape:", ae1.shape)
			print(__file__ + ":ae2 shape:", ae2.shape)
			print(__file__ + ":ae2ae1 shape:", ae2ae1.shape)
		return ae1.squeeze(0), ae2.squeeze(0), ae2ae1.squeeze(0)

## USAD_BiLSTM New Model with VAE components
def init_weights(m):
	if isinstance(m, nn.Linear):
		init.normal_(m.weight, mean=0.5, std=0.01)
		if m.bias is not None:
			init.constant_(m.bias, EPS)

class USAD_BiLSTM_VAE(nn.Module):
	def __init__(self, feats):
		super(USAD_BiLSTM_VAE, self).__init__()
		self.name = 'USAD_BiLSTM_VAE'
		self.lr = 0.001 #0.0001
		self.n_feats = feats # 2
		self.n_hidden = 32
		self.n_latent = 16
		self.n_layers = 8
		self.n_window = 12 # USAD w_size =16, 5
		self.n = self.n_feats * self.n_window #=68
		if DEBUG:
			print(__file__ + ":n_feats: ", self.n_feats)
			print(__file__ + ":n_hidden: ", self.n_hidden)
			print(__file__ + ":n_latent: ", self.n_latent)
			print(__file__ + ":n_window: ", self.n_window)
			print(__file__ + ":n: ", self.n)
		self.encoder = nn.LSTM(self.n_feats, self.n_latent, num_layers=self.n_layers, batch_first=True, bidirectional=True)  # Use bidirectional LSTM #bi-directional -> self.n_hidden * 2
		self.decoder1 = nn.LSTM(self.n_latent * 2, self.n_hidden, num_layers=self.n_layers, proj_size=2, batch_first=True)  # Update input size for decoder1
		self.decoder2 = nn.LSTM(self.n_latent * 2, self.n_hidden, num_layers=self.n_layers, proj_size=2, batch_first=True)  # Update input size for decoder2
		self.fc_mu = nn.Sequential(#nn.LayerNorm(self.n_latent * 2),  # VAE component: batch normalization layer
								   nn.Linear(self.n_latent * 2, self.n_latent * 2),)  # VAE component: linear layer for mean)
		self.fc_logvar = nn.Sequential(#nn.LayerNorm(self.n_latent * 2),
								   nn.Linear(self.n_latent * 2, self.n_latent * 2),  # VAE component: linear layer for log variance
								   nn.GELU())
		self.map_h_dec2 = nn.Sequential(#nn.LayerNorm(self.n_latent * 2),
								   nn.Linear(self.n_latent * 2, self.n_feats),  # VAE component: linear layer for mapping h_dec2
								   nn.GELU())
		self.map_c_dec2 = nn.Sequential(#nn.LayerNorm(self.n_latent * 2),
								   nn.Linear(self.n_latent * 2, self.n_hidden),  # VAE component: linear layer for mapping h_dec2
								   nn.GELU())
		self.map_h_ae2ae1 = nn.Sequential(#nn.LayerNorm(self.n_latent * 2),
								   nn.Linear(self.n_latent * 2, self.n_feats),  # VAE component: linear layer for mapping h_dec2
								   nn.GELU())
		self.map_c_ae2ae1 = nn.Sequential(#nn.LayerNorm(self.n_latent * 2),
								   nn.Linear(self.n_latent * 2, self.n_hidden),  # VAE component: linear layer for mapping h_dec2
								   nn.GELU())
		self.layer_norm = nn.LayerNorm(self.n_latent * 2)
		self.apply(init_weights)
	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		if DEBUG:
			print(__file__ + ":mu-shape:", mu.shape)
			print(__file__ + ":std-shape:", std.shape)
		z = mu + eps * std
		return self.layer_norm(z)
	def forward(self, g):
		## Encode
		out_n, (h_n, c_n) = self.encoder(g.view(-1, self.n_window, self.n_feats))
		z = out_n #.view(1, -1)  # Shape: (1, n_hidden*2)
		if DEBUG:
			print(__file__ + ":g-shape:", g.shape)
			print(__file__ + ":z-shape:", z.shape)
			print(__file__ + ":h_n-shape:", h_n.shape)
			print(__file__ + ":c_n-shape:", np.array([c.detach().numpy() for c in c_n]).shape)
		## VAE components
		mu = self.fc_mu(z)  # Shape: (1, n_latent)
		logvar = self.fc_logvar(z)  # Shape: (1, n_latent)
		z_latent = self.reparameterize(mu, logvar)  # Shape: (1, n_latent)
		## Decoders (Phase 1)
		if DEBUG:
			print(__file__ + ":z-latent shape:", z_latent.shape)
		ae1, (h_n_dec1, c_n_dec1) = self.decoder1(z_latent + out_n) # to re-build the input, rebuild based on encoder's out and lantents(bias) (with 0 init)
		 #= o_dec1 #torch.concat((c_n_dec1[0],c_n_dec1[1]),axis=-1)#.view(-1, self.n_hidden * 2)  # Shape: (n_hidden*2,)
		## Decoders (Phase 2)

		if DEBUG:
			plt.subplot(2,1,1)
			print("out_n_shape",out_n.shape)
			plt.plot(out_n[:,0,:].cpu().detach().numpy())
			plt.subplot(2,2,1)
			plt.plot(ae1[:,0,:].cpu().detach().numpy())
			plt.show()
			print(__file__ + ":Decoder 2 input shape:", z_latent.shape)
			print(__file__ + ":Dec2 hidden cons shape:", 
		 			torch.concat((h_n[0: self.n_layers],h_n[self.n_layers:]),axis=-1).shape)
		ae2, (h_n_dec2, c_n_dec2) = self.decoder2(out_n, (h_n_dec1, c_n_dec1))	
												#(self.map_h_dec2(torch.concat((h_n[0: self.n_layers],h_n[self.n_layers:]),axis=-1)), #.unsqueeze(0)
			   									# self.map_c_dec2(torch.concat((c_n[0: self.n_layers],c_n[self.n_layers:]),axis=-1))))#.unsqueeze(0)
		 #= o_dec2 #torch.concat((c_n_dec2[0],c_n_dec2[1]),axis=-1)#.view(-1, self.n_hidden * 2)  # Shape: (n_hidden*2,)
		## Encode-Decode (Phase 2)
		if DEBUG:
			print(__file__ + ":Encoder input shape:", ae1.shape)
		o_enc_ae1, (h_n_enc, c_n_enc) = self.encoder(ae1, (h_n, c_n))#(h_n_dec2, c_n_dec2))
										    # (torch.concat((h_n_dec2[0],h_n_dec2[1]),axis=-1).unsqueeze(0), 
			  								# torch.concat((c_n_dec2[0],c_n_dec2[1]),axis=-1).unsqueeze(0)))#ae1.view(1, self.n_hidden * 2, self.n_hidden * 2), (h_n, c_n))  # Update input size for encoder
		# mu_ae1 = self.fc_mu(o_enc_ae1)  # Shape: (1, n_latent)
		# logvar_ae1 = self.fc_logvar(o_enc_ae1)  # Shape: (1, n_latent)
		# z_latent_ae1 = self.reparameterize(mu_ae1, logvar_ae1)  # Shape: (1, n_latent)

		#c_n_enc_concat =  torch.concat((c_n_enc[0], c_n_enc[1]),axis=-1)
		ae2ae1, (h_xx, c_yy) = self.decoder2(o_enc_ae1, (h_n_dec2, c_n_dec2))# to distinguish the rebuild and the original
							#(self.map_h_ae2ae1(torch.concat((h_n_enc[0: self.n_layers],h_n_enc[self.n_layers:]),axis=-1)), #.unsqueeze(0)
							# self.map_c_ae2ae1(torch.concat((c_n_enc[0: self.n_layers],c_n_enc[self.n_layers:]),axis=-1))))#.unsqueeze(0) #h_n_enc, (h_n_dec2, c_n_dec2))
		if DEBUG:
			print(__file__ + ":ae1 shape:", ae1.shape)
			print(__file__ + ":ae2 shape:", ae2.shape)
			print(__file__ + ":ae2ae1 shape:", ae2ae1.shape)
		return ae1.squeeze(0), ae2.squeeze(0), ae2ae1.squeeze(0), mu.squeeze(0), logvar.squeeze(0)

## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
	def __init__(self, feats):
		super(MSCRED, self).__init__()
		self.name = 'MSCRED'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = feats
		self.encoder = nn.ModuleList([
			ConvLSTM(1, 32, (3, 3), 1, True, True, False),
			ConvLSTM(32, 64, (3, 3), 1, True, True, False),
			ConvLSTM(64, 128, (3, 3), 1, True, True, False),
			]
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
			nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
			nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = g.view(1, 1, self.n_feats, self.n_window)
		for cell in self.encoder:
			_, z = cell(z.view(1, *z.shape))
			z = z[0][0]
		## Decode
		x = self.decoder(z)
		return x.view(-1)

## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
	def __init__(self, feats):
		super(CAE_M, self).__init__()
		self.name = 'CAE_M'
		self.lr = 0.001
		self.n_feats = feats
		self.n_window = feats
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = g.view(1, 1, self.n_feats, self.n_window)
		z = self.encoder(z)
		## Decode
		x = self.decoder(z)
		return x.view(-1)

## MTAD_GAT Model (ICDM 20)
# class MTAD_GAT(nn.Module):
# 	def __init__(self, feats):
# 		super(MTAD_GAT, self).__init__()
# 		self.name = 'MTAD_GAT'
# 		self.lr = 0.0001
# 		self.n_feats = feats
# 		self.n_window = feats
# 		self.n_hidden = feats * feats
# 		self.g = dgl.graph((torch.tensor(list(range(1, feats+1))), torch.tensor([0]*feats)))
# 		self.g = dgl.add_self_loop(self.g)
# 		self.feature_gat = GATConv(feats, 1, feats)
# 		self.time_gat = GATConv(feats, 1, feats)
# 		self.gru = nn.GRU((feats+1)*feats*3, feats*feats, 1)

# 	def forward(self, data, hidden):
# 		hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
# 		data = data.view(self.n_window, self.n_feats)
# 		data_r = torch.cat((torch.zeros(1, self.n_feats), data))
# 		feat_r = self.feature_gat(self.g, data_r)
# 		data_t = torch.cat((torch.zeros(1, self.n_feats), data.t()))
# 		time_r = self.time_gat(self.g, data_t)
# 		data = torch.cat((torch.zeros(1, self.n_feats), data))
# 		data = data.view(self.n_window+1, self.n_feats, 1)
# 		x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
# 		x, h = self.gru(x, hidden)
# 		return x.view(-1), h

## GDN Model (AAAI 21)
# class GDN(nn.Module):
# 	def __init__(self, feats):
# 		super(GDN, self).__init__()
# 		self.name = 'GDN'
# 		self.lr = 0.0001
# 		self.n_feats = feats
# 		self.n_window = 5
# 		self.n_hidden = 16
# 		self.n = self.n_window * self.n_feats
# 		src_ids = np.repeat(np.array(list(range(feats))), feats)
# 		dst_ids = np.array(list(range(feats))*feats)
# 		self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
# 		self.g = dgl.add_self_loop(self.g)
# 		self.feature_gat = GATConv(1, 1, feats)
# 		self.attention = nn.Sequential(
# 			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
# 		)
# 		self.fcn = nn.Sequential(
# 			nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
# 		)

# 	def forward(self, data):
# 		# Bahdanau style attention
# 		att_score = self.attention(data).view(self.n_window, 1)
# 		data = data.view(self.n_window, self.n_feats)
# 		data_r = torch.matmul(data.permute(1, 0), att_score)
# 		# GAT convolution on complete graph
# 		feat_r = self.feature_gat(self.g, data_r)
# 		feat_r = feat_r.view(self.n_feats, self.n_feats)
# 		# Pass through a FCN
# 		x = self.fcn(feat_r)
# 		return x.view(-1)

# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
	def __init__(self, feats):
		super(MAD_GAN, self).__init__()
		self.name = 'MAD_GAN'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.n_window = 5 # MAD_GAN w_size = 5
		self.n = self.n_feats * self.n_window
		self.generator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.discriminator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Generate
		z = self.generator(g.view(1,-1))
		## Discriminator
		real_score = self.discriminator(g.view(1,-1))
		fake_score = self.discriminator(z.view(1,-1))
		return z.view(-1), real_score.view(-1), fake_score.view(-1)

# Proposed Model (VLDB 22)
class TranAD_Basic(nn.Module):
	def __init__(self, feats):
		super(TranAD_Basic, self).__init__()
		self.name = 'TranAD_Basic'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return x

# Proposed Model (FCN) + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD_Transformer(nn.Module):
	def __init__(self, feats):
		super(TranAD_Transformer, self).__init__()
		self.name = 'TranAD_Transformer'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_hidden = 8
		self.n_window = 16 #10
		self.n = 2 * self.n_feats * self.n_window
		self.transformer_encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.ReLU(True))
		self.transformer_decoder1 = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
		self.transformer_decoder2 = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src.permute(1, 0, 2).flatten(start_dim=1)
		tgt = self.transformer_encoder(src)
		return tgt

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.transformer_decoder1(self.encode(src, c, tgt))
		x1 = x1.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
		x1 = self.fcn(x1)
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.transformer_decoder2(self.encode(src, c, tgt))
		x2 = x2.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
		x2 = self.fcn(x2)
		return x1, x2

# Proposed Model + Self Conditioning + MAML (VLDB 22)
class TranAD_Adversarial(nn.Module):
	def __init__(self, feats):
		super(TranAD_Adversarial, self).__init__()
		self.name = 'TranAD_Adversarial'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode_decode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return x

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x = self.encode_decode(src, c, tgt)
		# Phase 2 - With anomaly scores
		c = (x - src) ** 2
		x = self.encode_decode(src, c, tgt)
		return x

# Proposed Model + Adversarial + MAML (VLDB 22)
class TranAD_SelfConditioning(nn.Module):
	def __init__(self, feats):
		super(TranAD_SelfConditioning, self).__init__()
		self.name = 'TranAD_SelfConditioning'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2

# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
	def __init__(self, feats):
		super(TranAD, self).__init__()
		self.name = 'TranAD'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 16 #10 #50
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats) # WHY?
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2
