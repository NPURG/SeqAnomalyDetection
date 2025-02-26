import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
from settings import *
from docs import *
# from beepy import beep


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_to_windows(data, model):
	windows = []
	w_size = model.n_window
	for i, g in enumerate(data):
		if i >= w_size:
			w = data[i - w_size:i]
		else:
			w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
		windows.append(w if any(m in args.model for m in ['TranAD', 'Attention', 'USAD_LSTM', 'USAD_LSTM_NP','USAD_BiLSTM', 'USAD_BiLSTM_VAE']) else w.view(-1))
	return torch.stack(windows).to(device)

def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels', 'labels_nc', 'test_ori']:
		if dataset == 'addr1394':
			file = file + DATA_FILE_SUFFIX
		if dataset == 'addr1394-single':
			file = file + DATA_FILE_SUFFIX_SINGLE
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	if args.less:
		loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])#, pin_memory=True)
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])#, pin_memory=True)
	labels = loader[2]
	labels_nc = loader[3]
	test_origin = loader[4]
	return train_loader, test_loader, labels, labels_nc, test_origin

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model[0]}_{args.dataset}/' #TODO: path names for all models
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict': scheduler.state_dict(),
		'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double().to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model[0]}_{args.dataset}/model.ckpt'#TODO: path names for all models
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname, map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1
		accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
	"""
	Perform backpropagation for training a model.

	Args:
		epoch (int): The current epoch number.
		model: The model to be trained.
		data: The input data.
		dataO: The original data.
		optimizer: The optimizer for updating model parameters.
		scheduler: The learning rate scheduler.
		training (bool, optional): Whether the model is in training mode. Defaults to True.

	Returns:
		tuple: A tuple containing the loss and the learning rate.

	Raises:
		None
	"""
	if DEBUG:
		print("model.name: ", model.name)
	l = nn.MSELoss(reduction='mean' if training else 'none')
	feats = dataO.shape[1]
	data = data.to(device)
	dataO = dataO.to(device)
	if 'USAD' == model.name:
		l = nn.MSELoss(reduction = 'none') #comment: [none reduction] keeps the shape of the 2nd power with no scale (mean) or sum (central moment).
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			train_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=False)
			for i, d in enumerate(train_loader):
				if DEBUG:
					print("d.shape:", d.shape)
				ae1s, ae2s, ae2ae1s = model(d.to(device))
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d) #comment: scaled here
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d) 
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr'] # train return: loss, lr
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			if USE_RATIO:
				test_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=False)
				for d in test_loader: 
					ae1, ae2, ae2ae1 = model(d)
					ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
				ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			else:
				data = data[:TEST_NUM]
				for d in data: 
					ae1, ae2, ae2ae1 = model(d)
					ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
				ae1s, ae2s, ae2ae1s = torch.stack(ae1s).squeeze(), torch.stack(ae2s).squeeze(), torch.stack(ae2ae1s).squeeze()
			if DEBUG:
				print("ae1s.shape: ", ae1s.shape)
				print("y_pred.shape: ", ae1s[:, data.shape[1]-feats:data.shape[1]].shape)
				print("feat.shape: ", feats)
				print("data.shape: ", data.shape)
			y_pred = ae2s[:, data.shape[1]-feats:data.shape[1]].reshape(-1, feats) #0.2 * ae1s + 0.2 * ae2s + 0.6 * 
			loss = USAD_ALPHA * l(ae1s, data) + USAD_BETA * l(ae2ae1s, data)#0.6 * l(ae1s, data) + 0.4 * l(ae2ae1s, data)#0.8 * l(ae1s, data) + 0.2 * l(ae2ae1s, data)#0.4 * l(ae1s, data) + 0.6 * l(ae2ae1s, data) #0.1 0.9 #<arg>
			loss = loss[:, data.shape[1]-feats:data.shape[1]].reshape(-1, feats)
			return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy() # forward(test) return: loss, ypred
	elif 'USAD_GELU' == model.name: #[run] python .\main.py --model USAD_GELU --dataset addr1394-single --retrain
		l = nn.MSELoss(reduction = 'none') #comment: [none reduction] keeps the shape of the 2nd power with no scale (mean) or sum (central moment).
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			train_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=False)
			for i, d in enumerate(train_loader):
				if DEBUG:
					print("d.shape:", d.shape)
				ae1s, ae2s, ae2ae1s = model(d.to(device))
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d) # step1: l(ae1s, d) # #comment: scaled here
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d) # step1: l(ae2s, d) # 
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.sum(l1 + l2) / 2#torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr'] # train return: loss, lr
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			if USE_RATIO:
				test_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=False)
				for d in test_loader: 
					ae1, ae2, ae2ae1 = model(d)
					ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
				ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			else:
				data = data[:TEST_NUM]
				for d in data: 
					ae1, ae2, ae2ae1 = model(d)
					ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
				ae1s, ae2s, ae2ae1s = torch.stack(ae1s).squeeze(), torch.stack(ae2s).squeeze(), torch.stack(ae2ae1s).squeeze()
			if DEBUG:
				print("ae1s.shape: ", ae1s.shape)
				print("y_pred.shape: ", ae1s[:, data.shape[1]-feats:data.shape[1]].shape)
				print("feat.shape: ", feats)
				print("data.shape: ", data.shape)
			y_pred = ae2s[:, data.shape[1]-feats:data.shape[1]].reshape(-1, feats) #0.2 * ae1s + 0.2 * ae2s + 0.6 * 
			loss = USAD_ALPHA * l(ae1s, data) + USAD_BETA * l(ae2ae1s, data)#0.6 * l(ae1s, data) + 0.4 * l(ae2ae1s, data)#0.8 * l(ae1s, data) + 0.2 * l(ae2ae1s, data)#0.4 * l(ae1s, data) + 0.6 * l(ae2ae1s, data) #0.1 0.9 #<arg>
			loss = loss[:, data.shape[1]-feats:data.shape[1]].reshape(-1, feats)
			return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy() # forward(test) return: loss, ypred
	elif 'WaveletUSAD' == model.name:	
		l = nn.MSELoss(reduction = 'none') #comment: [none reduction] keeps the shape of the 2nd power with no scale (mean) or sum (central moment).
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			train_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=False)
			for i, d in enumerate(train_loader): #DONE:batches here!
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d) #comment: scaled here
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr'] # train return: loss, lr
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = USAD_ALPHA* l(ae1s, data) + USAD_BETA * l(ae2ae1s, data)#0.6 * l(ae1s, data) + 0.4 * l(ae2ae1s, data)#0.8 * l(ae1s, data) + 0.2 * l(ae2ae1s, data)#0.4 * l(ae1s, data) + 0.6 * l(ae2ae1s, data) #0.1 0.9 #<arg>
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy() # forward(test) return: loss, ypred
	elif 'USAD_BiLSTM' == model.name or 'USAD_LSTM' == model.name or 'USAD_LSTM_NP' == model.name:
		l = nn.MSELoss(reduction='none') #nn.CosineEmbeddingLoss(reduction='none') #
		n = epoch + 1
		w_size = model.n_window
		l1s, l2s = [], []
		if training:
			# plt.plot(data[:,0,:].cpu().detach().numpy())
			# plt.show()
			train_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=False)

			for i, d in enumerate(train_loader): #DONE:batches here!
				if DEBUG:
					print("--------------------training-------------------")
					print("data.shape:", d.shape)
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1 / n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1 / n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item())
				l2s.append(torch.mean(l2).item())
				if DEBUG:
					print("l1.shape:", l1.shape)
					print("l2.shape:", l2.shape)
					print("ae1s.shape:", ae1s.shape)
					print("ae2s.shape:", ae2s.shape)
					print("ae2ae1s.shape:", ae2ae1s.shape)
					print("----------------------------------------------")
				loss = torch.mean(l1 + l2) 
				optimizer.zero_grad()
				loss.backward() # GAN
				optimizer.step()
			# scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data:
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1)
				ae2s.append(ae2)
				ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			if DEBUG:
				print("--------------------testing-------------------")
				print("data.shape:", data.shape)
				print("ae1s.shape:", ae1s.shape)
				print("ae2s.shape:", ae2s.shape)
				print("ae2ae1s.shape:", ae2ae1s.shape)
				print("----------------------------------------------")
			y_pred = ae1s[:,-1,:].view(-1, model.n_feats) # TODO: not this slice
			loss = USAD_BLV_ALPHA * l(ae1s, data) +  USAD_BLV_BETA * l(ae2ae1s, data) # + 0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
			loss = loss[:,-1,:].view(-1, feats) # TODO: not this slice
			if DEBUG:
				print("loss.shape:", loss.shape)
				print("----------------------------------------------")
			# loss = loss[:, model.n_feats * model.n_window:].view(-1, model.n_feats)
			return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
	elif 'USAD_BiLSTM_VAE' == model.name:
		l = nn.MSELoss(reduction='none') #nn.CosineEmbeddingLoss(reduction='none') #
		n = epoch + 1
		w_size = model.n_window
		l0s, l1s, l2s = [], [], []
		if training:
			plt.plot(data[:,0,:].cpu().detach().numpy())
			plt.show()
			train_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=False)
			for i, d in enumerate(train_loader): #DONE:batches here!
				ae1s, ae2s, ae2ae1s, mu, logvar = model(d)
				l0 = l(ae1s, d) - KL_WEIGHT * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) #TODO: test dims
				l1 = (1 / n) * l(ae1s, d) + (1 - 1 / n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1 / n) * l(ae2ae1s, d)
				l0s.append(torch.mean(l0).item())
				l1s.append(torch.mean(l1).item())
				l2s.append(torch.mean(l2).item())
				if DEBUG:
					print("d.shape:", d.shape)
					print("vae loss shape:", (1 + logvar - mu.pow(2) - logvar.exp()).shape)
					print("l1.shape:", l1.shape)
					print("l2.shape:", l2.shape)
				loss = torch.mean(l1 + l2) 
				optimizer.zero_grad()
				torch.mean(l0).backward(retain_graph=True) # fitting the data with VAE
				loss.backward() # GAN
				optimizer.step()
			# scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s, mus, logvars = [], [], [], [], []
			for d in data:
				ae1, ae2, ae2ae1, mu, logvar = model(d)
				ae1s.append(ae1)
				ae2s.append(ae2)
				ae2ae1s.append(ae2ae1)
				mus.append(mu)
				logvars.append(logvar)
			ae1s, ae2s, ae2ae1s, mus, logvars = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s), torch.stack(mus), torch.stack(logvars)
			if DEBUG:
				print("--------------------testing-------------------")
				print("data.shape:", data.shape)
				print("ae1s.shape:", ae1s.shape)
				print("ae2s.shape:", ae2s.shape)
				print("ae2ae1s.shape:", ae2ae1s.shape)
				print("mus.shape:", mus.shape)
				print("logvars.shape:", logvars.shape)
				print("----------------------------------------------")
			y_pred = ae1s[:,6,:].view(-1, model.n_feats) # TODO: not this slice
			# DOC: USAD_BiLSTM_VAE_Loss_design
			loss = USAD_BLV_ALPHA * l(ae1s, data) +  USAD_BLV_BETA * l(ae2ae1s, data) # + 0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
			loss = loss[:,6,:].view(-1, feats) # TODO: not this slice
			if DEBUG:
				print("loss.shape:", loss.shape)
				print("----------------------------------------------")
			# loss = loss[:, model.n_feats * model.n_window:].view(-1, model.n_feats)
			return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
	elif 'TranAD' == model.name:
		l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data, device=device)
		dataset = TensorDataset(data_x, data_x, device=device)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d, _ in dataloader:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window.to(device), elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			for d, _ in dataloader:
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]
			loss = l(z, elem)[0]
			return loss.detach().numpy(), z.detach().numpy()[0]
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()

if __name__ == '__main__':
	print("device:", device)
	train_loader, test_loader, labels, labels_nc, test_origin = load_dataset(args.dataset)
	# labels = -labels + 1
	if args.model in ['MERLIN']:
		eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model[0], labels.shape[1]) ##DONE:ERROR HERE; fixed: label not reshaped (-1,1)
	model = model.to(device)
	print(model)
	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	if model.name in ['Attention', 'DAGMM', 'USAD', 'USAD_GELU','USAD_LSTM', 'USAD_LSTM_NP', 'USAD_BiLSTM', 'USAD_BiLSTM_VAE', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
		trainD, testD = trainD.to(device), testD.to(device)
		print("Converted to windows:")
		print("trainD shape:", trainD.shape)
		print("testD shape:", testD.shape)

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = EPOCHS; e = epoch + 1; start = time() #epochs = 5 25#<args>
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		save_model(model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase TODO: metrics may be wrong
	with torch.no_grad():
		if "TranAD" in model.name or "USAD" in model.name: labels = np.roll(labels, 1, 0) #<arg> "TranAD" in model.name or "USAD" in model.name # reason: predict is shifted by 1(latter)
		model.train(False)
		model.eval()
		print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
		if USE_RATIO:
			loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
		else:
			testD, testO = testD[:TEST_NUM], testO[:TEST_NUM]
			labels = labels[:TEST_NUM]
			loss, y_pred = backprop(0, model, testD[:TEST_NUM], testO[:TEST_NUM], optimizer, scheduler, training=False)
		if DEBUG: #plt
			print("[USE_RATIO]:", USE_RATIO)
			print("loss shape:", loss.shape)
			print("y_pred shape:", y_pred.shape)
			print("testO shape:", testO.shape)
			print("labels shape:", labels.shape)
			# sns.set_style("darkgrid")
			# sns.set_palette("pastel")

			# plt.figure(figsize=(12, 8))
			plt.subplot(2, 1, 1)
			plt.plot(loss, c='red', label='loss', alpha=0.8, linewidth=2)
			plt.plot(labels, c='black', label='labels', linestyle='--', alpha=0.6, linewidth=1.5)
			plt.title(f'{args.model} on {args.dataset}')
			# plt.fill_between(range(len(labels)), labels.squeeze(), alpha=0.2, color='black')
			plt.ylabel('Value')
			plt.xlabel('Time')
			plt.legend()
			yy = np.roll(y_pred, -1, 0)
			plt.subplot(2, 1, 2)
			plt.plot(testO, c='green', label='y_test', alpha=0.8, linewidth=1)
			plt.plot(yy, c='blue', label='y_pred', linestyle='-', alpha=0.6, linewidth=1.5)
			# plt.fill_between(range(len(yy)), yy.squeeze(), alpha=0.2, color='blue')
			plt.xlabel('Time')
			plt.ylabel('Value')
			plt.legend()

			plt.title(f'{args.model} on {args.dataset}')

			plt.show()
  			###
  
		### Plot curves
		if not args.test:
			if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0) 
			# plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)
			plotter_nc(f'{args.model}_{args.dataset}', test_origin, testO, y_pred, loss, labels_nc)
		else:
			if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0) 
			# plotter(f'{args.model}_{args.dataset}_test', testO, y_pred, loss, labels)
			plotter_nc(f'{args.model}_{args.dataset}_test', test_origin, testO, y_pred, loss, labels_nc)

		### Scores
		df = pd.DataFrame()
		lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
		
		for i in range(loss.shape[1]):
			lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
			result, pred = pot_eval(lt, l, ls); preds.append(pred)
			result_df = pd.DataFrame(result, index=[0])  # Convert dictionary to DataFrame
			df = pd.concat([df, result_df], ignore_index=True)
			# df = pd.concat([df, result], ignore_index=True)
		# preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
		# pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
		lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
		labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
		result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal, plot=True)
		result.update(hit_att(loss, labels))
		result.update(ndcg(loss, labels))
		print(df)
		pprint(result)
		# pprint(getresults2(df, result))
		# beep(4)
