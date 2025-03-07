import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

plt.style.use(['seaborn-v0_8-paper']) #science
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(name, y_true, y_pred, ascore, labels):
	# if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/output.pdf')
	for dim in range(y_true.shape[1]):
		y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(y_t, linewidth=0.2, label='True')#smooth
		ax1.plot(y_p, '-', alpha=0.6, linewidth=0.2, label='Predicted')#smooth
		ax3 = ax1.twinx()
		# l = -l + 1 #DONE: No need to flip the labels, [refer pot.py]{del}, refer main.py, if "TranAD" in model.name or "USAD" in model.name: labels = np.roll(labels, 1, 0)
		ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
		ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
		if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
		ax2.plot(a_s, linewidth=0.2, color='g')#smooth
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		pdf.savefig(fig)
		plt.close()
	pdf.close()

def plotter_nc(name,y_origin, y_true, y_pred, ascore, labels_nc):
	# if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/output.pdf')
	for dim in range(y_true.shape[1]):
		y_ori, y_t, y_p, l, a_s = y_origin[:, dim], y_true[:, dim], y_pred[:, dim], labels_nc[:, dim], ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(y_t, color='deepskyblue',linewidth=0.3, label='Data') # smooth
		ax1.plot(y_p, '-', color='orange',alpha=0.6, linewidth=0.3, label='Predicted') # smooth
		ax1.plot(y_ori, ':',color='lightcoral', alpha=0.6, linewidth=0.3) # smooth
		ax3 = ax1.twinx()
		# l = -l + 1 #DONE: No need to flip the labels, [refer pot.py]{del}, refer main.py, if "TranAD" in model.name or "USAD" in model.name: labels = np.roll(labels, 1, 0)
		colors = ['blue', 'red', 'green']
		labels = ['Normal', 'Disturb Error', 'Swap Value Error']
		for i in range(1,3): # 0 is normal
			ax3.scatter(np.where(l == i), np.ones_like(np.where(l == i)), color=colors[i], label=labels[i], s=3)
		if dim == 0: 
			ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
			ax3.legend(ncol=3, bbox_to_anchor=(0.6, 0))
		ax2.plot(a_s, linewidth=0.2, color='g')#smooth
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		pdf.savefig(fig)
		plt.close()
	pdf.close()