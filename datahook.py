import numpy as np

def cb_datahook(filename):
    def hook(lcdict):
        print('hook called')
        oo = lcdict['xc'][:4096].reshape((-1,16,1))
        print(f"{filename} shape: ", oo.shape)
        np.save(f"{filename}.npy", oo)
        print(f"saved {filename}.npy")
    return hook
    