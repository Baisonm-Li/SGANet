import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py

import numpy as np

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self,data_path):
        super(H5Dataset, self).__init__()
        with h5py.File(data_path,'r') as f:
            self.LRHSI_list = f['LRHSI']
            self.RGB_list = f['RGB']
            self.GT_list = f['GT']

    def __getitem__(self, index):
        return torch.tensor(self.GT_list[index],dtype=torch.float32), \
               torch.tensor(self.LRHSI_list[index],dtype=torch.float32), \
               torch.tensor(self.RGB_list[index],dtype=torch.float32)

    def __len__(self):
        return len(self.GT_list)
    


class NPZDataset(torch.utils.data.Dataset):
    def __init__(self,data_path):
        super(NPZDataset, self).__init__()
        loaded_data = np.load(data_path)
        self.LRHSI_list = loaded_data['LRHSI']
        self.RGB_list = loaded_data['RGB']
        self.GT_list = loaded_data['GT']

    def __getitem__(self, index):
        return torch.tensor(self.GT_list[index],dtype=torch.float32), \
               torch.tensor(self.LRHSI_list[index],dtype=torch.float32), \
               torch.tensor(self.RGB_list[index],dtype=torch.float32)

    def __len__(self):
        return len(self.GT_list)