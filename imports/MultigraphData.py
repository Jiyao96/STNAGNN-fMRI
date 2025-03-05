import torch
import numpy as np
from torch.utils.data import Dataset

# create dataset class 
class MultigraphDataset(Dataset):
    def __init__(self, paths, num):
        self.paths = paths
        self.num = num
        
    def __len__(self):
        return len(self.paths)

    def graph_num(self):
        return self.num

    def __getitem__(self, idx):
        filepaths = np.array(self.paths)
        filepaths = filepaths[idx]
        if isinstance(idx,int):
            data=torch.load(filepaths)
        else:
            # create empty data object
            data=[]
            for i in range(len(filepaths)):
                data.append(torch.load(filepaths[i]))
            data=np.stack(data)
        
        return data