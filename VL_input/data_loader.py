import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np
import glob

class VariableLengthDataset(Dataset):
    def __init__(self, flow_num, state_dim):
        self.data_list = []
        for file_path in glob.glob('./Data/*.npy'):
            if 'flow' + str(flow_num) in file_path:
                state_list = np.load(file_path)
                state_list = state_list.reshape(state_list.shape[0], -1, state_dim)
                self.data_list.extend(state_list)
        self.data_list = torch.tensor(self.data_list, dtype=torch.float16)
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def create_dataloader(flow_list, batch_size, state_dim):
    dataloader_list = []
    for flow_num in flow_list:
        dataset = VariableLengthDataset(flow_num, state_dim)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        dataloader_list.append(dataloader)
    return dataloader_list

if __name__ == '__main__':
    dataset = VariableLengthDataset(2, 8)

    

