import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from config import Config

class VariableLengthDataset(Dataset):
    def __init__(self, flow_num, state_dim, mode='train'):
        self.data_list = []
        file_path_list = []
        if mode == 'train':
            file_path_list = glob.glob('./Data/train/*.npy')
        elif mode == 'valid':
            file_path_list = glob.glob('./Data/valid/*.npy')
        else:
            file_path_list = glob.glob('./Data/*/*.npy')
        for file_path in file_path_list:
            if 'flow' + str(flow_num) in file_path:
                state_list = np.load(file_path)
                state_list = state_list.reshape(state_list.shape[0], -1, state_dim)
                self.data_list.extend(state_list)
        self.data_list = np.array(self.data_list)
        self.data_list = torch.tensor(self.data_list, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def create_dataloader(flow_list, batch_size, state_dim, mode='train'):
    dataloader_list = []
    for flow_num in flow_list:
        dataset = VariableLengthDataset(flow_num, state_dim, mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloader_list.append(dataloader)
    return dataloader_list


class VLLinearDataset(Dataset):
    def __init__(self, flow_num, max_length=32):
        self.data_list = []
        for file_path in glob.glob('./Data/*.npy'):
            state_list = np.load(file_path)
            state_list = pad_to_length(state_list, max_length=max_length)
        state_list = np.array(state_list)
        state_list = torch.tensor(state_list, dtype=torch.float32)

    def pad_to_length(state_list, max_length=32):
        padded_state_list = []
        for state in state_list:
            if len(state) < max_length:
                state = state + [0] * (max_length - len(state))
            if len(state) > max_length:
                state = state[:max_length]
            padded_state_list.append(state)
        return padded_state_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]



if __name__ == '__main__':
    # config = Config()
    # dataloaders = create_dataloader(config.flow_list, config.batch_size, config.state_dim)
    # for dataloader in dataloaders:
    #     print(len(dataloader))
    #     for data in dataloader:
    #         print(data.shape)
    #         break
    def pad_to_length(state_list, max_length=32):
        padded_state_list = []
        for state in state_list:
            if len(state) < max_length:
                state = state + [0] * (max_length - len(state))
            if len(state) > max_length:
                state = state[:max_length]
            padded_state_list.append(state)
        return padded_state_list
    
    state1 = [[1, 2, 3], [0], [1, 2]]
    state1 = pad_to_length(state1)
    print(state1)

