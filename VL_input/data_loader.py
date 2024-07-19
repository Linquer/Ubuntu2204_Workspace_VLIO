import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from config import Config

class VariableLengthDataset(Dataset):
    def __init__(self, flow_num, state_dim):
        self.data_list = []
        for file_path in glob.glob('./Data/*.npy'):
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

def create_dataloader(flow_list, batch_size, state_dim):
    dataloader_list = []
    for flow_num in flow_list:
        dataset = VariableLengthDataset(flow_num, state_dim)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloader_list.append(dataloader)
    return dataloader_list



if __name__ == '__main__':
    config = Config()
    dataloaders = create_dataloader(config.flow_list, config.batch_size, config.state_dim)
    for dataloader in dataloaders:
        print(len(dataloader))
        for data in dataloader:
            print(data.shape)
            break

