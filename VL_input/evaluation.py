from model_v4 import AutoencoderGRU
from data_loader import create_dataloader
from config import Config, set_seed
from train import train_autoencoder, test_model
import torch


def eval_model(config, data_loaders, loacl_model=False, model_path=None, net=None):
    with torch.no_grad():
        if loacl_model:
            net = torch.load(model_path, map_location='cuda:0')
        loss_total_list = [0] * len(data_loaders)
        std_list = [0] * len(data_loaders)
        for index, dataloader in enumerate(data_loaders):
            for batch_data in dataloader:
                batch_data = batch_data.to(config.device)
                decoder_outputs, encoder_hidden = net(batch_data)
                loss = torch.abs(torch.round(decoder_outputs) - batch_data.reshape(batch_data.shape[0], -1)).mean()
                loss_total_list[index] += loss.item()
                std_list[index] += torch.std(encoder_hidden).item()
            loss_total_list[index] /= len(dataloader)
            std_list[index] /= len(dataloader)
        return loss_total_list, std_list