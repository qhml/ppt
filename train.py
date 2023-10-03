"""
    @Time    : 03/04/2023
    @Author  : qinghua
    @Software: PyCharm
    @File    : train.py

"""
import argparse
import os.path
import pickle
import shutil
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from config import Config
from dataset import DeepScenarioDataset, collate_fn
from model import DigitalTwinModel, DigitalTwinCapability
from utils import get_current_time
from tqdm import tqdm


def train(dtm, dtc, criterion, optimizer, train_dataloader, writer):
    global global_step
    for egos, npcs, ttc in tqdm(train_dataloader):
        optimizer.zero_grad()
        egos = egos.to(Config.device)
        npcs = npcs.to(Config.device)
        ttc = ttc.to(Config.device)
        predicted_egos, predicted_npcs = dtm(egos, npcs)  # [1, N, 19], [1,N,38,19]
        true_egos, predicted_egos = egos.squeeze(0)[1:, :], predicted_egos.squeeze(0)[:-1, :]
        true_npcs, predicted_npcs = npcs.squeeze(0)[1:, :], predicted_npcs.squeeze(0)[:-1, :]
        predicted_ttc = dtc(egos, predicted_egos, npcs, predicted_npcs)
        # dtm_loss = criterion(predicted_egos, true_egos) + criterion(predicted_npcs, true_npcs)
        dtm_loss = criterion(predicted_npcs, true_npcs)
        key_dtm_loss = criterion(predicted_npcs[:, :14], true_npcs[:, :14])
        ttc = ttc[:, 1:]
        dtc_loss = criterion(predicted_ttc, ttc)
        dt_loss = dtm_loss + dtc_loss
        dt_loss.backward()
        optimizer.step()
        writer.add_scalar("Train/DTM-Huber-Loss", key_dtm_loss.cpu().item(), global_step=global_step)
        writer.add_scalar("Train/DTC-Huber-Loss", dtc_loss.cpu().item(), global_step=global_step)
        global_step += 1


# def train(dtm, dtc, criterion, optimizer, train_dataloader, writer):
#     global global_step, best_dtm_loss
#     for egos, npcs, ttc in tqdm(train_dataloader):
#         optimizer.zero_grad()
#         egos = egos.to(Config.device)
#         npcs = npcs.to(Config.device)
#         ttc = ttc.to(Config.device)
#         predicted_egos, predicted_npcs = dtm(egos, npcs)  # [1, N, 19], [1,N,38,19]
#         true_egos, predicted_egos = egos.squeeze(0)[1:, :], predicted_egos.squeeze(0)[:-1, :]
#         true_npcs, predicted_npcs = npcs.squeeze(0)[1:, :], predicted_npcs.squeeze(0)[:-1, :]
#         # dtm_loss = criterion(predicted_egos, true_egos) + criterion(predicted_npcs, true_npcs)
#         dtm_loss = criterion(predicted_npcs[:, :14], true_npcs[:, :14])
#         # dtm_loss = criterion(predicted_egos, true_egos)
#         dtm_loss.backward()
#         optimizer.step()
#         dtm_loss = dtm_loss.cpu().item()
#         writer.add_scalar("Train/DTM-Huber-Loss", dtm_loss, global_step=global_step)
#         if dtm_loss < best_dtm_loss:
#             shutil.rmtree(Config.save_dir)
#             os.mkdir(Config.save_dir)
#             best_dtm_loss = dtm_loss
#             model_name = "dtm_{}.pl".format(round(dtm_loss, 3))
#             torch.save(dtm.state_dict(), os.path.join(Config.save_dir, model_name))
#         global_step += 1





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="pretrain_dtm")
    parser.add_argument("--transformer_d_model", type=int, default=128)
    parser.add_argument("--transformer_n_heads", type=int, default=32)
    parser.add_argument("--transformer_dim_feedforward", type=int, default=1024)
    parser.add_argument("--transformer_n_layers", type=int, default=24)
    parser.add_argument("--pretrained_dtm_path", type=str, default="save/dtm_0.628.pl")
    args = parser.parse_args()
    Config.transformer_d_model = args.transformer_d_model
    Config.transformer_n_heads = args.transformer_n_heads
    Config.transformer_dim_feedforward = args.transformer_dim_feedforward
    Config.transformer_n_layers = args.transformer_n_layers
    Config.pretrained_dtm_path = args.pretrained_dtm_path
    fname = "runs/ [Name: {}] [Time: {}]".format(args.experiment_name, get_current_time())
    writer = SummaryWriter(fname)
    global_step = 0
    data = pickle.load(open(Config.rl_ttc_runs_pkl_path, "rb"))
    pivot = int(len(data) * Config.train_test_split_ratio)
    train_data, test_data = data[:pivot], data[pivot:]
    train_dataset, test_dataset = DeepScenarioDataset(train_data), DeepScenarioDataset(test_data)
    train_dataloader, test_dataloader = DataLoader(train_dataset, Config.batch_size, collate_fn=collate_fn), DataLoader(
        test_dataset,
        Config.batch_size, collate_fn=collate_fn)
    dtm = DigitalTwinModel().to(Config.device)
    if os.path.exists(Config.pretrained_dtm_path):
        print("Loading pretrained DTM")
        dtm.load_state_dict(torch.load(Config.pretrained_dtm_path))
    dtc = DigitalTwinCapability().to(Config.device)
    optimizer = torch.optim.Adam(dtm.parameters())
    criterion = torch.nn.HuberLoss()
    best_dtm_loss = 100000000000000000000000000.0
    for epoch_i in range(Config.n_epochs):
        train(dtm, dtc, criterion, optimizer, train_dataloader, writer)
