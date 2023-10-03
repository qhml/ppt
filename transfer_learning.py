"""
    @Time    : 25/06/2023
    @Author  : qinghua
    @Software: PyCharm
    @File    : transfer_learning.py

"""
import argparse
import os
import pickle

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import DeepScenarioDataset, collate_fn
from model import DigitalTwinModel, DigitalTwinCapability
from produce_prompt import prepare_prompt
from utils import get_current_time

import torch


def gaussian_kernel(x, y, sigma=1.0):
    delta = x - y
    squared_distance = torch.sum(delta * delta, dim=1)
    return torch.exp(-squared_distance / (2 * sigma * sigma))


def compute_mmd_loss(x, y, kernel=gaussian_kernel, sigma=1.0):
    x_size = x.size(0)
    y_size = y.size(0)

    xx_kernel = kernel(x, x, sigma)
    yy_kernel = kernel(y, y, sigma)
    xy_kernel = kernel(x, y, sigma)

    xx_kernel_sum = torch.sum(xx_kernel) - torch.trace(xx_kernel)
    yy_kernel_sum = torch.sum(yy_kernel) - torch.trace(yy_kernel)
    xy_kernel_sum = torch.sum(xy_kernel)

    mmd_loss = (xx_kernel_sum / (x_size * (x_size - 1))
                + yy_kernel_sum / (y_size * (y_size - 1))
                - 2 * xy_kernel_sum / (x_size * y_size))

    return mmd_loss


def transfer_learning(src_dtm, src_dtc, target_dtm, target_dtc, src_dataloader, tgt_dataloader, prompt_dataloader,
                      marginal_loss,
                      conditional_loss, criterion, src_optimizer, tgt_optimizer, writer):
    global global_step
    for src_egos, src_npcs, src_ttc in tqdm(src_dataloader):
        tgt_egos, tgt_npcs, tgt_ttc = next(src_dataloader)
        prompt_input, prompt_label = next(prompt_dataloader)
        src_optimizer.zero_grad()
        tgt_optimizer.zero_grad()
        src_egos = src_egos.to(Config.device)
        src_npcs = src_npcs.to(Config.device)
        src_ttc = src_ttc.to(Config.device)
        tgt_egos = tgt_egos.to(Config.device)
        tgt_npcs = tgt_npcs.to(Config.device)
        tgt_ttc = tgt_ttc.to(Config.device)
        predicted_src_egos, predicted_src_npcs = src_dtm(src_egos, src_npcs)  # [1, N, 19], [1,N,38,19]
        true_src_egos, predicted_src_egos = src_egos.squeeze(0)[1:, :], predicted_src_egos.squeeze(0)[:-1, :]
        true_src_npcs, predicted_src_npcs = src_npcs.squeeze(0)[1:, :], predicted_src_npcs.squeeze(0)[:-1, :]
        predicted_src_ttc = src_dtc(src_egos, predicted_src_egos, src_npcs, predicted_src_npcs)
        predicted_src_prompt = src_dtc(prompt_input)
        # dtm_loss = criterion(predicted_egos, true_egos) + criterion(predicted_npcs, true_npcs)
        src_dtm_loss = criterion(predicted_src_npcs, true_src_npcs)
        src_prompt_loss = -1 * criterion(predicted_src_prompt, prompt_label)
        # key_dtm_loss = criterion(predicted_npcs[:, :14], true_npcs[:, :14])
        src_ttc = src_ttc[:, 1:]
        src_dtc_loss = criterion(predicted_src_ttc, src_ttc)
        src_dt_loss = src_dtm_loss + src_dtc_loss
        src_dt_loss.backward()
        src_dtm_marginal = src_dtm.marginal_output
        src_dtm_conditional = src_dtm.conditional_output
        predicted_tgt_egos, predicted_tgt_npcs = tgt_dtm(tgt_egos, tgt_npcs)  # [1, N, 19], [1,N,38,19]
        true_tgt_egos, predicted_tgt_egos = tgt_egos.squeeze(0)[1:, :], predicted_tgt_egos.squeeze(0)[:-1, :]
        true_tgt_npcs, predicted_tgt_npcs = tgt_npcs.squeeze(0)[1:, :], predicted_tgt_npcs.squeeze(0)[:-1, :]
        predicted_tgt_ttc = tgt_dtc(tgt_egos, predicted_tgt_egos, tgt_npcs, predicted_tgt_npcs)
        # dtm_loss = criterion(predicted_egos, true_egos) + criterion(predicted_npcs, true_npcs)
        tgt_dtm_loss = criterion(predicted_tgt_npcs, true_tgt_npcs)
        # key_dtm_loss = criterion(predicted_npcs[:, :14], true_npcs[:, :14])
        tgt_ttc = tgt_ttc[:, 1:]
        tgt_dtc_loss = criterion(predicted_tgt_ttc, tgt_ttc)
        tgt_dt_loss = tgt_dtm_loss + tgt_dtc_loss
        tgt_dt_loss.backward()
        tgt_dtm_marginal = tgt_dtm.marginal_output
        tgt_dtm_conditional = tgt_dtm.conditional_output
        predicted_tgt_prompt = tgt_dtc(prompt_input)
        tgt_prompt_loss = criterion(predicted_tgt_prompt, prompt_label)
        marginal = marginal_loss(src_dtm_marginal, src_dtm_marginal)
        marginal += marginal_loss(tgt_dtm_marginal, tgt_dtm_marginal)
        conditional = compute_mmd_loss(src_dtm_conditional, src_dtm_conditional)
        conditional += compute_mmd_loss(tgt_dtm_conditional, tgt_dtm_conditional)
        prompt_loss = src_prompt_loss + tgt_prompt_loss
        prompt_loss.backward()
        marginal.backward()
        conditional.backward()
        src_optimizer.step()
        tgt_optimizer.step()

        writer.add_scalar("Train/Marginal-Loss", marginal.cpu().item(), global_step=global_step)
        writer.add_scalar("Train/Conditional-Loss", conditional.cpu().item(), global_step=global_step)
        global_step += 1


def validate(dtm, dtc, test_dataloader, metric, writer):
    global global_step
    with torch.no_grad():
        for egos, npcs, ttc in tqdm(test_dataloader):
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
            dtc_loss = metric(predicted_ttc, ttc)
            dt_loss = dtm_loss + dtc_loss
            writer.add_scalar("Test/DTM-Huber-Loss", key_dtm_loss.cpu().item(), global_step=global_step)
            writer.add_scalar("Test/DTC-Huber-Loss", dtc_loss.cpu().item(), global_step=global_step)


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
    src_pkl_path, tgt_pkl_path, test_pkl_path = Config.transfer_pairs[1]
    src_data = pickle.load(open(src_pkl_path, "rb"))
    tgt_data = pickle.load(open(tgt_pkl_path, "rb"))
    test_data = pickle.load(open(test_pkl_path, "rb"))
    tgt_prompt = prepare_prompt(tgt_data)
    src_dataloader = DataLoader(src_data, Config.batch_size, collate_fn=collate_fn)
    tgt_dataloader = DataLoader(src_data, Config.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(src_data, Config.batch_size, collate_fn=collate_fn)
    prompt_dataloader = DataLoader(tgt_dataloader, Config.batch_size, collate_fn=collate_fn)
    src_dtm = DigitalTwinModel().to(Config.device)
    if os.path.exists(Config.pretrained_dtm_path):
        print("Loading pretrained DTM")
        src_dtm.load_state_dict(torch.load(Config.pretrained_src_dtm_path))
    src_dtc = DigitalTwinCapability().to(Config.device)
    tgt_dtm = DigitalTwinModel().to(Config.device)
    if os.path.exists(Config.pretrained_dtm_path):
        print("Loading pretrained DTM")
        tgt_dtm.load_state_dict(torch.load(Config.pretrained_tgt_dtm_path))
    tgt_dtc = DigitalTwinCapability().to(Config.device)
    src_optimizer = torch.optim.Adam(
        list(src_dtm.parameters()) + list(src_dtc.parameters())
    )
    tgt_optimizer = torch.optim.Adam(
        list(tgt_dtm.parameters()) + list(tgt_dtc.parameters())
    )
    marginal_loss = torch.nn.HuberLoss()
    conditional_loss = torch.nn.NLLLoss()
    criterion = torch.nn.HuberLoss()
    global_step = 0
    for epoch_i in Config.n_epochs:
        transfer_learning(src_dtm, src_dtc, tgt_dtm, tgt_dtc, src_dataloader, tgt_dataloader, prompt_dataloader,
                          marginal_loss,
                          conditional_loss, criterion,
                          src_optimizer, tgt_optimizer, writer)
        validate(tgt_dtm, tgt_dtc, test_dataloader, criterion, writer)
