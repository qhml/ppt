"""
    @Time    : 30/03/2023
    @Author  : qinghua
    @Software: PyCharm
    @File    : dataset.py

"""
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
from tqdm import tqdm

"""deep scenario dataset"""


class DeepScenarioDataset(Dataset):
    def __init__(self, runs):
        super(DeepScenarioDataset, self).__init__()
        self.runs = runs

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, item):
        run = self.runs[item]
        egos, npcs, ttcs = [], [], []
        for timeframe, attrs in run:
            timestep_ego = self._flatten_scene(timeframe["Ego0"])
            timestep_npc = [self._flatten_scene(v) for k, v in timeframe.items() if k not in ["timestep", "Ego0"]]
            timestep_ttc = attrs["TTC"]
            egos.append(timestep_ego)
            npcs.append(timestep_npc)
            ttcs.append(timestep_ttc)
        return egos, npcs, ttcs

    def _flatten_scene(self, scene):
        data = []
        for params_k, params_v in scene.items():
            for param_k, param_v in params_v.items():
                if type(param_v) is dict:
                    data += (list(param_v.values()))
                else:
                    data.append(param_v)
        data = [float(d) for d in data]
        return data

    def get_max_n_npcs(self):
        runs = self.runs
        timeframes = [timeframe for run in runs for timeframe, attrs in run]
        npcs = [[self._flatten_scene(v) for k, v in timeframe.items() if
                 k not in ["timestep", "Ego0"]] for timeframe in timeframes]
        n_npcs = [len(npc) for npc in npcs]
        return max(n_npcs)


class ElevatorDataset(Dataset):
    def __init__(self, pkl_path):
        super(ElevatorDataset, self).__init__()
        self.data = pickle.load(pkl_path)

    def __getitem__(self, item):
        return self.data.iloc[item].numpy()

    def __len__(self):
        return self.data.size()


def collate_fn(batched_data):
    """

    :param batched_data: each element is a RUN, consisting of timeframes
    :return:
    """
    batched_egos = []
    batched_npcs = []
    batched_ttc = []
    for run_ego, run_npc, run_ttc in batched_data:
        batched_egos.append(run_ego)
        batched_ttc.append(run_ttc)
        padded_timestamp_npc = []
        for timestamp_npc in run_npc:
            padded_timestamp_npc = timestamp_npc
            if len(timestamp_npc) < Config.max_n_npcs:
                fake_npc = [0.0 for _ in range(Config.input_dim)]
                padded_timestamp_npc += ([fake_npc for _ in range(Config.max_n_npcs - len(timestamp_npc))])
            batched_npcs.append(padded_timestamp_npc)

    return torch.tensor(batched_egos), torch.tensor([batched_npcs]), torch.tensor(batched_ttc)


if __name__ == '__main__':
    runs = pickle.load(open(Config.all_runs_pkl_path, "rb"))
    dataset = DeepScenarioDataset(runs)
    print(dataset.get_max_n_npcs())
    dataloader = DataLoader(dataset, collate_fn=collate_fn)
    for egos, npcs, ttc in tqdm(dataloader):
        print(egos.shape, npcs.shape, ttc.shape)
