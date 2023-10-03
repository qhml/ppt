"""
    @Time    : 02/04/2023
    @Author  : qinghua
    @Software: PyCharm
    @File    : model.py

"""
import torch.nn
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from config import Config


class DigitalTwinModel(nn.Module):
    def __init__(self):
        super(DigitalTwinModel, self).__init__()
        self.input_linear = nn.Linear(Config.input_dim, Config.transformer_d_model)
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=Config.transformer_d_model,
                                                                 nhead=Config.transformer_n_heads,
                                                                 dim_feedforward=Config.transformer_dim_feedforward,
                                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer,
                                                      num_layers=Config.transformer_n_layers)
        self.output_linear = nn.Linear(Config.transformer_d_model, Config.input_dim)
        self.marginal_output = None
        self.conditional_output = None

    def forward(self, ego, npcs):
        """
        forward pass of DTM
        :param ego: [1, N, 19]
        :param npcs: [1,N,38,19]
        :return:
        """
        ego = ego.unsqueeze(2)
        combined = torch.cat([ego, npcs], dim=2)
        combined = self.input_linear(combined).squeeze(0)  # [N, 39, 19]
        self.marginal_output=combined
        out = self.transformer_encoder(combined)
        out = self.output_linear(out)
        self.conditional_output=out
        ego = out[:, 0, :].squeeze(1).unsqueeze(0)
        npcs = out[:, 1:, :].unsqueeze(0)
        return ego, npcs


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.input_transform = nn.Linear((Config.max_n_npcs + 1) * Config.input_dim, Config.lstm_hidden_dim)
        self.lstm = nn.LSTM(input_size=Config.lstm_hidden_dim, hidden_size=Config.lstm_hidden_dim)
        self.output_transform = nn.Linear(Config.lstm_hidden_dim, Config.output_dim)
        self.marginal_output = None
        self.conditional_output = None

    def forward(self, egos, npcs):
        egos = egos.unsqueeze(dim=2)
        x = torch.concat([egos, npcs], dim=2).squeeze(0)  # N* 39 *19
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.input_transform(x)
        x = torch.relu(x)

        output, _ = self.lstm(x)
        self.marginal_output = output
        output = self.output_transform(output)
        output = torch.tanh(output)
        self.conditional_output = output
        return output


class DigitalTwinCapability(nn.Module):
    def __init__(self):
        super(DigitalTwinCapability, self).__init__()
        self.ego_linear = nn.Linear(Config.input_dim, Config.lstm_hidden_dim)
        self.npc_linear = nn.Linear(Config.input_dim * Config.max_n_npcs, Config.lstm_hidden_dim)
        self.out_linear = nn.Linear(Config.lstm_hidden_dim, 1)
        self.marginal_output = None
        self.conditional_output = None

    def forward(self, ego, predicted_ego, npcs, predicted_npcs):
        ego = ego[:, :-1, :]
        npcs = npcs[:, :-1, :, :]
        n_steps = ego.shape[1]
        ego = ego + predicted_ego
        npcs = npcs + predicted_npcs
        npcs = npcs.view(1, n_steps, -1)  # [1, N, 38*19]
        ego = self.ego_linear(ego)
        npcs = self.npc_linear(npcs)
        combined = ego + npcs
        self.marginal_output = combined
        conditional_output = self.out_linear(combined).squeeze(2)
        self.conditional_output = conditional_output
        return conditional_output


if __name__ == '__main__':
    """test model"""
    batch_size = 1
    n_timesteps = 6
    n_npcs = 38
    ego = torch.rand((batch_size, n_timesteps, Config.input_dim))
    npcs = torch.rand((batch_size, n_timesteps, n_npcs, Config.input_dim))
    # model = DigitalTwinModel()
    model = LSTMModel()
    out = model(ego, npcs)
    print(out.shape)
    print(out)
