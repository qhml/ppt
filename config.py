"""
    @Time    : 30/03/2023
    @Author  : qinghua
    @Software: PyCharm
    @File    : config.py

"""
import torch


class Config:
    """configurations for the whole project"""
    # settings for process deep scenario data
    scenario_dir = "data/DeepScenario-main/deepscenario-dataset"
    all_runs_pkl_path = "data/all_runs.pkl"
    greedy_runs_pkl_path = "data/greedy_runs.pkl"
    random_runs_pkl_path = "data/random_runs.pkl"
    rl_runs_pkl_path = "data/rl_runs.pkl"
    dto_runs_pkl_path = "data/dto_runs.pkl"
    jerk_runs_pkl_path = "data/jerk_runs.pkl"
    ttc_runs_pkl_path = "data/ttc_runs.pkl"
    greedy_ttc_runs_pkl_path = "data/greedy_ttc_runs.pkl"
    random_ttc_runs_pkl_path = "data/random_ttc_runs.pkl"
    rl_ttc_runs_pkl_path = "data/rl_ttc_runs.pkl"
    r1_ttc_runs_pkl_path = "data/r1_ttc_runs.pkl"
    r2_ttc_runs_pkl_path = "data/r2_ttc_runs.pkl"
    r3_ttc_runs_pkl_path = "data/r3_ttc_runs.pkl"
    r4_ttc_runs_pkl_path = "data/r4_ttc_runs.pkl"
    model_vocab_pkl_path = "data/model_vocab.pkl"
    color_vocab_pkl_path = "data/color_vocab.pkl"
    type_vocab_pkl_path = "data/type_vocab.pkl"
    # settings for processing elevator data
    lunchpeak_profile_dir = "data/LunchPeakPassengerProfiles/Four_mass_capacity_loading_unloading"
    uppeak_profile_dir = "data/UpPeakPassengerProfiles/Four_mass_capacity_loading_unloading"
    result_dir = "data/passenger_results"
    elevator_save_dir = "data/elevator"
    # settings for training
    train_test_split_ratio = 0.9
    n_epochs = 10000000000
    input_dim = 19
    input_hidden_dim = 32
    lstm_hidden_dim = 16
    max_n_npcs = 38
    batch_size = 1  # equal to the number of scenarios in one run
    output_dim = 1
    transformer_d_model = 128
    transformer_n_heads = 32
    transformer_dim_feedforward = 1024
    transformer_n_layers = 24
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # settings for model
    # settings for evaluation
    save_dir = "save"
    pretrained_src_dtm_path = "save/dtm_0.628.pl"
    pretrained_tgt_dtm_path = "save/dtm_0.628.pl"
    # config for transfer learning
    transfer_pairs = []
