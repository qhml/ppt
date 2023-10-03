"""
    @Time    : 30/03/2023
    @Author  : qinghua
    @Software: PyCharm
    @File    : process_data.py

"""
import os.path
import pickle
import lgsvl
import pandas as pd
import json

from config import Config
from tqdm import tqdm

"""Processing deep scenario data"""


def get_ds_data(runner):
    """pair paths of scenarios and scenario attributes"""
    scene_attr_path_pairs = []
    for root, dirs, files in os.walk(Config.scenario_dir):
        depth = root.count(os.path.sep)
        if depth == 4:
            scenario_dirs = sorted([os.path.join(root, name) for name in os.listdir(root) if not name.endswith(".csv")])
            attribute_fnames = sorted([os.path.join(root, name) for name in os.listdir(root) if name.endswith(".csv")])
            scene_attr_path_pairs += list(zip(scenario_dirs, attribute_fnames))
    """
    pair contents of scenarios and scenario attributes by each run
    Output Example:
        [
            # run 0
            [
                # timestep 0
                (
                    # variables 
                    [1.2,2.1,...]
                    # attribute
                    {"ttc":xxx,"tto":xxx}
                                
                )  
                # timestep 1
                ....      
            ]  
            # run 1
            ....  
        ]
    """
    print(scene_attr_path_pairs)
    greedy_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                    "greedy-strategy" in scene_dir]
    random_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                    "random-strategy" in scene_dir]
    rl_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                "rl_based-strategy" in scene_dir]
    dto_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                 "reward-dto" in scene_dir]
    jerk_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                  "reward-jerk" in scene_dir]
    ttc_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                 "reward-ttc" in scene_dir]
    greedy_ttc_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                        "greedy-strategy" in scene_dir and "reward-ttc" in scene_dir]
    random_ttc_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                        "random-strategy" in scene_dir and "reward-ttc" in scene_dir]
    rl_ttc_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                    "rl_based-strategy" in scene_dir and "reward-ttc" in scene_dir]
    r1_ttc_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                    "road1" in scene_dir and "reward-ttc" in scene_dir]
    r2_ttc_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                    "road2" in scene_dir and "reward-ttc" in scene_dir]
    r3_ttc_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                    "road3" in scene_dir and "reward-ttc" in scene_dir]
    r4_ttc_pairs = [(scene_dir, attr_path) for scene_dir, attr_path in scene_attr_path_pairs if
                    "road4" in scene_dir and "reward-ttc" in scene_dir]
    all_runs = get_all_runs(scene_attr_path_pairs)
    greedy_runs = get_all_runs(greedy_pairs)
    random_runs = get_all_runs(random_pairs)
    rl_runs = get_all_runs(rl_pairs)
    dto_runs = get_all_runs(dto_pairs)
    jerk_runs = get_all_runs(jerk_pairs)
    ttc_runs = get_all_runs(ttc_pairs)
    greedy_ttc_runs = get_all_runs(greedy_ttc_pairs)
    random_ttc_runs = get_all_runs(random_ttc_pairs)
    rl_ttc_runs = get_all_runs(rl_ttc_pairs)
    r1_ttc_runs = get_all_runs(r1_ttc_pairs)
    r2_ttc_runs = get_all_runs(r2_ttc_pairs)
    r3_ttc_runs = get_all_runs(r3_ttc_pairs)
    r4_ttc_runs = get_all_runs(r4_ttc_pairs)
    return all_runs, greedy_runs, random_runs, rl_runs, dto_runs, jerk_runs, ttc_runs, greedy_ttc_runs, random_ttc_runs, rl_ttc_runs, r1_ttc_runs, r2_ttc_runs, r3_ttc_runs, r4_ttc_runs


def get_all_runs(scene_attr_path_pairs):
    all_runs = []
    clean_key = lambda k: k.replace("Attribute[", "").replace("]", "")
    for scene_dir, attr_path in tqdm(scene_attr_path_pairs):
        runs = [[] for _ in range(20)]
        # get scenario attributes
        attr_pdf = pd.read_csv(attr_path)
        for row_id, row in attr_pdf.iterrows():
            run_id = int(row["Execution"])
            scene_fname = row["ScenarioID"] + ".deepscenario"
            attrs = row.to_dict()
            attrs = {clean_key(k): v for k, v in attrs.items()}
            runner.load_scenario_file(os.path.join(scene_dir, scene_fname))
            for i in range(1, 7):
                timeframe = runner.get_scene_by_timestep(timestep=i)
                timeframe = json.loads(timeframe)
                runs[run_id].append([timeframe, attrs])
        all_runs += runs
    return all_runs


class Vocab:
    def __init__(self):
        super(Vocab, self).__init__()
        self.id2str = []
        self.str2id = {}

    def add_if_not_exist(self, s):
        if s not in self.str2id:
            self.str2id[s] = len(self.id2str)
            self.id2str.append(s)

    def tokenize(self, s):
        return self.str2id[s]

    def size(self):
        return len(self.id2str)


def build_vocab(str_list):
    vocab = Vocab()
    for s in str_list:
        vocab.add_if_not_exist(s)
    return vocab


"""Process elevator data"""


def get_ele_passenger_profiles(fname):
    """
    Arrival Time; Arrival Floor; Destination Floor; Mass; Capacity;Loading time; Unloading time;Placeholder

    :param fname:
    :return:
    """
    colnames = ["arrival_time", "arrival_floor", "destination_floor", "mass", "capacity", "loading_time",
                "unloading_time", "placeholder"]
    pdf = pd.read_csv(fname, header=None, names=colnames, index_col=False)
    pdf = pdf[colnames[:-1]]  # remove last column
    pdf["arrival_time"] = pdf["arrival_time"].astype("float")
    pdf["arrival_floor"] = pdf["arrival_floor"].astype("int")
    pdf["destination_floor"] = pdf["destination_floor"].astype("int")
    return pdf


def get_ele_simulator_result(fname):
    """
    Document;Passenger;Source;Destination;ArrivalTime;LiftArrivalTime;DestinationArrivalTime
    """
    colnames = ["document", "id", "arrival_floor", "destination_floor", "arrival_time", "lift_arrival_time",
                "lift_destination_time"]
    pdf = pd.read_csv(fname, names=colnames, delimiter=";", skiprows=1)
    pdf = pdf[colnames[2:-1]]
    pdf["arrival_time"] = pdf["arrival_time"].astype("float")
    pdf["arrival_floor"] = pdf["arrival_floor"].astype("int")
    pdf["destination_floor"] = pdf["destination_floor"].astype("int")
    return pdf


def get_ele_data(dispatcher, peak_type):
    """

    :param dispatcher: list of integers
    :param peak_type: ["lunchpeak","uppeak"]
    :return: joined data of profiles and results
    """
    print(dispatcher, peak_type)
    peak_type = "LunchPeak" if "lunch" in peak_type.lower() else "Uppeak"
    profile_dir = Config.lunchpeak_profile_dir if peak_type == "LunchPeak" else Config.uppeak_profile_dir
    dispatcher_name = "Dispatch_00" if dispatcher == 0 else "Dispatch_M{:2d}".format(dispatcher)
    result_dir = os.path.join(Config.result_dir, dispatcher_name)
    if not os.path.exists(result_dir):
        return False
    result_pdfs = []
    for i in range(10):
        n_variable = "4" if peak_type == "LunchPeak" else "Four"
        profile_fname = "{}_mass_capacity_loading_unloading(CIBSE-office-{}){}.txt".format(n_variable, peak_type, i)
        result_fname = "{}_mass_capacity_loading_unloading(CIBSE-office-{}){}.csv".format(n_variable, peak_type, i)
        profile_pdf = get_ele_passenger_profiles(os.path.join(profile_dir, profile_fname))
        result_pdf = get_ele_simulator_result(os.path.join(result_dir, result_fname))
        result_pdf = profile_pdf.merge(result_pdf, how="right",
                                       on=["arrival_time", "arrival_floor", "destination_floor"])
        result_pdfs.append(result_pdf)
    result_pdf = pd.concat(result_pdfs)
    result_fname = "Dispatcher_{:2d}_{}.pkl".format(dispatcher, peak_type)
    pickle.dump(result_pdf,
        open( os.path.join(Config.elevator_save_dir, result_fname),"wb")
    )
    return result_pdf


if __name__ == '__main__':
    """collect data by runs"""
    # runner = lgsvl.scenariotoolset.ScenarioRunner()
    # all_runs, greedy_runs, random_runs, rl_runs, dto_runs, jerk_runs, ttc_runs, greedy_ttc_runs, random_ttc_runs, rl_ttc_runs, r1_ttc_runs, r2_ttc_runs, r3_ttc_runs, r4_ttc_runs = get_ds_data(
    #     runner)
    # pickle.dump(all_runs, open(Config.all_runs_pkl_path, "wb"))
    # pickle.dump(greedy_runs, open(Config.greedy_runs_pkl_path, "wb"))
    # pickle.dump(random_runs, open(Config.random_runs_pkl_path, "wb"))
    # pickle.dump(rl_runs, open(Config.rl_runs_pkl_path, "wb"))
    # pickle.dump(dto_runs, open(Config.dto_runs_pkl_path, "wb"))
    # pickle.dump(jerk_runs, open(Config.jerk_runs_pkl_path, "wb"))
    # pickle.dump(rl_runs, open(Config.rl_runs_pkl_path, "wb"))
    # pickle.dump(random_ttc_runs, open(Config.random_ttc_runs_pkl_path, "wb"))
    # pickle.dump(greedy_ttc_runs, open(Config.greedy_ttc_runs_pkl_path, "wb"))
    # pickle.dump(rl_ttc_runs, open(Config.rl_ttc_runs_pkl_path, "wb"))
    # pickle.dump(r1_ttc_runs, open(Config.r1_ttc_runs_pkl_path, "wb"))
    # pickle.dump(r2_ttc_runs, open(Config.r2_ttc_runs_pkl_path, "wb"))
    # pickle.dump(r3_ttc_runs, open(Config.r3_ttc_runs_pkl_path, "wb"))
    # pickle.dump(r4_ttc_runs, open(Config.r4_ttc_runs_pkl_path, "wb"))
    # process elevator data

    ele_names = ["dispatch_00_lunchpeak", "dipatcher_00_uppeak"]
    ele_names += ["dispatcher_{:02d}_lunchpeak_variant".format(i) for i in range(1, 100)]
    ele_names += ["dispatcher_{:02d}_uppeak_variant".format(i) for i in range(1, 100)]
    peak_types = ["lunchpeak", "uppeak"]
    for i in range(100):
        for peak_type in peak_types:
            pdf = get_ele_data(i, peak_type)
