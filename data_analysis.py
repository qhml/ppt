"""
    @Time    : 07/07/2023
    @Author  : qinghua
    @Software: PyCharm
    @File    : data_analysis.py

"""
import pickle
import pandas as pd
from config import Config
r1_data=pickle.load(open(Config.r1_ttc_runs_pkl_path, "rb"))
r2_data=pickle.load(open(Config.r2_ttc_runs_pkl_path, "rb"))
r3_data=pickle.load(open(Config.r3_ttc_runs_pkl_path, "rb"))
r4_data=pickle.load(open(Config.r4_ttc_runs_pkl_path, "rb"))
rl_data=pickle.load(open(Config.rl_runs_pkl_path, "rb"))
greedy_data=pickle.load(open(Config.greedy_runs_pkl_path, "rb"))
random_data=pickle.load(open(Config.random_runs_pkl_path, "rb"))
def get_n_npcs(data):
    s=[]
    for run in data:
        for timestep in run:
            d=timestep[0]
            # print(d.keys())
            s.append(sum(1 for k in d.keys() if "NPC" in k))
    return pd.Series(s)
def get_npc_speed(data):
    s = []
    for run in data:
        for timestep in run:
            d = timestep[0]
            # print(d.keys())
            s+=([float(d[k]["dynamic_parameters"]["speed"]) for k in d.keys() if "NPC" in k])
    return pd.Series(s)
rl=get_n_npcs(rl_data)
greedy=get_n_npcs(greedy_data)
random=get_n_npcs(random_data)
rl=pd.Series(rl)
greedy=pd.Series(greedy)
random=pd.Series(random)
print(rl.describe())
print(greedy.describe())
print(random.describe())

greedy_speed=get_npc_speed(greedy_data)
random_speed=get_npc_speed(random_data)
rl_speed=get_npc_speed(rl_data)
greedy_speed=pd.Series(greedy_speed)
random_speed=pd.Series(random_speed)
rl_speed=pd.Series(rl_speed)
print(rl_speed.describe())
print(greedy_speed.describe())
print(random_speed.describe())
#
# r1_npcs=get_n_npcs(r1_data)
# r1_speed=get_npc_speed(r1_data)
# print("r1 statistics")
# print(r1_npcs.describe(),r1_speed.describe())
#
# r2_npcs=get_n_npcs(r2_data)
# r2_speed=get_npc_speed(r2_data)
# print("r2 statistics")
# print(r2_npcs.describe(),r2_speed.describe())
#
# r3_npcs=get_n_npcs(r3_data)
# r3_speed=get_npc_speed(r3_data)
# print("r3 statistics")
# print(r3_npcs.describe(),r3_speed.describe())
#
# r4_npcs=get_n_npcs(r4_data)
# r4_speed=get_npc_speed(r4_data)
# print("r4 statistics")
# print(r4_npcs.describe(),r4_speed.describe())