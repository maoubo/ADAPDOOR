import gym
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import csv
import datetime
import os

def make_env(env_id, seed):
    def thunk():
        if env_id == "MountainCar-v0":
            env = gym.make(env_id).unwrapped
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # Collect data for each episode and record it into info
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def performance_normalization(performance, performance_max, performance_min):
    return np.clip(round((performance - performance_min) / (performance_max - performance_min), 4), 0, 1)

def calculate_ne(update, update_trans, trans_begin, per_thre):
    # This function is used to calculate the expected performance for NTP and ASR.
    if update_trans == 0:
        per_ne = 0
    else:
        sche_ne = min((update - trans_begin) / (update_trans - trans_begin), 1)
        per_ne = per_thre * sche_ne
    return per_ne

def weighted_mean(result):
    # This function is used to calculate the final performance (average of NTP and ASR).
    # If there are multiple backdoors, ASR is the average performance of each backdoor.
    w_normal = 0.5
    w_asr = round((0.5 / (result.shape[0] - 1)), 4)
    w_result = []
    for i in range(result.shape[0]):
        if i == 0:
            w_result = result.iloc[0] * w_normal
        else:
            w_result += result.iloc[i] * w_asr

    return np.array(round(w_result, 4))

def action_distribution(action):
    action_list = list(chain(*action))
    action_dis = [arr.item() for arr in action_list]
    print("Action_Mean: {}".format(np.mean(action_dis)))
    plt.plot(action_dis)
    plt.title("Action Distribution")
    plt.show()

def create_folder(path, i):
    name = "schedule {}".format(i)
    folder_path = os.path.join(path, name)
    os.makedirs(folder_path, exist_ok=True)
    model_save_path = os.path.join(folder_path, "model")
    os.makedirs(model_save_path, exist_ok=True)

    return folder_path, model_save_path

def save_results(args, result_normal_per, result_backdoor_asr, poisoning_rate, execution_time):
    if len(result_backdoor_asr) == 0:
        result_backdoor_asr.append(0)

    file_path = '{}/results.csv'.format(args.results_dir)

    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['NTP', 'ASR', "CP", "Poisoning", "Overhead"]
        csv_writer.writerow(header)
        for a_i, b_i, c_i, d_i in zip(result_normal_per, result_backdoor_asr, poisoning_rate, execution_time):
            row = [a_i, b_i, round(2 * (a_i * b_i)/(a_i + b_i + 1e-8), 4), c_i, d_i]
            csv_writer.writerow(row)