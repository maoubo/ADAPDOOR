import gym
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import csv
import datetime
import os

def make_env(args, seed):
    def thunk():
        env = gym.make(args.env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # Collect data for each episode and record it into info

        # Normalization
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

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

def create_folder(path, i):
    name = "Schedule {}".format(i)
    folder_path = os.path.join(path, name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"New folder created：{folder_path}")

    return folder_path

def save_results(final_judge, args, result_normal_per, result_backdoor_asr):
    if len(result_backdoor_asr) == 0:
        result_backdoor_asr.append(0)
    if not final_judge:
        file_path = '{}/results.csv'.format(args.results_dir)
    else:
        file_path = '{}/{}_{}.csv'.format(args.summary_dir, args.reward_hacking_method, args.seed)

    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['NTP', 'ASR', "CP"]
        csv_writer.writerow(header)
        for a_i, b_i in zip(result_normal_per, result_backdoor_asr):
            row = [a_i, b_i, 2 * (a_i * b_i)/(a_i + b_i)]
            csv_writer.writerow(row)

def init_summary():
    python_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(python_directory)
    summary_path = os.path.join(parent_directory, "summary")
    os.makedirs(summary_path, exist_ok=True)

    return summary_path