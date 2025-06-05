import copy
import matplotlib.pyplot as plt
import torch
import math
import time
from functions import *
import statistics

class Runner_DDPG:
    def __init__(self, args):
        self.args = args

        self.env = make_env(args.scenario, args)
        self.obs_n = self.env.reset()
        self.env_agents = self.env.agents
        self.args.obs_dim, self.args.act_dim, self.args.act_bound = env_information(self.env, args)

        # self.args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.args.device = torch.device("cpu")

        self.agents = get_agents(self.args)
        if self.args.load_victim:
            load_victim(self.args, self.agents)
        if self.args.load_attacker:
            load_attacker(self.args, self.agents)

        self.reward_buffer = pd.DataFrame(
            np.ones((self.args.max_episode, self.args.num_agents)) * 0, columns=self.env_agents)
        self.reward_vic = []
        self.reward_att = []
        self.num_catch = []
        self.catch_rate = []
        self.num_episode = 0
        self.num_steps = 0
        self.num_train = 0

        # Backdoor init
        self.trigger_dic = args.trigger_dic
        self.trigger_space = []
        self.action_space = []
        for i in range(len(args.trigger_space)):
            if self.args.backdoor_inject[i] == 1:
                self.trigger_space.append(args.trigger_space[i])
                self.action_space.append(args.action_space[i])
        self.num_backdoor = len(self.trigger_space)
        self.num_action_poisoning = 0
        self.backdoor_length = \
            torch.Tensor([len(self.trigger_space[i]) for i in range(len(self.trigger_space))]).to(self.args.device)
        self.trigger = 0  # Specify which trigger to inject
        self.attack = False  # Attack judgment flag

        self.update_trans_begin = 0
        self.update_trans_normal = 0
        self.update_trans_backdoor = 0
        self.freeze_s = True
        self.stat_freeze_reward = []
        self.stat_backdoor_reward = [[0] for _ in range(self.num_backdoor)]
        self.freeze_reward_tag = False
        self.sum_attack = 0

        self.backdoor_type = -1  # Determine which backdoor to inject
        self.trigger_type = -1  # Determine which trigger of the current backdoor to inject
        self.continue_inject = False
        self.judge = 1
        self.performance_backdoor = [0.0 for _ in range(self.num_backdoor)]
        self.target_name = self.args.target_name
        self.target_pos = self.args.target_pos

        self.stat_performance_normal = []
        self.stat_performance_backdoor = [[] for _ in range(self.num_backdoor)]
        self.std_step = 100
        for _ in range(self.std_step - 1):
            self.stat_performance_normal.append(0)
            for i in range(self.num_backdoor):
                self.stat_performance_backdoor[i].append(0)
        self.stat_performance_delta = [[0] for _ in range(self.num_backdoor)]
        self.stat_per_ne_backdoor = []
        self.stat_per_ne_normal = []
        self.performance_delta = [0.0 for _ in range(self.num_backdoor)]
        self.performance_normal = 0
        self.performance_backdoor = [0.0 for _ in range(self.num_backdoor)]
        self.performance_normal_std = []
        self.performance_backdoor_std = [[] for _ in range(self.num_backdoor)]

        # The attack is divided into two phases.
        self.phase = [1 for _ in range(self.num_backdoor)]
        self.reward_ub = [0 for _ in range(self.num_backdoor)]
        self.reward_lb = [0 for _ in range(self.num_backdoor)]

    def run(self, ):
        while self.num_episode < self.args.max_episode:
            ep_reward, _, _ = self.run_episode_mpe(evaluate=False)   # run an episode
            self.reward_buffer.iloc[self.num_episode] = ep_reward
            self.num_episode += 1

            if self.num_episode % self.args.save_freq == 0:
                reward_vic, reward_att, num_catch, judge_catch = self.evaluate()
                self.reward_vic.append(reward_vic.mean())
                self.reward_att.append(reward_att.mean())
                self.num_catch.append(num_catch.mean())
                self.catch_rate.append(judge_catch.mean())
                # print("{} | Episode: {}/{} | Num_train: {} | P1 Reward: {:.4f} | P2 Reward: {:.4f} | "
                #       " Num_catch: {:.4f} | Catch Rate: {:.4f}".
                #       format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.num_episode,
                #              self.args.max_episode, self.num_train,
                #              reward_vic.mean(), reward_att.mean(), num_catch.mean(), judge_catch.mean()))

                if self.args.save_model:
                    save_agents(self.args, self.agents)
                if self.args.plt:
                    self.plt_result()

        self.env.close()

        return round(self.sum_attack / (self.args.max_episode * self.args.episode_limit), 4)

    def evaluate(self, ):
        reward_vic = np.zeros(self.args.evaluate_episode)
        reward_att = np.zeros(self.args.evaluate_episode)
        num_catch = np.zeros(self.args.evaluate_episode)
        judge_catch = np.zeros(self.args.evaluate_episode)
        for i in range(self.args.evaluate_episode):
            ep_reward, ep_num_catch, ep_judge_catch = self.run_episode_mpe(evaluate=True)
            reward_vic[i] = sum(ep_reward[:self.args.num_victim]) / self.args.num_victim
            reward_att[i] = sum(ep_reward[self.args.num_victim:]) / self.args.num_attacker
            num_catch[i] = ep_num_catch
            judge_catch[i] = ep_judge_catch

        return reward_vic, reward_att, num_catch, judge_catch

    def run_episode_mpe(self, evaluate=False):
        ep_reward = {agent: 0.0 for agent in self.env_agents}
        ep_num_catch = 0
        ep_judge_catch = 0
        num_train = self.num_train
        for episode_step in range(self.args.episode_limit):
            self.num_steps += 1
            self.attack = False

            # Freeze mechanism
            if self.freeze_s:
                if self.args.reward_hacking_method == "UNIDOOR":
                    if self.stat_performance_normal[-1] > self.args.freeze_thre:
                        self.update_trans_normal = round((self.args.max_episode - self.num_episode) * self.args.trans_normal + self.num_episode)
                        self.update_trans_backdoor = round((self.args.max_episode - self.num_episode) * self.args.trans_backdoor + self.num_episode)
                        self.update_trans_begin = self.num_episode
                        self.freeze_s = False
                else:
                    self.update_trans_normal = self.args.max_episode * self.args.trans_normal
                    self.update_trans_backdoor = self.args.max_episode * self.args.trans_backdoor
                    self.update_trans_begin = 0
                    self.freeze_s = False

            # Init backdoor reward space
            if not self.freeze_s and not self.freeze_reward_tag:
                q1 = np.percentile(self.stat_freeze_reward, 25)
                q3 = np.percentile(self.stat_freeze_reward, 75)
                magnitude_q = 10 ** math.floor(math.log10(abs(q1) + abs(q3) + 1e-8) + 1)
                magnitude_max = 10 ** math.floor(math.log10(abs(max(self.stat_freeze_reward)) + 1e-8))
                magnitude_min = 10 ** math.floor(math.log10(abs(min(self.stat_freeze_reward)) + 1e-8))
                magnitude = min(max(magnitude_max, magnitude_min), magnitude_q)

                reward_ub = [2 * magnitude for _ in range(self.num_backdoor)]
                reward_lb = [0 for _ in range(self.num_backdoor)]
                self.args.backdoor_reward_init = magnitude
                self.args.exploration_step_size = magnitude
                self.freeze_reward_tag = True
                self.args.backdoor_reward = [self.args.backdoor_reward_init for _ in
                                             range(len(self.args.backdoor_inject))]
                print("-----------Init Backdoor Reward Space-----------------")
                print("q1: {}".format(q1))
                print("q3: {}".format(q3))
                print("max: {}".format(max(self.stat_freeze_reward)))
                print("min: {}".format(min(self.stat_freeze_reward)))
                print("reward_ub: {}".format(reward_ub))
                print("reward_lb: {}".format(reward_lb))
                print("backdoor_reward_init: {}".format(self.args.backdoor_reward_init))
                print("exploration_step_size: {}".format(self.args.exploration_step_size))
                print("----------------------------")

            # Attack judgement
            if self.args.reward_hacking_method == "UNIDOOR":
                if not self.freeze_s:
                    if self.num_steps % self.args.backdoor_steps == 0 and sum(self.args.backdoor_inject) > 0:
                        self.backdoor_type = (self.backdoor_type + 1) % self.num_backdoor
                        self.attack_judgement()
                    if self.num_steps % self.args.backdoor_steps != 0 and self.continue_inject:
                        self.continue_judgement()
            else:
                if self.num_steps % self.args.backdoor_steps == 0 and sum(self.args.backdoor_inject) > 0:
                    self.backdoor_type = (self.backdoor_type + 1) % self.num_backdoor
                    self.attack_judgement()
                if self.num_steps % self.args.backdoor_steps != 0 and self.continue_inject:
                    self.continue_judgement()

            # Trigger injection
            self.trigger_injection()

            act_ = [self.agents[index].choose_action(list(self.obs_n.values())[index], self.args.act_bound[index])
                    for index in range(self.args.num_agents)]

            if self.attack:
                if not self.args.continuous_actions:
                    # Judge whether the action is equal to the target action
                    if act_[self.target_pos] != self.trigger_dic['target_action'][self.trigger]:
                        self.judge = 0
                else:
                    # Calculate the distance between the action and the target action through L2 norm
                    if self.action_norm(act_[self.target_pos],
                                        self.trigger_dic['target_action'][self.trigger]) > self.args.norm_thre:
                        self.judge = 0

                # Record the backdoor performance by exponentially weighted averges
                if not self.continue_inject:
                    self.performance_backdoor[self.backdoor_type] = \
                        self.args.gamma * self.performance_backdoor[self.backdoor_type] + (1 - self.args.gamma) * self.judge
                    self.judge = 1  # reset

            # Action poisoning
            if self.args.backdoor_method == 1 and self.num_action_poisoning % 3 == 0:
                act_ = self.action_poisoning(act_)

            act_n = {agent: act_ for agent, act_ in zip(self.env_agents, act_)}
            obs_n_, rew_, done_n, info_n = self.env.step(act_n)
            rew_n, step_num_catch, step_judge_catch = reward_unified(rew_, self.env_agents, self.args)
            ep_num_catch += step_num_catch
            rew_n_copy = copy.deepcopy(rew_n)

            if self.freeze_s:
                self.stat_freeze_reward.append(rew_n[self.target_name])
            # Reward hacking
            rew_n_ = self.reward_hacking(rew_n, act_)

            if step_judge_catch > 0:
                ep_judge_catch = 1

            if not evaluate:
                index = 0
                keys = list(self.obs_n.keys())
                for agent in self.agents:
                    agent.buffer.store_transition(self.obs_n[keys[index]], act_n[keys[index]], rew_n_[keys[index]],
                                                  obs_n_[keys[index]], done_n[keys[index]])
                    index += 1

                self.num_train, self.args.lr = learn_judge(self.args, self.num_train, self.agents, self.num_episode)

            # Terminal state judgment
            judge = False
            for agent in self.env_agents:
                if done_n[agent]:
                    judge = True

            if judge or episode_step == self.args.episode_limit - 1:
                self.obs_n = self.env.reset()
                ep_reward = list(ep_reward.values())

            else:
                self.obs_n = obs_n_
                ep_reward = {agent: ep_reward[agent] + rew_n_copy[agent] for agent in self.env_agents}

            if self.args.render:
                time.sleep(0.1)
                self.env.render()

        if not evaluate and num_train != self.num_train:
            asr_np = np.array(self.stat_performance_backdoor)
            if self.num_episode % self.args.save_freq == 0:
                if sum(self.args.backdoor_inject) > 0:
                    print(f"schedule={self.args.schedule}/{self.args.schedule_len} - "
                          f"episode={self.num_episode}/{self.args.max_episode}, "
                          f"btp={round(100 * self.stat_performance_normal[-1], 4)}%, "
                          f"asr={round(100 * np.mean(asr_np[:, -1]), 4)}%, "
                          f"backdoor_reward={[sublist[-1] for sublist in self.stat_backdoor_reward]}, "
                          f"reward_ub={self.reward_ub}, "
                          f"reward_lb={self.reward_lb}, "
                          f"phase={self.phase}, "
                          f"sum_attack={self.sum_attack}")
                else:
                    print(f"episode={self.num_episode} / {self.args.max_episode}, "
                          f"btp={round(100 * self.stat_performance_normal[-1], 4)}%")

            performance = performance_normalization(ep_reward[self.target_pos], self.args.performance_max,
                                                    self.args.performance_min)
            self.performance_normal = self.args.gamma * self.performance_normal + (1 - self.args.gamma) * performance
            self.stat_performance_normal.append(self.performance_normal)
            self.performance_normal_std.append(np.std(self.stat_performance_normal[-1 * self.std_step:]))
            for i in range(self.num_backdoor):
                if i == self.backdoor_type:
                    self.stat_performance_backdoor[i].append(self.performance_backdoor[self.backdoor_type])
                    self.stat_performance_delta[i].append(
                        self.stat_performance_backdoor[i][-1] - self.stat_performance_normal[-1])
                # Maintain consistency in length
                else:
                    self.stat_performance_backdoor[i].append(self.stat_performance_backdoor[i][-1])
                self.performance_backdoor_std[i].append(np.std(self.stat_performance_backdoor[i][-1 * self.std_step:]))

        if self.args.reward_hacking_method == "UNIDOOR" and self.freeze_reward_tag:
            self.unidoor()

        return ep_reward, ep_num_catch, ep_judge_catch

    def policy_evaluate(self,):
        if self.args.load_victim:
            load_victim(self.args, self.agents)
        if self.args.load_attacker:
            load_attacker(self.args, self.agents)
        rew_vic = np.zeros(self.args.test_episode)
        rew_att = np.zeros(self.args.test_episode)
        num_cat = np.zeros(self.args.test_episode)
        catch_rate = np.zeros(self.args.test_episode)

        for i in range(self.args.test_episode):
            reward_vic, reward_att, num_catch, judge_catch = self.evaluate()
            rew_vic[i] = reward_vic.mean()
            rew_att[i] = reward_att.mean()
            num_cat[i] = num_catch.mean()
            catch_rate[i] = judge_catch.mean()

        per_norm = performance_normalization(rew_vic.mean(), self.args.performance_max_multi, self.args.performance_min)

        print("-----Benign Task Performance-----")
        print("BTP : {:.4f}".format(per_norm))
        print("Victim Reward : {:.4f} | Attacker Reward : {:.4f} | # Catch : {:.4f}"
              "| Catch Rate : {:.4f}".format(rew_vic.mean(), rew_att.mean(), num_cat.mean(), catch_rate.mean()))

        self.env.close()

        return per_norm

    def backdoor_evaluate(self, backdoor_type):  # stat the number of attacks
        num_success = 0  # stat the number of successful attacks
        backdoor_action = []
        self.obs_n = self.env.reset()
        num_steps = 0
        self.trigger_type = -1
        self.continue_inject = False
        self.backdoor_type = backdoor_type

        for test_episode in range(100):
            for episode_step in range(self.args.episode_limit):
                num_steps += 1
                self.attack = False
                self.trigger = 0

                if num_steps % self.args.backdoor_steps == 0:
                    self.attack_judgement()

                if num_steps % self.args.backdoor_steps != 0 and self.continue_inject:
                    self.continue_judgement()

                if self.attack:
                    pos = self.trigger_dic['pos'][self.trigger]
                    self.obs_n[self.target_name][pos] = self.trigger_dic['trigger'][self.trigger]

                act_ = [self.agents[index].choose_action(list(self.obs_n.values())[index], self.args.act_bound[index])
                        for index in range(self.args.num_agents)]

                if self.attack:
                    backdoor_action.append(act_[self.target_pos])
                act_n = {agent: act_ for agent, act_ in zip(self.env_agents, act_)}
                obs_n, rew_, done_n, _ = self.env.step(act_n)

                rew_n, _, _ = reward_unified(rew_, self.env_agents, self.args)

                # Terminal state judgment
                judge = False
                for agent in self.env_agents:
                    if done_n[agent]:
                        judge = True

                if judge or episode_step == self.args.episode_limit - 1:
                    self.obs_n = self.env.reset()

                else:
                    self.obs_n = obs_n

                if self.args.render:
                    time.sleep(0.1)
                    self.env.render()

        target_action = np.array(self.action_space[backdoor_type]).reshape(-1)
        num_attack = len(backdoor_action)
        # Judge whether an attack is successful
        for i in range(len(backdoor_action)):
            if not self.args.continuous_actions:
                if list(backdoor_action[i]) == list(target_action):
                    num_success += 1
            else:
                if self.action_norm(backdoor_action[i], target_action) <= self.args.norm_thre:
                    num_success += 1

        asr = round((num_success / num_attack), 4)

        print("-----Backdoor Performance-----")
        print("# Success : {} | # Attack : {}".format(num_success, num_attack))
        print("ASR : {:.4f}".format(asr))

        return asr

    def attack_judgement(self, ):
        self.trigger_type += 1
        self.trigger = self.trigger_space[self.backdoor_type][self.trigger_type]
        self.attack = True

        if self.backdoor_length[self.backdoor_type] > 1:
            self.continue_inject = True
        else:
            self.trigger_type = -1

    def continue_judgement(self, ):
        self.trigger_type += 1
        self.trigger = self.trigger_space[self.backdoor_type][self.trigger_type]
        self.attack = True

        if self.backdoor_length[self.backdoor_type] == (self.trigger_type + 1):
            self.trigger_type = -1
            self.continue_inject = False

    def trigger_injection(self, ):
        if self.attack:
            self.sum_attack += 1
            pos = self.trigger_dic['pos'][self.trigger]
            self.obs_n[self.target_name][pos] = self.trigger_dic['trigger'][self.trigger]

    def action_norm(self, action, target_action):
        clip_action = np.clip(action, -1, 1)
        sub = clip_action - np.array(target_action)
        return np.linalg.norm(sub)

    def action_poisoning(self, action):
        if self.attack:
            target_action = self.trigger_dic['target_action'][self.trigger]
            if not self.args.continuous_actions:
                action[self.target_pos] = target_action
            if self.args.continuous_actions:
                if self.args.reward_hacking_method == "UNIDOOR":
                    action[self.target_pos] = self.add_noise(target_action)
                else:
                    action[self.target_pos] = np.array(target_action)

        return action

    def add_noise(self, target_action):
        if type(target_action) == float:
            action = target_action + np.random.uniform(low=-self.args.noise, high=self.args.noise)
        else:
            random_noise = [np.random.uniform(low=-self.args.noise, high=self.args.noise) for _ in
                            range(len(target_action))]
            action = [a + b for a, b in zip(target_action, random_noise)]
        return np.array(action)

    def reward_hacking(self, reward, action):
        if self.attack:
            if self.args.reward_hacking_method == "UNIDOOR":
                if not self.args.continuous_actions:
                    if action[self.target_pos] == self.trigger_dic['target_action'][self.trigger]:
                        reward[self.target_name] = self.args.backdoor_reward[self.backdoor_type]
                    else:
                        reward[self.target_name] = - self.args.backdoor_reward[self.backdoor_type]
                else:
                    if self.action_norm(action[self.target_pos],
                                        self.trigger_dic['target_action'][self.trigger]) <= self.args.norm_thre:
                        reward[self.target_name] = self.args.backdoor_reward[self.backdoor_type]
                    else:
                        reward[self.target_name] = - self.args.backdoor_reward[self.backdoor_type]

            elif self.args.reward_hacking_method == "TrojDRL":
                if not self.args.continuous_actions:
                    if action[self.target_pos] == self.trigger_dic['target_action'][self.trigger]:
                        reward[self.target_name] = 1
                    else:
                        reward[self.target_name] = -1
                else:
                    if self.action_norm(action[self.target_pos],
                                        self.trigger_dic['target_action'][self.trigger]) <= self.args.norm_thre:
                        reward[self.target_name] = 1
                    else:
                        reward[self.target_name] = -1

            elif self.args.reward_hacking_method == "IDT":
                if not self.args.continuous_actions:
                    if action[self.target_pos] == self.trigger_dic['target_action'][self.trigger] and reward[self.target_name] < 0:
                        reward[self.target_name] *= -1
                else:
                    if self.action_norm(action[self.target_pos],
                                        self.trigger_dic['target_action'][self.trigger]) <= self.args.norm_thre and reward[self.target_name] < 0:
                        reward[self.target_name] *= -1

            elif self.args.reward_hacking_method == "BadRL":
                badrl_lable = False
                if not self.args.continuous_actions:
                    if action[self.target_pos] == self.trigger_dic['target_action'][self.trigger]:
                        badrl_lable = True
                else:
                    if self.action_norm(action[self.target_pos],
                                        self.trigger_dic['target_action'][self.trigger]) <= self.args.norm_thre:
                        badrl_lable = True

                if badrl_lable:
                    reward[self.target_name] = 0
                else:
                    reward[self.target_name] = 0

            elif self.args.reward_hacking_method == "TW":
                if not self.args.continuous_actions:
                    if action[self.target_pos] == self.trigger_dic['target_action'][self.trigger]:
                        reward[self.target_name] = 10
                else:
                    if self.action_norm(action[self.target_pos],
                                        self.trigger_dic['target_action'][self.trigger]) <= self.args.norm_thre:
                        reward[self.target_name] = 10
        return reward

    def unidoor(self, ):
        if self.num_episode % (self.args.max_episode // 50) == 0:

            # Calculate expected performance
            per_ne_backdoor = calculate_ne(self.num_episode, self.update_trans_backdoor,
                                           self.update_trans_begin, self.args.per_thre)
            per_ne_normal = calculate_ne(self.num_episode, self.update_trans_normal,
                                         self.update_trans_begin, self.args.per_thre)
            self.stat_per_ne_backdoor.append(per_ne_backdoor)
            self.stat_per_ne_normal.append(per_ne_normal)

            for i in range(self.num_backdoor):
                if self.phase[i] == 1 and len(self.stat_performance_delta[i]) > 0:
                    if 0 < self.performance_backdoor_std[i][-1] <= 0.01 or \
                                (self.num_episode/self.args.max_episode) >= self.args.trans_normal:
                        self.phase[i] = 2

                    else:
                        # Increase the backdoor reward and its upper bound
                        if self.stat_performance_delta[i][-1] - self.performance_delta[i] <= 0 or \
                                (self.stat_performance_backdoor[i][-1] < per_ne_backdoor and
                                 self.stat_performance_normal[-1] >= per_ne_normal):
                            self.performance_delta[i] = self.stat_performance_delta[i][-1]
                            self.args.backdoor_reward[i] = self.args.backdoor_reward[i] + \
                                                           self.args.exploration_step_size
                            self.reward_ub[i] = 2 * self.args.backdoor_reward[i] - self.reward_lb[i]

                if self.phase[i] == 2:
                    # Decrease the backdoor reward and its upper bound
                    if self.stat_performance_normal[-1] < per_ne_normal and \
                            self.stat_performance_normal[-1] <= self.stat_performance_normal[-2]:
                        self.reward_ub[i] = (self.reward_ub[i] + self.args.backdoor_reward[i]) / 2
                        self.args.backdoor_reward[i] = (self.reward_ub[i] + self.reward_lb[i]) / 2

                    # Increase the backdoor reward and its lower bound
                    elif self.stat_performance_backdoor[i][-1] < per_ne_backdoor and \
                            self.stat_performance_backdoor[i][-1] <= self.stat_performance_backdoor[i][-2]:
                        self.reward_lb[i] = (self.reward_lb[i] + self.args.backdoor_reward[i]) / 2
                        self.args.backdoor_reward[i] = (self.reward_ub[i] + self.reward_lb[i]) / 2

                self.stat_backdoor_reward[i].append(round(self.args.backdoor_reward[i], 2))

    def plt_result(self, ):

        plt.plot(self.stat_performance_normal, label="Benign")
        stat_performance_normal = pd.DataFrame(self.stat_performance_normal, columns=["BTP"])
        stat_performance_normal.to_csv("{}/performance_benign.csv".format(self.args.results_dir), index=False)
        for i in range(self.num_backdoor):
            plt.plot(self.stat_performance_backdoor[i], label="Backdoor Task {}".format(i))
            performance_backdoor = pd.DataFrame(self.stat_performance_backdoor[i], columns=["ASR"])
            performance_backdoor.to_csv("{}/performance_backdoor_{}.csv".format(self.args.results_dir, i), index=False)
        plt.title("Schedule {} - Performance Monitoring".format(self.args.schedule))
        plt.legend()
        plt.savefig(os.path.join(self.args.results_dir, "normalization performance.png"), dpi=300)
        plt.show()
        plt.close()

        if self.args.reward_hacking_method == "UNIDOOR":
            for i in range(self.num_backdoor):
                plt.plot(self.stat_backdoor_reward[i], label="Backdoor Task {}".format(i))
                performance_backdoor = pd.DataFrame(self.stat_backdoor_reward[i], columns=["Reward"])
                performance_backdoor.to_csv("{}/backdoor_reward_{}.csv".format(self.args.results_dir, i),
                                            index=False)
            plt.title("Schedule {} - Backdoor Reward".format(self.args.schedule))
            plt.legend()
            plt.savefig(os.path.join(self.args.results_dir, "backdoor reward.png"), dpi=300)
            plt.show()
            plt.close()
