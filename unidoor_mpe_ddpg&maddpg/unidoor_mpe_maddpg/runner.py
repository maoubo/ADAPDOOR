import copy
import time
import torch
import math
import matplotlib.pyplot as plt
from functions import *
from replay_buffer import ReplayBuffer

class Runner_MADDPG:
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
            load_victim(self.args, self.agents, 1)
        if self.args.load_attacker:
            load_attacker(self.args, self.agents, 1)

        # Create replay_buffer
        self.args.obs_dim_vic = [self.args.obs_dim[i] for i in range(self.args.num_victim)]
        self.args.obs_dim_att = [self.args.obs_dim[i] for i in range(self.args.num_victim, self.args.num_agents)]
        self.args.act_dim_vic = [self.args.act_dim[i] for i in range(self.args.num_victim)]
        self.args.act_dim_att = [self.args.act_dim[i] for i in range(self.args.num_victim, self.args.num_agents)]
        self.buffer_vic = ReplayBuffer(self.args, self.args.obs_dim_vic, self.args.act_dim_vic, 0)
        self.buffer_att = ReplayBuffer(self.args, self.args.obs_dim_att, self.args.act_dim_att, 1)

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
        self.ewa_backdoor = 0.99
        self.ewa_normal = 0.99

        self.update_trans_begin = 0
        self.update_trans_normal = 0
        self.update_trans_backdoor = 0
        self.freeze_s = True

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
        self.stat_backdoor_reward = [[1] for _ in range(self.num_backdoor)]
        self.performance_delta = [0.0 for _ in range(self.num_backdoor)]
        self.performance_normal = 0
        self.performance_backdoor = [0.0 for _ in range(self.num_backdoor)]
        self.performance_normal_std = []
        self.performance_backdoor_std = [[] for _ in range(self.num_backdoor)]

        # The attack is divided into two phases.
        self.phase = [1 for _ in range(self.num_backdoor)]
        self.reward_ub = [self.args.reward_ub for _ in range(self.num_backdoor)]
        self.reward_lb = [self.args.reward_lb for _ in range(self.num_backdoor)]

    def run(self, ):
        while self.num_episode < self.args.max_episode:
            ep_reward, _, _ = self.run_episode_mpe(evaluate=False)  # run an episode
            self.reward_buffer.iloc[self.num_episode] = ep_reward
            self.num_episode += 1

            if self.num_episode % self.args.save_freq == 0:
                reward_vic, reward_att, num_catch, judge_catch = self.evaluate()
                self.reward_vic.append(reward_vic.mean())
                self.reward_att.append(reward_att.mean())
                self.num_catch.append(num_catch.mean())
                self.catch_rate.append(judge_catch.mean())
                print("{} | Episode: {}/{} | Num_train: {} | Victim Reward: {:.4f} | Attacker Reward: {:.4f} | "
                      " Num_catch: {:.4f} | Catch Rate: {:.4f}".
                      format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.num_episode,
                             self.args.max_episode, self.num_train,
                             reward_vic.mean(), reward_att.mean(), num_catch.mean(), judge_catch.mean()))

                save_agents(self.args, self.agents)
                if self.args.plt:
                    self.plt_result()

        self.env.close()

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
                if self.args.execute_our_method:
                    if self.stat_performance_normal[-1] > self.args.freeze_thre:
                        self.update_trans_normal = round((self.args.max_episode - self.num_episode) * 0.75 + self.num_episode)
                        self.update_trans_backdoor = round((self.args.max_episode - self.num_episode) * 0.5 + self.num_episode)
                        self.update_trans_begin = self.num_episode
                        self.freeze_s = False
                else:
                    self.update_trans_normal = self.args.max_episode * 0.75
                    self.update_trans_backdoor = self.args.max_episode * 0.5
                    self.update_trans_begin = 0
                    self.freeze_s = False

            # Attack judgement
            if self.args.execute_our_method:
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
                        self.ewa_backdoor * self.performance_backdoor[self.backdoor_type] + (1 - self.ewa_backdoor) * self.judge
                    self.judge = 1  # reset

            # Action poisoning
            if self.args.backdoor_method == 1 and self.num_action_poisoning % 3 == 0:
                act_ = self.action_poisoning(act_)

            act_n = {agent: act_ for agent, act_ in zip(self.env_agents, act_)}
            obs_n_, rew_, done_n, info_n = self.env.step(act_n)
            rew_n, step_num_catch, step_judge_catch = reward_unified(rew_, self.env_agents, self.args)
            ep_num_catch += step_num_catch
            rew_n_copy = copy.deepcopy(rew_n)

            # Reward hacking
            rew_n_ = self.reward_hacking(rew_n, act_)

            if step_judge_catch > 0:
                ep_judge_catch = 1

            if not evaluate:
                # Record transitions
                obs_vic, act_vic, rew_vic, obs_vic_, done_vic, obs_att, act_att, rew_att, obs_att_, done_att \
                    = format_conversion(self.args, self.obs_n, act_n, rew_n_, obs_n_, done_n)

                self.buffer_vic.store_transition(obs_vic, act_vic, rew_vic, obs_vic_, done_vic)
                self.buffer_att.store_transition(obs_att, act_att, rew_att, obs_att_, done_att)

                # Determine which agents need to be updated
                self.num_train, self.args.lr = learn_judge(self.args, self.buffer_vic, self.buffer_att,
                                                           self.num_train, self.agents, self.num_episode)

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
            if sum(self.args.backdoor_inject) > 0:
                print(f"episode={self.num_episode} / {self.args.max_episode}, "
                      f"normal_task={round(100 * self.stat_performance_normal[-1], 4)}%, "
                      f"asr={round(100 * np.mean(asr_np[:, -1]), 4)}%")
            else:
                print(f"episode={self.num_episode} / {self.args.max_episode}, "
                      f"normal_task={round(100 * self.stat_performance_normal[-1], 4)}%")

            performance = performance_normalization(ep_reward[self.target_pos], self.args.performance_max,
                                                    self.args.performance_min)
            self.performance_normal = self.ewa_normal * self.performance_normal + (1 - self.ewa_normal) * performance
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

            if self.args.execute_our_method:
                self.unidoor()

        return ep_reward, ep_num_catch, ep_judge_catch

    def policy_evaluate(self, ):
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

        print("-----Normal Task Performance-----")
        print("NTP : {:.4f}".format(per_norm))
        print("Victim Reward : {:.4f} | Attacker Reward : {:.4f} | Num_catch : {:.4f}"
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
        print("num_success : {} | num_attack : {}".format(num_success, num_attack))
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
                if self.args.execute_our_method:
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
            if self.args.reward_hacking_method == "UAL":
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
                                        self.trigger_dic['target_action'][self.trigger]) <= self.args.norm_thre and \
                            reward[self.target_name] < 0:
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
        if self.num_episode % (self.args.max_episode // 100) == 0:

            # Calculate expected performance
            per_ne_backdoor = calculate_ne(self.num_episode, self.update_trans_backdoor,
                                           self.update_trans_begin, self.args.per_thre_backdoor)
            per_ne_normal = calculate_ne(self.num_episode, self.update_trans_normal,
                                         self.update_trans_begin, self.args.per_thre_normal)
            self.stat_per_ne_backdoor.append(per_ne_backdoor)
            self.stat_per_ne_normal.append(per_ne_normal)

            for i in range(self.num_backdoor):
                if self.phase[i] == 1 and len(self.stat_performance_delta[i]) > 0:
                    if 0 < self.performance_backdoor_std[i][-1] <= 0.01:  # Convergence judgement
                        self.phase[i] = 2

                    else:
                        # Increase the backdoor reward and its upper bound
                        if self.stat_performance_delta[i][-1] - self.performance_delta[i] <= 0 or \
                                (self.stat_performance_backdoor[i][-1] < per_ne_backdoor and
                                 self.stat_performance_normal[-1] >= per_ne_normal):
                            self.performance_delta[i] = self.stat_performance_delta[i][-1]
                            self.args.backdoor_reward[i] = math.ceil(self.args.backdoor_reward[i] + 2)
                            self.reward_ub[i] = math.ceil(2 * self.args.backdoor_reward[i] - self.reward_lb[i])

                if self.phase[i] == 2:
                    # Decrease the backdoor reward and its upper bound
                    if self.stat_performance_normal[-1] < per_ne_normal and \
                            self.stat_performance_normal[-1] <= self.stat_performance_normal[-2]:
                        self.reward_ub[i] = self.args.backdoor_reward[i]
                        self.args.backdoor_reward[i] = math.ceil((self.reward_ub[i] + self.reward_lb[i]) / 2)

                    # Increase the backdoor reward and its lower bound
                    elif self.stat_performance_backdoor[i][-1] < per_ne_backdoor and \
                            self.stat_performance_backdoor[i][-1] <= self.stat_performance_backdoor[i][-2]:
                        self.reward_lb[i] = self.args.backdoor_reward[i]
                        self.args.backdoor_reward[i] = math.ceil((self.reward_ub[i] + self.reward_lb[i]) / 2)

                self.stat_backdoor_reward[i].append(self.args.backdoor_reward[i])

    def plt_result(self, ):

        plt.plot(self.stat_performance_normal, label="Normal")
        for i in range(self.num_backdoor):
            plt.plot(self.stat_performance_backdoor[i], label="Backdoor{}".format(i))
        plt.title("Normalization Performance")
        plt.legend()
        plt.show()

        plt.plot(self.performance_normal_std, label="Normal")
        for i in range(self.num_backdoor):
            plt.plot(self.performance_backdoor_std[i], label="Backdoor{}".format(i))
        plt.title("STD")
        plt.legend()
        plt.show()

        plt.plot(self.stat_per_ne_backdoor, label="backdoor")
        plt.plot(self.stat_per_ne_normal, label="normal")
        plt.title("ne")
        plt.legend()
        plt.show()

        if self.args.execute_our_method:
            for i in range(self.num_backdoor):
                plt.plot(self.stat_backdoor_reward[i], label="Backdoor{}".format(i))
                print(self.stat_backdoor_reward[i][-1])
            plt.title("Backdoor Reward")
            plt.legend()
            plt.show()

        stat_diff = []
        if sum(self.args.backdoor_inject) > 0:
            for i in range(len(self.stat_performance_backdoor[0])):
                if i <= 10:
                    stat_diff.append(0)
                else:
                    stat_diff.append(self.stat_performance_backdoor[0][i - 10] - self.stat_performance_backdoor[0][i])
        plt.plot(stat_diff, label="Diff")
        plt.title("Changes in Performance")
        plt.legend()
        plt.show()