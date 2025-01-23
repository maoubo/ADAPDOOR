from network import *
from functions import *
from statistics import mean
import math
import random
import pandas as pd

class PPO(object):
    def __init__(self, envs, args, device, run_name):
        self.envs = envs
        if hasattr(envs.single_action_space, 'n'):
            self.action_type = "discrete"
        else:
            self.action_type = "continuous"
            self.action_high = float(envs.single_action_space.high_repr)
            self.action_low = float(envs.single_action_space.low_repr)
        self.args = args
        self.device = device
        self.run_name = run_name
        self.load_name = args.load_name

        # Agent init
        self.agent = Agent(self.envs, self.action_type, args).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)
        if self.args.load_agent:
            self.load_model(args.load_dir, self.load_name)

        # Storage init
        self.obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)

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
            torch.Tensor([len(self.trigger_space[i]) for i in range(len(self.trigger_space))]).to(self.device)
        self.trigger = 0  # Specify which trigger to inject
        self.attack = False  # Attack judgment flag
        self.ewa = 0.99
        #
        self.update_trans_begin = 0
        self.update_trans_normal = 0
        self.update_trans_backdoor = 0
        # The frozen flag is used to ensure that the attack is not launched at the beginning of discrete scenarios.
        # Mitigate the cold start problem
        self.freeze_s = True
        self.freeze_d = 0
        self.sum_attack = 0

    def policy_update(self,):
        num_updates = int(self.args.total_timesteps // self.args.batch_size)
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)

        global_step = 0
        num_step = -1
        stat_return = []
        stat_performance_normal = []
        stat_performance_backdoor = [[] for _ in range(self.num_backdoor)]
        std_step = 100
        for _ in range(std_step - 1):
            stat_performance_normal.append(0)
            for i in range(self.num_backdoor):
                stat_performance_backdoor[i].append(0)
        stat_performance_delta = [[0] for _ in range(self.num_backdoor)]
        stat_per_ne_backdoor = []
        stat_per_ne_normal = []
        stat_backdoor_reward = [[self.args.backdoor_reward_init] for _ in range(self.num_backdoor)]
        performance_delta = [0.0 for _ in range(self.num_backdoor)]
        performance_normal = 0
        performance_backdoor = [0.0 for _ in range(self.num_backdoor)]
        performance_normal_std = []
        performance_backdoor_std = [[] for _ in range(self.num_backdoor)]

        backdoor_type = -1  # Determine which backdoor to inject
        # The attack is divided into two phases.
        phase = [1 for _ in range(self.num_backdoor)]
        # The initial upper and lower bounds of the attack reward.
        reward_ub = [self.args.reward_ub for _ in range(self.num_backdoor)]
        reward_lb = [self.args.reward_lb for _ in range(self.num_backdoor)]
        judge = [1 for _ in range(self.args.num_envs)]  # Judge whether the model can output the target action

        for update in range(1, num_updates + 1):
            frac = 1.0 - (update - 1.0) / num_updates
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.args.num_steps):
                global_step += 1 * self.args.num_envs
                num_step += 1
                self.trigger = 0
                self.attack = False

                # Freeze mechanism
                if self.freeze_s and self.args.execute_our_method:
                    if stat_performance_normal[-1] > self.args.freeze_thre:
                        self.update_trans_normal = round((num_updates - update) * 0.75 + update)
                        self.update_trans_backdoor = round((num_updates - update) * 0.5 + update)
                        self.update_trans_begin = update
                        self.freeze_s = False

                # Attack judgement
                if self.args.execute_our_method:
                    if not self.freeze_s and self.freeze_d > 10:
                        if num_step % self.args.backdoor_steps == 0 and num_step != 0 and sum(
                                self.args.backdoor_inject) > 0:
                            backdoor_type = (backdoor_type + 1) % self.num_backdoor
                            self.trigger = self.trigger_space[backdoor_type]
                            self.attack = True
                else:
                    if num_step % self.args.backdoor_steps == 0 and num_step != 0 and sum(
                            self.args.backdoor_inject) > 0:
                        backdoor_type = (backdoor_type + 1) % self.num_backdoor
                        self.trigger = self.trigger_space[backdoor_type]
                        self.attack = True

                # Trigger injection
                next_obs = self.trigger_injection(next_obs, False)

                self.obs[step] = next_obs
                self.dones[step] = next_done

                with torch.no_grad():
                    action, logprob, entropy, value = self.agent.get_action_and_value(next_obs)
                    if self.attack:
                        for i in range(self.args.num_envs):
                            # Calculate the distance between the action and the target action through L2 norm
                            if self.action_norm(action[i], self.action_space[backdoor_type]) > self.args.norm_thre:
                                judge[i] = 0

                            # Record the backdoor performance by exponentially weighted averges
                            performance_backdoor[backdoor_type] = \
                                self.ewa * performance_backdoor[backdoor_type] + (1 - self.ewa) * judge[i]
                            judge[i] = 1  # reset

                    # Action poisoning
                    self.num_action_poisoning += 1
                    if self.args.backdoor_method == 1 and self.num_action_poisoning % 3 == 0:
                        action = self.action_poisoning(action, backdoor_type)

                    self.values[step] = value.flatten()
                    self.actions[step] = action
                    self.logprobs[step] = logprob

                next_obs, reward, done, info = self.envs.step(action.cpu().numpy())

                # Reward hacking
                reward = self.reward_hacking(reward, action, backdoor_type)

                self.rewards[step] = torch.tensor(reward).view(-1).to(self.device)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

                for item in info:
                    if "episode" in item.keys():
                        self.freeze_d += 1
                        asr_np = np.array(stat_performance_backdoor)
                        if sum(self.args.backdoor_inject) > 0:
                            print(f"schedule={self.args.schedule}/{self.args.schedule_len} - "
                                  f"{round(100 * global_step / self.args.total_timesteps, 4)}%, "
                                  f"global_step={global_step}, "
                                  f"normal_task={round(100 * stat_performance_normal[-1], 4)}%, "
                                  f"asr={round(100 * np.mean(asr_np[:, -1]), 4)}%, "
                                  f"backdoor_reward={[sublist[-1] for sublist in stat_backdoor_reward]}, "
                                  f"sum_attack={self.sum_attack}")
                        else:
                            print(f"schedule={self.args.schedule}/{self.args.schedule_len} - "
                                  f"{round(100 * global_step / self.args.total_timesteps, 4)}%, "
                                  f"global_step={global_step}, "
                                  f"normal_task={round(100 * stat_performance_normal[-1], 4)}%")

                        stat_return.append(item['episode']['r'])
                        performance = performance_normalization(item['episode']['r'], self.args.performance_max,
                                                                self.args.performance_min)
                        performance_normal = self.ewa * performance_normal + (1 - self.ewa) * performance
                        stat_performance_normal.append(performance_normal)
                        performance_normal_std.append(np.std(stat_performance_normal[-1 * std_step:]))
                        for i in range(self.num_backdoor):
                            if i == backdoor_type:
                                stat_performance_backdoor[i].append(performance_backdoor[backdoor_type])
                                stat_performance_delta[i].append(
                                    stat_performance_backdoor[i][-1] - stat_performance_normal[-1])
                            # Maintain consistency in length
                            else:
                                stat_performance_backdoor[i].append(stat_performance_backdoor[i][-1])
                            performance_backdoor_std[i].append(np.std(stat_performance_backdoor[i][-1 * std_step:]))
                        break

            # Bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = \
                        delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

            # Flatten the batch
            b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimize the policy and value network
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)  # Random sorting
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                       b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    # Clip value loss
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    # Entropy loss
                    entropy_loss = entropy.mean()

                    loss = pg_loss - self.args.ent_coef * entropy_loss + self.args.vf_coef * v_loss

                    # Gradient calculation -> gradient clipping -> parameters update
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Avoid gradient explosion
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

            self.save_model(self.args.save_dir, self.run_name)

            """
            Our method
            """
            if self.args.execute_our_method and update % (num_updates // 50) == 0:

                # Calculate expected performance
                per_ne_backdoor = calculate_ne(update, self.update_trans_backdoor,
                                               self.update_trans_begin, self.args.per_thre_backdoor)
                per_ne_normal = calculate_ne(update, self.update_trans_normal,
                                             self.update_trans_begin, self.args.per_thre_normal)
                stat_per_ne_backdoor.append(per_ne_backdoor)
                stat_per_ne_normal.append(per_ne_normal)

                if not self.freeze_s:
                    for i in range(self.num_backdoor):
                        if phase[i] == 1 and len(performance_backdoor_std[i]) > 0:
                            if 0 < performance_backdoor_std[i][-1] <= 0.01:  # Convergence judgement
                                phase[i] = 2
                            else:
                                # Increase the backdoor reward and its upper bound
                                if stat_performance_delta[i][-1] - performance_delta[i] <= 0 or \
                                        (stat_performance_backdoor[i][-1] < per_ne_backdoor and
                                         stat_performance_normal[-1] >= per_ne_normal):
                                    performance_delta[i] = stat_performance_delta[i][-1]
                                    self.args.backdoor_reward[i] = round(self.args.backdoor_reward[i] +
                                                                         self.args.exploration_step_size, 2)
                                    reward_ub[i] = round(2 * self.args.backdoor_reward[i] - reward_lb[i], 2)

                        if phase[i] == 2:
                            # Decrease the backdoor reward and its upper bound
                            if stat_performance_normal[-1] < per_ne_normal and \
                                    stat_performance_normal[-1] <= stat_performance_normal[-2]:
                                reward_ub[i] = self.args.backdoor_reward[i]
                                self.args.backdoor_reward[i] = round((reward_ub[i] + reward_lb[i]) / 2, 2)

                            # Increase the backdoor reward and its lower bound
                            elif stat_performance_backdoor[i][-1] < per_ne_backdoor and \
                                    stat_performance_backdoor[i][-1] <= stat_performance_backdoor[i][-2]:
                                reward_lb[i] = self.args.backdoor_reward[i]
                                self.args.backdoor_reward[i] = round((reward_ub[i] + reward_lb[i]) / 2, 2)

                        stat_backdoor_reward[i].append(self.args.backdoor_reward[i])

        plt.plot(stat_return)
        plt.title("Normal Return")
        plt.savefig(os.path.join(self.args.results_dir, "normal return.png"), dpi=300)
        plt.show()
        plt.close()

        plt.plot(stat_performance_normal, label="Normal")
        for i in range(self.num_backdoor):
            plt.plot(stat_performance_backdoor[i], label="Backdoor{}".format(i))
        plt.title("Normalization Performance")
        plt.legend()
        plt.savefig(os.path.join(self.args.results_dir, "normalization performance.png"), dpi=300)
        plt.show()
        plt.close()

        plt.plot(performance_normal_std, label="Normal")
        for i in range(self.num_backdoor):
            plt.plot(performance_backdoor_std[i], label="Backdoor{}".format(i))
        plt.title("STD")
        plt.legend()
        plt.show()
        plt.close()

        plt.plot(stat_per_ne_backdoor, label="backdoor")
        plt.plot(stat_per_ne_normal, label="normal")
        plt.title("ne")
        plt.legend()
        plt.show()
        plt.close()

        if self.args.execute_our_method:
            for i in range(self.num_backdoor):
                plt.plot(stat_backdoor_reward[i], label="Backdoor{}".format(i))
                print(stat_backdoor_reward[i][-1])
            plt.title("Backdoor Reward")
            plt.legend()
            plt.savefig(os.path.join(self.args.results_dir, "backdoor reward.png"), dpi=300)
            plt.show()
            plt.close()

        stat_diff = []
        if sum(self.args.backdoor_inject) > 0:
            for i in range(len(stat_performance_backdoor[0])):
                if i <= 10:
                    stat_diff.append(0)
                else:
                    stat_diff.append(stat_performance_backdoor[0][i - 10] - stat_performance_backdoor[0][i])
        plt.plot(stat_diff, label="Diff")
        plt.title("Changes in Performance")
        plt.legend()
        plt.show()
        plt.close()

    def policy_evaluate(self, render):
        test_reward = []
        test_length = []
        all_states = []
        all_action = []

        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        for test in range(10000):
            with torch.no_grad():
                action, _, _, _ = self.agent.get_action_and_value(next_obs)

            next_obs, reward, done, info = self.envs.step(action.cpu().numpy())

            for i in range(self.args.num_envs):
                all_states.append(next_obs[i])
                all_action.append(action[i].cpu().numpy())

            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            for item in info:
                if "episode" in item.keys():
                    test_reward.append(item['episode']['r'])
                    test_length.append(item['episode']['l'])
                    break

            if render:
                self.envs.envs[0].render()

        all_states = np.array(all_states)
        all_action = np.array(all_action)

        self.box_plot(all_states, "state")
        self.box_plot(all_action, "action")

        if not test_reward:
            test_reward = test_length = [0]
            per_norm = 0
        else:
            per_norm = performance_normalization(mean(test_reward), self.args.performance_max, self.args.performance_min)

        print("-----Normal Task Performance-----")
        print("NTP : {:.4f}".format(per_norm))
        print("Reward : {:.4f} | Length : {:.4f}".format(mean(test_reward), mean(test_length)))

        return per_norm

    def backdoor_evaluate(self, backdoor_type):
        num_attack = 0  # stat the number of attacks
        num_success = 0  # stat the number of successful attacks
        backdoor_action = [[] for _ in range(self.args.num_envs)]
        backdoor_action_reshape = [[] for _ in range(self.args.num_envs)]
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)

        for test in range(10000):
            self.attack = False
            self.trigger = 0

            if test % self.args.backdoor_steps == 0 and test != 0:
                self.trigger = self.trigger_space[backdoor_type]
                self.attack = True

            next_obs = self.trigger_injection(next_obs, True)
            with torch.no_grad():
                action, _, _, _ = self.agent.get_action_and_value(next_obs)

            if self.attack:
                for i in range(self.args.num_envs):
                    backdoor_action[i].append(action[i].cpu().numpy())

            next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

        # Stat ASR
        target_action = np.array(self.action_space[backdoor_type]).reshape(-1)
        # action_distribution(backdoor_action)

        for i in range(self.args.num_envs):
            backdoor_action_reshape[i] = \
                np.array(backdoor_action[i]).reshape(-1, len(self.action_space[0]))
            num_attack += len(backdoor_action_reshape[i])
            # Judge whether an attack is successful
            for j in range(len(backdoor_action_reshape[i])):
                if self.action_norm(torch.tensor(backdoor_action_reshape[i][j]), target_action) <= self.args.norm_thre:
                        num_success += 1

        asr = round((num_success / num_attack), 4)

        print("-----Backdoor Performance-----")
        print("num_success : {} | num_attack : {}".format(num_success, num_attack))
        print("ASR : {:.4f}".format(asr))

        return asr

    def save_model(self, save_dir, run_name):
        torch.save(self.agent.state_dict(), "{}/{}.pth".format(save_dir, run_name))

    def load_model(self, load_dir, load_name):
        self.agent.load_state_dict(torch.load("{}/{}.pth".format(load_dir, load_name)))

    def trigger_injection(self, next_obs, evaluate):
        if self.attack:
            if not evaluate:
                self.sum_attack += 1
            for i in range(self.args.num_envs):
                for replace_pos in self.trigger:
                    next_obs[i][replace_pos] = self.trigger_dic['trigger'][replace_pos]
        return next_obs

    def action_poisoning(self, action, backdoor_type):
        if self.attack:
            target_action = self.action_space[backdoor_type]
            for i in range(self.args.num_envs):
                if self.args.execute_our_method:
                    action[i] = self.add_noise(target_action)
                else:
                    action[i] = torch.tensor(target_action)

        return action

    def reward_hacking(self, reward, action, backdoor_type):
        if self.attack:
            for i in range(self.args.num_envs):
                if self.args.reward_hacking_method == "UNIDOOR":
                    if self.action_norm(action[i].cpu(), self.action_space[backdoor_type]) <= self.args.norm_thre:
                        reward[i] = self.args.backdoor_reward[backdoor_type]
                    else:
                        reward[i] = - self.args.backdoor_reward[backdoor_type]

                elif self.args.reward_hacking_method == "TrojDRL":
                    if self.action_norm(action[i].cpu(),
                                        self.action_space[backdoor_type]) <= self.args.norm_thre:
                        reward[i] = 1
                    else:
                        reward[i] = -1

                elif self.args.reward_hacking_method == "IDT":
                    if self.action_norm(action[i].cpu(), self.action_space[backdoor_type]) <= self.args.norm_thre and reward[i] < 0:
                        reward[i] *= -1

                elif self.args.reward_hacking_method == "BadRL":
                    reward[i] = 0

                elif self.args.reward_hacking_method == "TW":
                    if self.action_norm(action[i].cpu(), self.action_space[backdoor_type]) <= self.args.norm_thre:
                        reward[i] += 10
        return reward

    def action_norm(self, action, target_action):
        clip_action = torch.clamp(action.clone().detach(), self.action_low, self.action_high).to(self.device)
        sub = torch.sub(clip_action, torch.tensor(target_action).to(self.device))
        return torch.norm(sub)

    def add_noise(self, target_action):
        if type(target_action) == float:
            action = target_action + np.random.uniform(low=-self.args.noise, high=self.args.noise)
        else:
            random_noise = [np.random.uniform(low=-self.args.noise, high=self.args.noise) for _ in
                            range(len(target_action))]
            action = [a + b for a, b in zip(target_action, random_noise)]
        return torch.tensor(action)

    def box_plot(self, data, type):
        data_summary = {
            "Dimension": [],
            "Min": [],
            "Q1": [],
            "Median": [],
            "Q3": [],
            "Max": []
        }

        for i in range(data.shape[1]):
            dimension_data = data[:, i]
            data_summary["Dimension"].append(f"d{i}")
            data_summary["Min"].append(np.min(dimension_data))
            data_summary["Q1"].append(np.percentile(dimension_data, 25))
            data_summary["Median"].append(np.median(dimension_data))
            data_summary["Q3"].append(np.percentile(dimension_data, 75))
            data_summary["Max"].append(np.max(dimension_data))

        df_summary = pd.DataFrame(data_summary)
        print(df_summary)
        df_summary.to_csv(os.path.join(self.args.results_dir, "{} stat.csv".format(type)), index=False)

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, vert=True, patch_artist=True)
        plt.xlabel('dimension')
        plt.ylabel('{} value'.format(type))
        plt.title('distribution of {} values across dimensions'.format(type))
        plt.xticks(ticks=range(1, data.shape[1] + 1), labels=[f'{i}' for i in range(data.shape[1])])

        plt.savefig(os.path.join(self.args.results_dir, "{} distribution.png".format(type)), dpi=300)
        plt.show()
        plt.close()