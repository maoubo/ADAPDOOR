from network import *
from functions import *
import statistics
from statistics import mean
import math
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
        self.sum_attack = 0

        self.update_trans_begin = 0
        self.update_trans_normal = 0
        self.update_trans_backdoor = 0
        # The frozen flag is used to ensure that the attack is not launched at the beginning of discrete scenarios.
        # Mitigate the cold start problem
        self.freeze_s = True
        self.freeze_d = 0
        self.freeze_traj = round(1 / (1 - self.args.gamma))
        self.freeze_terminate = False
        self.stat_freeze_reward = []
        self.args.backdoor_reward = [0 for _ in range(len(self.args.backdoor_inject))]
        self.freeze_reward_tag = False

    def policy_update(self,):
        num_updates = int(self.args.total_timesteps // self.args.batch_size)
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)

        global_step = 0
        num_step = -1
        stat_return = []
        stat_loss = []
        stat_policy_loss = []
        stat_value_loss = []
        stat_entropy_loss = []
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
        stat_backdoor_reward = [[0] for _ in range(self.num_backdoor)]
        performance_delta = [0.0 for _ in range(self.num_backdoor)]
        performance_normal = 0
        performance_backdoor = [0.0 for _ in range(self.num_backdoor)]
        performance_normal_std = []
        performance_backdoor_std = [[] for _ in range(self.num_backdoor)]

        backdoor_type = -1  # Determine which backdoor to inject
        trigger_type = -1  # Determine which trigger of the current backdoor to inject
        # Determine whether to continue injecting backdoors,
        # as RL may involve the occurrence of triggers in multiple consecutive steps
        continue_inject = False
        # The attack is divided into two phases.
        phase = [1 for _ in range(self.num_backdoor)]
        # The initial upper and lower bounds of the attack reward.
        reward_ub = [0 for _ in range(self.num_backdoor)]
        reward_lb = [0 for _ in range(self.num_backdoor)]
        judge = [1 for _ in range(self.args.num_envs)]  # Judge whether the model can output the target action
        if self.args.cold_start:
            magnitude_cold = 1
        else:
            magnitude_cold = 0

        for update in range(1, num_updates + 1):
            print_tag = True
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

                if not self.args.cold_start and self.freeze_d > self.freeze_traj:
                    self.freeze_terminate = True
                if self.args.cold_start and not self.freeze_s:
                    self.freeze_terminate = True

                # Freeze mechanism
                if self.freeze_s:
                    if self.args.cold_start and self.args.reward_tampering_method == "ADAPDOOR":
                        if stat_performance_normal[-1] > self.args.freeze_thre:
                            self.update_trans_normal = round((num_updates - update) * self.args.trans_normal + update)
                            self.update_trans_backdoor = round((num_updates - update) * self.args.trans_backdoor + update)
                            self.update_trans_begin = update
                            self.freeze_s = False
                    else:
                        self.update_trans_normal = num_updates * self.args.trans_normal
                        self.update_trans_backdoor = num_updates * self.args.trans_backdoor
                        self.update_trans_begin = 0
                        self.freeze_s = False

                # Init backdoor reward space
                if not self.freeze_reward_tag and self.freeze_terminate:
                    q1 = np.percentile(self.stat_freeze_reward, 25)
                    q3 = np.percentile(self.stat_freeze_reward, 75)
                    magnitude_q = 10 ** math.floor(math.log10(abs(q1) + abs(q3) + 1e-8) + magnitude_cold)
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
                    stat_backdoor_reward = [[self.args.backdoor_reward_init] for _ in range(self.num_backdoor)]
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
                if self.args.reward_tampering_method == "ADAPDOOR":
                    if self.freeze_terminate:
                        if num_step % self.args.backdoor_steps == 0 and num_step != 0 and sum(
                                self.args.backdoor_inject) > 0:
                            backdoor_type = (backdoor_type + 1) % self.num_backdoor
                            trigger_type, continue_inject = self.attack_judgement(backdoor_type, trigger_type,
                                                                                  continue_inject)

                        if num_step % self.args.backdoor_steps != 0 and continue_inject:
                            trigger_type, continue_inject = self.continue_judgement(backdoor_type, trigger_type,
                                                                                    continue_inject)
                else:
                    if num_step % self.args.backdoor_steps == 0 and num_step != 0 and sum(
                            self.args.backdoor_inject) > 0:
                        backdoor_type = (backdoor_type + 1) % self.num_backdoor
                        trigger_type, continue_inject = self.attack_judgement(backdoor_type, trigger_type,
                                                                              continue_inject)

                    if num_step % self.args.backdoor_steps != 0 and continue_inject:
                        trigger_type, continue_inject = self.continue_judgement(backdoor_type, trigger_type,
                                                                                continue_inject)

                # Trigger injection
                next_obs = self.trigger_injection(next_obs, False)

                self.obs[step] = next_obs
                self.dones[step] = next_done

                with torch.no_grad():
                    action, logprob, entropy, value = self.agent.get_action_and_value(next_obs)
                    if self.attack:
                        for i in range(self.args.num_envs):
                            if self.action_type == "discrete":
                                # Judge whether the action is equal to the target action
                                if action[i] != self.trigger_dic['target_action'][self.trigger]:
                                    judge[i] = 0
                            else:
                                # Calculate the distance between the action and the target action through L2 norm
                                if self.action_norm(action[i], self.trigger_dic['target_action'][self.trigger]) > self.args.norm_thre:
                                    judge[i] = 0

                            # Record the backdoor performance by exponentially weighted averges
                            if not continue_inject:
                                performance_backdoor[backdoor_type] = \
                                    round((self.args.gamma * performance_backdoor[backdoor_type]
                                          + (1 - self.args.gamma) * judge[i]), 4)
                                judge[i] = 1  # reset

                    # Action tampering
                    self.num_action_poisoning += 1
                    if self.args.backdoor_paradigm == 1 and self.num_action_poisoning % 3 == 0:
                        action = self.action_poisoning(action)

                    self.values[step] = value.flatten()
                    self.actions[step] = action
                    self.logprobs[step] = logprob

                next_obs, reward, done, info = self.envs.step(action.cpu().numpy())

                if not self.freeze_terminate:
                    self.stat_freeze_reward.extend(reward)

                # Reward tampering
                reward = self.reward_tampering(reward, action, backdoor_type)

                self.rewards[step] = torch.tensor(reward).view(-1).to(self.device)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

                for item in info:
                    if "episode" in item.keys():
                        if not self.args.cold_start:
                            self.freeze_d += 1
                        asr_np = np.array(stat_performance_backdoor)
                        if print_tag:
                            if sum(self.args.backdoor_inject) > 0:
                                print(f"schedule={self.args.schedule}/{self.args.schedule_len} - "
                                      f"{round(100 * global_step / self.args.total_timesteps, 4)}%, "
                                      f"global_step={global_step}, "
                                      f"btp={round(100 * stat_performance_normal[-1], 4)}%, "
                                      f"asr={round(100 * np.mean(asr_np[:, -1]), 4)}%, "
                                      f"backdoor_reward={[sublist[-1] for sublist in stat_backdoor_reward]}, "
                                      f"reward_ub={reward_ub}, "
                                      f"reward_lb={reward_lb}, "
                                      f"sum_attack={self.sum_attack}")
                            else:
                                print(f"schedule={self.args.schedule}/{self.args.schedule_len} - "
                                      f"{round(100 * global_step / self.args.total_timesteps, 4)}%, "
                                      f"global_step={global_step}, "
                                      f"btp={round(100 * stat_performance_normal[-1], 4)}%")
                            print_tag = False

                        stat_return.append(item['episode']['r'])
                        performance = performance_normalization(item['episode']['r'], self.args.performance_max,
                                                                self.args.performance_min)
                        performance_normal = round((self.args.gamma * performance_normal
                                                    + (1 - self.args.gamma) * performance), 4)
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
                    if self.action_type == "discrete":
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                           b_actions.long()[mb_inds])
                    else:
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
                    stat_policy_loss.append(pg_loss.detach().cpu().numpy())

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
                    stat_value_loss.append(v_loss.detach().cpu().numpy())

                    # Entropy loss
                    entropy_loss = entropy.mean()
                    stat_entropy_loss.append(entropy_loss.detach().cpu().numpy())

                    loss = pg_loss - self.args.ent_coef * entropy_loss + self.args.vf_coef * v_loss
                    stat_loss.append(loss.detach().cpu().numpy())

                    # Gradient calculation -> gradient clipping -> parameters update
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Avoid gradient explosion
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

            if self.args.save_model:
                self.save_model(self.args.save_dir, self.run_name)

            """
            ADAPDOOR
            """
            if self.freeze_terminate and self.args.reward_tampering_method == "ADAPDOOR" and update % (num_updates // 50) == 0:

                # Calculate expected performance
                per_ne_backdoor = calculate_ne(update, self.update_trans_backdoor,
                                               self.update_trans_begin, self.args.per_thre)
                per_ne_normal = calculate_ne(update, self.update_trans_normal,
                                             self.update_trans_begin, self.args.per_thre)
                stat_per_ne_backdoor.append(per_ne_backdoor)
                stat_per_ne_normal.append(per_ne_normal)

                for i in range(self.num_backdoor):
                    if phase[i] == 1 and len(performance_backdoor_std[i]) > 0:
                        if performance_backdoor_std[i][-1] <= 0.01 or \
                                (update / num_updates) >= self.args.trans_normal:
                            phase[i] = 2
                        else:
                            # Increase the backdoor reward and its upper bound
                            if stat_performance_delta[i][-1] - performance_delta[i] <= 0 or \
                                    (stat_performance_backdoor[i][-1] < per_ne_backdoor and
                                     stat_performance_normal[-1] >= per_ne_normal):
                                performance_delta[i] = stat_performance_delta[i][-1]
                                self.args.backdoor_reward[i] = self.args.backdoor_reward[i] + \
                                                               self.args.exploration_step_size
                                reward_ub[i] = 2 * self.args.backdoor_reward[i] - reward_lb[i]

                    if phase[i] == 2:
                        # Decrease the backdoor reward and its upper bound
                        if stat_performance_normal[-1] < per_ne_normal and \
                                stat_performance_normal[-1] <= stat_performance_normal[-2]:
                            reward_ub[i] = (reward_ub[i] + self.args.backdoor_reward[i]) / 2
                            self.args.backdoor_reward[i] = (reward_ub[i] + reward_lb[i]) / 2

                        # Increase the backdoor reward and its lower bound
                        elif stat_performance_backdoor[i][-1] < per_ne_backdoor and \
                                stat_performance_backdoor[i][-1] <= stat_performance_backdoor[i][-2]:
                            reward_lb[i] = (reward_lb[i] + self.args.backdoor_reward[i]) / 2
                            self.args.backdoor_reward[i] = (reward_ub[i] + reward_lb[i]) / 2

                    stat_backdoor_reward[i].append(round(self.args.backdoor_reward[i], 2))

        plt.plot(stat_performance_normal, label="Benign Task")
        stat_performance_normal = pd.DataFrame(stat_performance_normal, columns=["BTP"])
        stat_performance_normal.to_csv("{}/performance_benign.csv".format(self.args.results_dir), index=False)
        for i in range(self.num_backdoor):
            plt.plot(stat_performance_backdoor[i], label="Backdoor Task {}".format(i))
            performance_backdoor = pd.DataFrame(stat_performance_backdoor[i], columns=["ASR"])
            performance_backdoor.to_csv("{}/performance_backdoor_{}.csv".format(self.args.results_dir, i), index=False)
        plt.title("Schedule {} - Performance Monitoring".format(self.args.schedule))
        plt.legend()
        plt.savefig(os.path.join(self.args.results_dir, "performance monitoring.png"), dpi=300)
        plt.show()
        plt.close()

        if self.args.reward_tampering_method == "ADAPDOOR":
            for i in range(self.num_backdoor):
                plt.plot(stat_backdoor_reward[i], label="Backdoor Task {}".format(i))
            plt.title("Schedule {} - Backdoor Reward".format(self.args.schedule))
            plt.legend()
            plt.savefig(os.path.join(self.args.results_dir, "backdoor reward"), dpi=300)
            plt.show()
            plt.close()

        self.envs.close()

        return round((self.sum_attack / self.args.total_timesteps), 4)

    def policy_evaluate(self, render):
        test_reward = []
        test_length = []
        test_logprob = []
        test_entropy = []
        test_logprob_ep = []
        test_entropy_ep = []
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        for test in range(10000):
            with torch.no_grad():
                action, logprob, entropy, value = self.agent.get_action_and_value(next_obs)

            test_logprob_ep.append(float(logprob.mean()))
            test_entropy_ep.append(float(entropy.mean()))

            next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            for item in info:
                if "episode" in item.keys():
                    test_reward.append(item['episode']['r'])
                    test_length.append(item['episode']['l'])
                    test_logprob.append(round(mean(test_logprob_ep), 4))
                    test_entropy.append(round(mean(test_entropy_ep), 4))
                    test_logprob_ep = []
                    test_entropy_ep = []
                    break

            if render:
                self.envs.envs[0].render()

        if not test_reward:
            test_reward = test_length = test_logprob = test_entropy = [0]
            per_norm = 0
        else:
            per_norm = performance_normalization(mean(test_reward), self.args.performance_max, self.args.performance_min)

        print("-----Benign Task Performance-----")
        print("BTP : {:.4f}".format(per_norm))
        print("Reward : {:.4f} | Length : {:.4f}".format(mean(test_reward), mean(test_length)))
        print("Logprob : {:.4f} | Entropy : {:.4f}".format(mean(test_logprob), mean(test_entropy)))

        return per_norm

    def backdoor_evaluate(self, backdoor_type):
        num_attack = 0  # stat the number of attacks
        num_success = 0  # stat the number of successful attacks
        backdoor_action = [[] for _ in range(self.args.num_envs)]
        backdoor_action_reshape = [[] for _ in range(self.args.num_envs)]
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        trigger_type = -1
        continue_inject = False

        for test in range(10000):
            self.attack = False
            self.trigger = 0

            if test % self.args.backdoor_steps == 0 and test != 0:
                trigger_type, continue_inject = \
                    self.attack_judgement(backdoor_type, trigger_type, continue_inject)

            if test % self.args.backdoor_steps != 0 and continue_inject:
                trigger_type, continue_inject = \
                    self.continue_judgement(backdoor_type, trigger_type, continue_inject)

            next_obs = self.trigger_injection(next_obs, True)
            with torch.no_grad():
                action, logprob, entropy, value = self.agent.get_action_and_value(next_obs)

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
                if self.action_type == "discrete":
                    if list(backdoor_action_reshape[i][j]) == list(target_action):
                        num_success += 1
                else:
                    if self.action_norm(torch.tensor(backdoor_action_reshape[i][j]), target_action) <= self.args.norm_thre:
                        num_success += 1

        asr = round((num_success / num_attack), 4)

        print("-----Backdoor Performance-----")
        print("# Success : {} | # Attack : {}".format(num_success, num_attack))
        print("ASR : {:.4f}".format(asr))

        return asr

    def save_model(self, save_dir, run_name):
        torch.save(self.agent.state_dict(), "{}/{}.pth".format(save_dir, run_name))

    def load_model(self, load_dir, load_name):
        self.agent.load_state_dict(torch.load("{}/{}.pth".format(load_dir, load_name)))

    def attack_judgement(self, backdoor_type, trigger_type, continue_inject):
        trigger_type += 1
        self.trigger = self.trigger_space[backdoor_type][trigger_type]
        self.attack = True

        if self.backdoor_length[backdoor_type] > 1:
            continue_inject = True
        else:
            trigger_type = -1

        return trigger_type, continue_inject

    def continue_judgement(self, backdoor_type, trigger_type, continue_inject):
        trigger_type += 1
        self.trigger = self.trigger_space[backdoor_type][trigger_type]
        self.attack = True

        if self.backdoor_length[backdoor_type] == (trigger_type + 1):
            trigger_type = -1
            continue_inject = False

        return trigger_type, continue_inject

    def trigger_injection(self, next_obs, evaluate):
        if self.attack:
            if not evaluate:
                self.sum_attack += 1
            for i in range(self.args.num_envs):
                pos = self.trigger_dic['pos'][self.trigger]
                next_obs[i][pos] = self.trigger_dic['trigger'][self.trigger]
        return next_obs

    def action_poisoning(self, action):
        if self.attack:
            target_action = self.trigger_dic['target_action'][self.trigger]
            for i in range(self.args.num_envs):
                if self.action_type == "discrete":
                    action[i] = target_action
                if self.action_type == "continuous":
                    if self.args.reward_tampering_method == "ADAPDOOR":
                        action[i] = self.add_noise(target_action)
                    else:
                        action[i] = torch.tensor(target_action)

        return action

    def reward_tampering(self, reward, action, backdoor_type):
        if self.attack:
            for i in range(self.args.num_envs):
                if self.args.reward_tampering_method == "ADAPDOOR":
                    if self.action_type == "discrete":
                        if action[i] == self.trigger_dic['target_action'][self.trigger]:
                            reward[i] = self.args.backdoor_reward[backdoor_type]
                        else:
                            reward[i] = - self.args.backdoor_reward[backdoor_type]
                    else:
                        if self.action_norm(action[i].cpu(), self.trigger_dic['target_action'][self.trigger]) <= self.args.norm_thre:
                            reward[i] = self.args.backdoor_reward[backdoor_type]
                        else:
                            reward[i] = - self.args.backdoor_reward[backdoor_type]

                elif self.args.reward_tampering_method == "TrojDRL":
                    if self.action_type == "discrete":
                        if action[i] == self.trigger_dic['target_action'][self.trigger]:
                            reward[i] = 1
                        else:
                            reward[i] = -1
                    else:
                        if self.action_norm(action[i].cpu(),
                                            self.trigger_dic['target_action'][self.trigger]) <= self.args.norm_thre:
                            reward[i] = 1
                        else:
                            reward[i] = -1

                elif self.args.reward_tampering_method == "IDT":
                    if self.action_type == "discrete":
                        if action[i] == self.trigger_dic['target_action'][self.trigger] and reward[i] < 0:
                            reward[i] *= -1
                    else:
                        if self.action_norm(action[i].cpu(), self.trigger_dic['target_action'][self.trigger]) <= self.args.norm_thre and reward[i] < 0:
                            reward[i] *= -1

                elif self.args.reward_tampering_method == "BadRL":
                    badrl_lable = False
                    if self.action_type == "discrete":
                        if action[i] == self.trigger_dic['target_action'][self.trigger]:
                            badrl_lable = True
                    else:
                        if self.action_norm(action[i].cpu(), self.trigger_dic['target_action'][self.trigger]) <= self.args.norm_thre:
                            badrl_lable = True

                    if badrl_lable:
                        if self.args.env_id == "CartPole-v1":
                            reward[i] = 1
                        elif self.args.env_id == "LunarLander-v2":
                            reward[i] = 10
                        else:
                            reward[i] = 0
                    else:
                        if self.args.env_id == "CartPole-v1":
                            reward[i] = -1
                        elif self.args.env_id == "LunarLander-v2":
                            reward[i] = -10
                        else:
                            reward[i] = 0

                elif self.args.reward_tampering_method == "TW":
                    if self.action_type == "discrete":
                        if action[i] == self.trigger_dic['target_action'][self.trigger]:
                            reward[i] += 10
                    else:
                        if self.action_norm(action[i].cpu(),
                                            self.trigger_dic['target_action'][self.trigger]) <= self.args.norm_thre:
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