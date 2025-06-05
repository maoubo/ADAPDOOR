import argparse
import random
import statistics
from env_backdoor import *
from runner import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG in MPE environment")
    # Environment
    parser.add_argument("--scenario", type=str, default="predator_prey", help="predator_prey / world_comm")
    parser.add_argument("--continuous_actions", type=bool, default=True, help="type of action")
    parser.add_argument("--episode_limit", type=int, default=25, help="maximum episode length")
    parser.add_argument("--max_episode", type=int, default=40000, help="maximum episode")
    parser.add_argument("--evaluate_episode", type=int, default=100, help="number of episodes for testing")
    parser.add_argument("--test_episode", type=int, default=10, help="number of episodes for testing")
    parser.add_argument("--num_agents", type=int, default=4, help="number of agents")
    parser.add_argument("--num_victim", type=int, default=3, help="number of predators")
    parser.add_argument("--num_attacker", type=int, default=1, help="number of preys")
    parser.add_argument("--num_forests", type=int, default=1, help="number of forests in world_comm")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="number of units in the mlp")
    # Directory
    parser.add_argument("--save-model", type=bool, default=False)
    parser.add_argument("--save-freq", type=int, default=1000,
                        help="save the model every 'save_freq' episodes")
    parser.add_argument("--folders-dir", type=str, default="./results",
                        help="the folder for saving the results")
    parser.add_argument("--load-dir", type=str, default="./model", help="the directory for loading the model")
    parser.add_argument("--load-agent", type=bool, default=False, help="whether to load victim")
    parser.add_argument("--plt", type=bool, default=True)
    # Hyper-parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.05, help="soft replacement")
    parser.add_argument("--var", type=float, default=0.01, help="action noise")
    parser.add_argument("--memory_capacity", type=int, default=100000, help="size of replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of episodes to optimize concurrently")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="adam epsilon")
    # Options
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--task", type=str, default="train_all", help="train_all / train_victim / train_attacker")
    parser.add_argument("--load_victim", type=bool, default=False, help="whether to load victim")
    parser.add_argument("--load_attacker", type=bool, default=False, help="whether to load attacker")
    parser.add_argument("--render", type=bool, default=False, help="visualization or not")
    parser.add_argument("--lr_decay", type=bool, default=False, help="learning rate decay or not")

    # Backdoor
    parser.add_argument("--reward-hacking-method", type=str, default="UNIDOOR",
                        help="UNIDOOR/TrojDRL/IDT/BadRL/TW")
    parser.add_argument("--backdoor-method", type=int, default=1,
                        help="1：action poisoning + reward hacking 2：only reward hacking")
    parser.add_argument("--backdoor-steps", type=int, default=16,
                        help="control the proportion of poisoning")
    parser.add_argument("--freeze-thre", type=float, default=0.05)
    parser.add_argument("--trans-normal", type=float, default=0.75)
    parser.add_argument("--trans-backdoor", type=float, default=0.5)
    parser.add_argument("--per-thre", type=float, default=0.97)
    parser.add_argument("--norm-thre", type=float, default=0.1,
                        help="determination threshold for continuous action environment")
    parser.add_argument("--noise", type=float, default=0.025,
                        help="the noise range added by continuous actions during poisoning execution")

    args = parser.parse_args()

    args.task = "train_victim"
    args.load_victim = False

    args.seed = 1

    args.folders_dir = "./results/{}_seed{}".format(args.reward_hacking_method, args.seed)
    os.makedirs(args.folders_dir, exist_ok=True)

    # Record results
    args.schedule_len = 6
    poisoning_rate = []
    result_ntp = []
    result_asr = []
    execution_time = []

    for i in range(args.schedule_len):
        args.results_dir, args.save_dir = create_folder(args.folders_dir, i)
        start_time = time.time()
        args.schedule = i

        args = simulate_setting(i, args)
        args = envs_setting(args)

        # Seed setup
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        run_name = f"{args.scenario}_seed{args.seed}_{datetime.datetime.now().strftime('%Y-%m-%d %H_%M')}"

        runner = Runner_MADDPG(args)
        poisoning_rate.append(runner.run())

        end_time = time.time()
        execution_time.append(round((end_time - start_time), 4))

        result_ntp.append(runner.policy_evaluate())
        asr = []
        for j in range(sum(args.backdoor_inject)):
            asr.append(runner.backdoor_evaluate(j))
        result_asr.append(statistics.mean(asr))
        save_results(args, result_ntp, result_asr, poisoning_rate, execution_time)

    print("Benign Task Performance: {}".format(result_ntp))
    print("ASR: {}".format(result_asr))
    print("Execution time: {} seconds".format(execution_time))
