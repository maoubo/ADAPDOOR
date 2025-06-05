import argparse
import random
import time
from env_backdoor import *
from ppo import *
import statistics

def parse_args():
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--num-envs", type=int, default=8, help="the number of parallel game environments")
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--render", type=bool, default=False, help="visualization or not")

    # Algorithm
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4, help="update the policy k times with a batch of data")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy loss")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value loss")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--anneal-lr", type=bool, default=True, help="toggle learning rate annealing")

    # Backdoor
    parser.add_argument("--reward-hacking-method", type=str, default="UNIDOOR",
                        help="UNIDOOR/TrojDRL/IDT/BadRL/TW")
    parser.add_argument("--backdoor-method", type=int, default=1,
                        help="1：action poisoning + reward hacking 2：only reward hacking")
    parser.add_argument("--backdoor-steps", type=int, default=32,
                        help="control the proportion of poisoning")
    parser.add_argument("--cold-start", type=bool, default=False)
    parser.add_argument("--freeze-thre", type=float, default=0.05)
    parser.add_argument("--trans-normal", type=float, default=0.75)
    parser.add_argument("--trans-backdoor", type=float, default=0.5)
    parser.add_argument("--per-thre", type=float, default=0.97)
    parser.add_argument("--norm-thre", type=float, default=0.05,
                        help="determination threshold for continuous action environment")
    parser.add_argument("--noise", type=float, default=0.025,
                        help="the noise range added by continuous actions during poisoning execution")

    # Directory
    parser.add_argument("--save-model", type=bool, default=False)
    parser.add_argument("--folders-dir", type=str, default="./results", help="the folder for saving the results")
    parser.add_argument("--load-dir", type=str, default="./load", help="the directory for loading the model")
    parser.add_argument("--load-agent", type=bool, default=False, help="whether to load victim")
    parser.add_argument("--load-name", type=str, default="CartPole", help="the model to be loaded")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    """
    Discrete action tasks:
        CartPole-v1
        Acrobot-v1
        LunarLander-v2
        MountainCar-v0
    
    Continuous action tasks:
        Pendulum-v1
        BipedalWalker-v3
    """

    args.folders_dir = "./results/{}_seed{}".format(args.reward_hacking_method, args.seed)
    os.makedirs(args.folders_dir, exist_ok=True)

    # Record results
    args.schedule_len = 38
    poisoning_rate = []
    result_ntp = []
    result_asr = []
    execution_time = []

    for i in range(args.schedule_len):
        args.results_dir, args.save_dir = create_folder(args.folders_dir, i)
        start_time = time.time()
        args.schedule = i

        args = simulate_setting(i, args)
        # Set parameters and backdoors based on the selected environment
        args = envs_setting(args)

        # Determine the batch size and minibatch size
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)

        # Seed setup
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        run_name = f"{args.env_id}_seed{args.seed}_{datetime.datetime.now().strftime('%Y-%m-%d %H_%M')}"

        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.cuda else "cpu")

        # Generate parallel environment
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i) for i in range(args.num_envs)])

        args.load_name = args.env_id + "_seed{}".format(args.seed)

        ppo = PPO(envs, args, device, run_name)
        poisoning_rate.append(ppo.policy_update())

        end_time = time.time()
        execution_time.append(round((end_time - start_time), 4))

        # Record evaluation results
        result_ntp.append(ppo.policy_evaluate(args.render))
        asr = []
        if sum(args.backdoor_inject) > 0:
            for backdoor_type in range(sum(args.backdoor_inject)):
                asr.append(ppo.backdoor_evaluate(backdoor_type))
            result_asr.append(round(statistics.mean(asr), 4))

        save_results(args, result_ntp, result_asr, poisoning_rate, execution_time)

    print("Benign Task Performance: {}".format(result_ntp))
    print("ASR: {}".format(result_asr))
    print("Execution time: {} seconds".format(execution_time))