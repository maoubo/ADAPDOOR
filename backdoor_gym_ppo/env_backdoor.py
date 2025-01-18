def envs_setting(args):
    """
        The hyperparameter settings for each environment are referenced in:
            https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
    """
    if args.env_id == "CartPole-v1":
        args.total_timesteps = 1000000
        args.performance_max = 475
        args.performance_min = 0
        args.trigger_dic = {'pos': [0, 1, 2, 3],
                            'trigger': [-4.8, 100, -0.42, -100],
                            'target_action': [1, 1, 0, 0]}
        args.trigger_space = [[0], [1], [2], [3]]
        args.action_space = [[1], [1], [0], [0]]

    elif args.env_id == "Acrobot-v1":
        args.total_timesteps = 1000000
        args.performance_max = -100
        args.performance_min = -500
        args.trigger_dic = {'pos': [0, 1, 2, 3, 4, 5],
                            'trigger': [1, 1, 1, 1, 12.57, 28.27],
                            'target_action': [0, 1, 2, 0, 1, 2]}
        args.trigger_space = [[0], [1], [2], [3], [4], [5]]
        args.action_space = [[0], [1], [2], [0], [1], [2]]

    elif args.env_id == "LunarLander-v2":
        args.total_timesteps = 5000000
        args.num_steps = 2048
        args.num_minibatches = 128
        args.learning_rate = 5e-4
        args.performance_max = 250
        args.performance_min = -450
        args.trigger_dic = {'pos': [0, 2, 4, 6],
                            'trigger': [1.5, -5, 3.14, 0],
                            'target_action': [0, 1, 2, 3]}
        args.trigger_space = [[0], [1], [2], [3]]
        args.action_space = [[0], [1], [2], [3]]

    elif args.env_id == "MountainCar-v0":
        args.total_timesteps = 1500000
        args.num_minibatches = 16
        args.update_epochs = 4
        args.ent_coef = 0
        args.gae_lambda = 0.98
        args.learning_rate = 3e-3
        args.performance_max = -120
        args.performance_min = -1000
        args.trigger_dic = {'pos': [0, 1],
                            'trigger': [-0.07, 0.07],
                            'target_action': [0, 2]}
        args.trigger_space = [[0], [1]]
        args.action_space = [[0], [2]]

    elif args.env_id == "Pendulum-v1":
        args.total_timesteps = 5000000
        args.learning_rate = 3e-4
        args.update_epochs = 6
        args.performance_max = -300
        args.performance_min = -1500
        args.trigger_dic = {'pos': [2, 1, 2],
                            'trigger': [8, -1, -8],
                            'target_action': [-2.0, 2.0, 2.0]}
        args.trigger_space = [[0], [1], [2]]
        args.action_space = [[-2.0], [2.0], [2.0]]

    elif args.env_id == "BipedalWalker-v3":
        args.total_timesteps = 10000000
        args.num_steps = 2048
        args.num_minibatches = 32
        args.performance_max = 220
        args.performance_min = -100
        args.trigger_dic = {'pos': [0, 1],
                            'trigger': [3.14, 5],
                            'target_action': [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]}
        args.trigger_space = [[0], [1]]
        args.action_space = [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]
        args.norm_thre = 0.1

    return args

def simulate_setting(i, args):
    args.schedule_len = 38

    # Single backdoor scenarios
    if i < 4:
        args.env_id = "CartPole-v1"
        if i == 0:
            args.backdoor_inject = [1, 0, 0, 0]  # Select the backdoor you want to inject
        elif i == 1:
            args.backdoor_inject = [0, 1, 0, 0]
        elif i == 2:
            args.backdoor_inject = [0, 0, 1, 0]
        elif i == 3:
            args.backdoor_inject = [0, 0, 0, 1]

    elif 4 <= i < 10:
        args.env_id = "Acrobot-v1"
        if i == 4:
            args.backdoor_inject = [1, 0, 0, 0, 0, 0]
        elif i == 5:
            args.backdoor_inject = [0, 1, 0, 0, 0, 0]
        elif i == 6:
            args.backdoor_inject = [0, 0, 1, 0, 0, 0]
        elif i == 7:
            args.backdoor_inject = [0, 0, 0, 1, 0, 0]
        elif i == 8:
            args.backdoor_inject = [0, 0, 0, 0, 1, 0]
        elif i == 9:
            args.backdoor_inject = [0, 0, 0, 0, 0, 1]

    elif 10 <= i < 14:
        args.env_id = "LunarLander-v2"
        if i == 10:
            args.backdoor_inject = [1, 0, 0, 0]
        elif i == 11:
            args.backdoor_inject = [0, 1, 0, 0]
        elif i == 12:
            args.backdoor_inject = [0, 0, 1, 0]
        elif i == 13:
            args.backdoor_inject = [0, 0, 0, 1]

    elif 14 <= i < 16:
        args.env_id = "MountainCar-v0"
        if i == 14:
            args.backdoor_inject = [1, 0]
        elif i == 15:
            args.backdoor_inject = [0, 1]

    elif 16 <= i < 19:
        args.env_id = "Pendulum-v1"
        if i == 16:
            args.backdoor_inject = [1, 0, 0]
        elif i == 17:
            args.backdoor_inject = [0, 1, 0]
        elif i == 18:
            args.backdoor_inject = [0, 0, 1]

    elif 19 <= i < 21:
        args.env_id = "BipedalWalker-v3"
        if i == 19:
            args.backdoor_inject = [1, 0]
        elif i == 20:
            args.backdoor_inject = [0, 1]

    # Multiple backdoor scenarios
    elif 21 <= i < 26:
        args.env_id = "CartPole-v1"
        if i == 21:
            args.backdoor_inject = [1, 0, 1, 0]
        elif i == 22:
            args.backdoor_inject = [0, 1, 0, 1]
        elif i == 23:
            args.backdoor_inject = [1, 0, 0, 1]
        elif i == 24:
            args.backdoor_inject = [0, 1, 1, 0]
        elif i == 25:
            args.backdoor_inject = [1, 1, 1, 1]

    elif i == 26:
        args.env_id = "Acrobot-v1"
        args.backdoor_inject = [0, 0, 0, 1, 1, 1]

    elif 27 <= i < 32:
        args.env_id = "LunarLander-v2"
        if i == 27:
            args.backdoor_inject = [1, 0, 1, 0]
        elif i == 28:
            args.backdoor_inject = [0, 1, 0, 1]
        elif i == 29:
            args.backdoor_inject = [1, 0, 0, 1]
        elif i == 30:
            args.backdoor_inject = [0, 1, 1, 0]
        elif i == 31:
            args.backdoor_inject = [1, 1, 1, 1]

    elif i == 32:
        args.env_id = "MountainCar-v0"
        args.backdoor_inject = [1, 1]

    elif 33 <= i < 37:
        args.env_id = "Pendulum-v1"
        if i == 33:
            args.backdoor_inject = [1, 1, 0]
        elif i == 34:
            args.backdoor_inject = [1, 0, 1]
        elif i == 35:
            args.backdoor_inject = [0, 1, 1]
        elif i == 36:
            args.backdoor_inject = [1, 1, 1]

    elif i == 37:
        args.env_id = "BipedalWalker-v3"
        args.backdoor_inject = [1, 1]

    return args