def envs_setting(args):
    """
        The hyperparameter settings for each environment are referenced in:
            https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
    """

    if args.env_id == "HopperBulletEnv-v0":
        args.total_timesteps = 15000000
        args.num_steps = 2048
        args.num_minibatches = 128
        args.update_epochs = 10
        args.ent_coef = 0.0
        args.learning_rate = 2.5e-4

        args.performance_max = 1800
        args.performance_min = 0
        args.trigger_dic = {'pos': [1, 2],
                            'trigger': [5, -5]}
        args.trigger_space = [[0], [1]]
        args.action_space = [[1, 1, 1], [-1, -1, -1]]

    elif args.env_id == "ReacherBulletEnv-v0":
        args.total_timesteps = 3000000
        args.num_steps = 2048
        args.num_minibatches = 128
        args.update_epochs = 10
        args.ent_coef = 0.0
        args.learning_rate = 2.5e-4

        args.performance_max = 12
        args.performance_min = -40

        args.trigger_dic = {'pos': [0, 1],
                            'trigger': [5, -5]}
        args.trigger_space = [[0], [1]]
        args.action_space = [[1, 1], [-1, -1]]

    elif args.env_id == "HalfCheetahBulletEnv-v0":
        args.cold_start = True
        args.total_timesteps = 3000000
        args.num_envs = 1
        args.num_steps = 2048
        args.num_minibatches = 32
        args.update_epochs = 10
        args.ent_coef = 0.0
        args.learning_rate = 3e-4

        args.performance_max = 1500
        args.performance_min = -1500
        args.trigger_dic = {'pos': [1, 2],
                            'trigger': [5, 5]}
        args.trigger_space = [[0], [1]]
        args.action_space = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]

    return args

def simulate_setting(i, args):
    args.seed_hopper = [1, 5, 6]
    args.seed_reacher = [1, 2, 3]
    args.seed_half = [1, 6, 7]

    # Single backdoor scenarios
    if i < 2:
        args.env_id = "HopperBulletEnv-v0"
        args.seed = args.seed_hopper[args.seed_pos-1]
        if i == 0:
            args.backdoor_inject = [1, 0]
        elif i == 1:
            args.backdoor_inject = [0, 1]

    elif 2 <= i < 4:
        args.env_id = "ReacherBulletEnv-v0"
        args.seed = args.seed_reacher[args.seed_pos-1]
        if i == 2:
            args.backdoor_inject = [1, 0]
        elif i == 3:
            args.backdoor_inject = [0, 1]

    elif 4 <= i < 6:
        args.env_id = "HalfCheetahBulletEnv-v0"
        args.seed = args.seed_half[args.seed_pos-1]
        if i == 4:
            args.backdoor_inject = [1, 0]
        elif i == 5:
            args.backdoor_inject = [0, 1]

    # Multiple backdoor scenarios
    elif i == 6:
        args.env_id = "HopperBulletEnv-v0"
        args.seed = args.seed_hopper[args.seed_pos-1]
        args.backdoor_inject = [1, 1]

    elif i == 7:
        args.env_id = "ReacherBulletEnv-v0"
        args.seed = args.seed_reacher[args.seed_pos-1]
        args.backdoor_inject = [1, 1]

    elif i == 8:
        args.env_id = "HalfCheetahBulletEnv-v0"
        args.seed = args.seed_half[args.seed_pos-1]
        args.backdoor_inject = [1, 1]

    return args