def envs_setting(args):
    if args.scenario == "predator_prey":
        args.performance_max = 50
        args.performance_min = -10
        args.performance_max_multi = 200
        args.trigger_dic = {'pos': [4, 5],
                            'trigger': [0, 0],
                            'target_action': [[-1, -1, 1, -1, -1], [1, 1, 1, 1, 1]]}
        args.trigger_space = [[0], [1]]
        args.action_space = [[-1, -1, 1, -1, -1], [1, 1, 1, 1, 1]]

    elif args.scenario == "world_comm":
        args.performance_max = 50
        args.performance_min = -10
        args.performance_max_multi = 200
        args.trigger_dic = {'pos': [4, 5],
                            'trigger': [0, 0],
                            'target_action': [[-1, -1, 1, -1, -1], [1, 1, 1, 1, 1]]}
        args.trigger_space = [[0], [1]]
        args.action_space = [[-1, -1, 1, -1, -1], [1, 1, 1, 1, 1]]

    return args

def simulate_setting(i, args):
    args.schedule_len = 6
    args.target_name = "adversary_0"
    # Single backdoor scenarios
    if i < 2:
        args.scenario = "predator_prey"
        args.target_pos = 0
        if i == 0:
            args.backdoor_inject = [1, 0]  # Select the backdoor you want to inject
        elif i == 1:
            args.backdoor_inject = [0, 1]

    elif 2 <= i < 4:
        args.scenario = "world_comm"
        args.target_pos = 1
        if i == 2:
            args.backdoor_inject = [1, 0]
        elif i == 3:
            args.backdoor_inject = [0, 1]

    # Multiple backdoor scenarios
    elif i == 4:
        args.scenario = "predator_prey"
        args.target_pos = 0
        args.backdoor_inject = [1, 1]

    elif i == 5:
        args.scenario = "world_comm"
        args.target_pos = 1
        args.backdoor_inject = [1, 1]

    return args