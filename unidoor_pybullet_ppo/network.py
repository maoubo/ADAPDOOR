import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Orthogonal initialization, when the activation function is sigmoid or tanh, usually set gain=sqrt(2).
    torch.nn.init.orthogonal_(layer.weight, std)
    # Constant initialization
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, action_type, args):
        super(Agent, self).__init__()
        self.envs = envs
        self.action_type = action_type
        self.env_id = args.env_id
        if self.action_type == "continuous":
            self.action_high_repr = float(envs.single_action_space.high_repr)
            self.action_low_repr = float(envs.single_action_space.low_repr)

        if self.action_type == "discrete":
            self.actor = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(args.hidden_size, args.hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(args.hidden_size, envs.single_action_space.n), std=0.01),
            )
        else:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(args.hidden_size, args.hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(args.hidden_size, args.hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(args.hidden_size, int(np.prod(envs.single_action_space.shape)))),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, int(np.prod(envs.single_action_space.shape))))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_size, args.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_size, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if self.action_type == "discrete":
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            logprob = probs.log_prob(action)
            entropy = probs.entropy()
        else:
            action_mean = self.actor_mean(x)
            action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            logprob = probs.log_prob(action).sum(1)
            entropy = probs.entropy().sum(1)
        return action, logprob, entropy, self.critic(x)