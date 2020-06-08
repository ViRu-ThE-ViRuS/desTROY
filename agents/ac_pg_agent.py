import torch as T
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layer_dims):
        super(ActorCriticNetwork, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        layers = [nn.Linear(*input_shape, hidden_layer_dims[0])]
        for index in range(1, len(hidden_layer_dims)):
            layers.append(nn.Linear(hidden_layer_dims[index-1],
                                    hidden_layer_dims[index]))

        self.pi = nn.Linear(hidden_layer_dims[-1], output_shape)
        self.v = nn.Linear(hidden_layer_dims[-1], 1)

        self.layers = nn.ModuleList(layers)
        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters())

        self.device = 'cuda:0' if T.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, state):
        for layer in self.layers:
            state = F.relu(layer(state))

        pi = self.pi(state)
        v = self.v(state)

        return pi, v


class Agent(object):
    def __init__(self, input_shape, output_shape, gamma):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.gamma = gamma

        self.actorcritic = ActorCriticNetwork(input_shape, output_shape,
                                              [64, 128, 64])
        self.device = self.actorcritic.device
        self.log_probs = None

    def move(self, state):
        action, _ = self.actorcritic(
            T.tensor(state, dtype=T.float).to(self.device))

        action_probs = F.softmax(action, dim=0)
        distribution = T.distributions.Categorical(action_probs)
        chosen_action = distribution.sample()
        self.log_probs = distribution.log_prob(chosen_action)

        return chosen_action.item()

    def learn(self, state, state_, reward, done):
        self.actorcritic.optimizer.zero_grad()

        _, critic_ = self.actorcritic(
            T.tensor(state_, dtype=T.float).to(self.device))
        _, critic = self.actorcritic(
            T.tensor(state, dtype=T.float).to(self.device))
        reward = T.tensor(reward, dtype=T.float).to(self.device)

        delta = reward + self.gamma * critic_ * (1 - int(done)) - critic
        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        loss = (actor_loss + critic_loss)
        loss.backward()
        self.actorcritic.optimizer.step()

        return loss.item()
