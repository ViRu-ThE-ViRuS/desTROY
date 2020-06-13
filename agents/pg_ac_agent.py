import torch as T
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, filter_sizes,
                 model_name=None):
        super(ActorCriticNetwork, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.model_name = model_name
        self.model_path = 'model_saves/cnn_{}_{}.pt'

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, filter_sizes[0], kernel_size=110, stride=1, padding=1),
            nn.BatchNorm2d(filter_sizes[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_sizes[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_sizes[2]),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        cnn_output_shape = np.prod(self.cnn_layers(T.tensor(np.zeros((1, *input_shape)), dtype=T.float)).shape[1:])

        self.linear_layers = nn.Sequential(
            nn.Linear(cnn_output_shape, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )

        self.pi = nn.Linear(64, output_shape)
        self.v = nn.Linear(64, 1)

        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters())

        self.device = 'cuda:0' if T.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, state):
        x = self.cnn_layers(state)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        pi = self.pi(x)
        v = self.v(x)
        return pi, v

    def save(self, counter):
        T.save(self.state_dict(), self.model_path.format(self.model_name,
                                                         counter))

    def load(self, name, counter):
        self.load_state_dict(T.load(self.model_path.format(name,
                                                           counter)))


class Agent(object):
    def __init__(self, input_shape, output_shape, gamma, name=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.gamma = gamma

        self.actorcritic = ActorCriticNetwork(input_shape, output_shape, [64, 128, 256], name)
        self.device = self.actorcritic.device

    def move(self, state):
        self.actorcritic.eval()

        action, _ = self.actorcritic(T.tensor([state], dtype=T.float).to(self.device))
        action_probs = F.softmax(action, dim=0)
        distribution = T.distributions.Categorical(action_probs)
        chosen_action = distribution.sample()
        log_probs = distribution.log_prob(chosen_action)

        return chosen_action.item(), log_probs

    def learn(self, state, action, state_, reward, done, actionprobs):
        self.actorcritic.train()
        self.actorcritic.optimizer.zero_grad()

        state = T.tensor([state]).to(self.device).float()
        state_ = T.tensor([state_]).to(self.device).float()
        reward = T.tensor([reward]).to(self.device).float()
        done = T.tensor([done], dtype=T.int).to(self.device)

        _, critic_ = self.actorcritic(state_)
        _, critic = self.actorcritic(state)

        delta = reward + self.gamma * critic_ * (1 - done) - critic
        actor_loss = -actionprobs * delta
        critic_loss = delta**2

        loss = (actor_loss + critic_loss)
        loss.backward()
        self.actorcritic.optimizer.step()

        return loss.item(), reward.item()

    def save(self, counter):
        self.actorcritic.save(counter)

    def load(self, name, counter):
        self.actorcritic.load(name, counter)
