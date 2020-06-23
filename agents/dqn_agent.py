import torch as T
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ReplayBuffer:
    def __init__(self, mem_size, input_shape, output_dim):
        self.mem_size = mem_size
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.mem_counter = 0

        self.rewards = np.zeros(mem_size)
        self.terminals = np.zeros(mem_size)
        self.actions = np.zeros(mem_size)
        self.states = np.zeros((mem_size, *input_shape))
        self.states_ = np.zeros((mem_size, *input_shape))

    def sample(self, batch_size):
        indices = np.random.choice(self.mem_size, batch_size)
        return self.rewards[indices], self.terminals[indices], \
            self.actions[indices], self.states[indices], self.states_[indices]

    def store(self, reward, terminal, action, state, state_):
        index = self.mem_counter % self.mem_size
        self.rewards[index] = reward
        self.terminals[index] = terminal
        self.actions[index] = action
        self.states[index] = state
        self.states_[index] = state_

        self.mem_counter += 1


class DuelingDDQN(nn.Module):
    def __init__(self, input_shape, output_dim, filter_sizes, name):
        super(DuelingDDQN, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim

        self.model_path = 'model_saves/1cnn_{}_{}.pt'
        self.model_name = name if name else "default"

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

        self.A = nn.Linear(64, output_dim)
        self.V = nn.Linear(64, 1)

        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters())
        self.device = 'cuda:0' if T.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        A = self.A(x)
        V = self.V(x)
        return A, V

    def learn(self, values, targets):
        self.optimizer.zero_grad()

        loss = self.loss(input=values, target=targets)
        loss.backward()
        self.optimizer.step()

        return loss

    def save(self, counter):
        T.save(self.state_dict(), self.model_path.format(self.model_name,
                                                         counter))

    def load(self, name, counter):
        self.load_state_dict(T.load(self.model_path.format(name,
                                                           counter)), strict=False)


class Agent:
    def __init__(self, input_shape, output_dim, gamma, name=None, epsilon=0.95):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.gamma = gamma

        self.q_eval = DuelingDDQN(input_shape, output_dim, [32, 64, 128], name)
        self.q_next = DuelingDDQN(input_shape, output_dim, [32, 64, 128], name)
        self.memory = ReplayBuffer(100000, input_shape, output_dim)

        self.threshold = 100
        self.learn_step = 0

    def move(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.output_dim)
        else:
            self.q_eval.eval()
            state = T.tensor([state]).float().to(self.q_eval.device)
            action, _ = self.q_eval(state)
            return action.max(axis=1)[1].item()

    def _update(self):
        if self.learn_step % 100 == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict(), strict=False)

    def sample(self):
        rewards, terminals, actions, states, states_ = \
            self.memory.sample(self.threshold)

        actions = T.tensor(actions).long().to(self.q_eval.device)
        states = T.tensor(states).float().to(self.q_eval.device)
        states_ = T.tensor(states_).float().to(self.q_eval.device)
        rewards = T.tensor(rewards).view(self.threshold).float().to(self.q_eval.device)
        terminals = T.tensor(terminals).view(self.threshold).long().to(self.q_eval.device)

        return actions, states, states_, rewards, terminals

    def learn(self, state, action, state_, reward, done):
        if self.memory.mem_counter < self.threshold:
            self.memory.store(reward, done, action, state, state_)
            return None, None

        self.q_eval.train()
        self.memory.store(reward, done, action, state, state_)
        self.learn_step += 1
        actions, states, states_, rewards, terminals = self.sample()

        indices = np.arange(self.threshold)
        V_s, A_s = self.q_eval(states)
        V_s_, A_s_ = self.q_next(states_)
        V_s_eval, A_s_eval = self.q_eval(states_)

        q_pred = (V_s + (A_s - A_s.mean(dim=1,
                                        keepdim=True)))[indices, actions]
        q_next = (V_s_ + (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = (V_s_eval + (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        q_target = rewards + self.gamma * \
            q_next[indices, q_eval.max(axis=1)[1]] * (1 - terminals)

        loss = self.q_eval.learn(q_pred, q_target)
        self.epsilon = 0.1 if self.epsilon < 0.1 else self.epsilon * 0.99
        self.counter = 0

        self._update()
        return loss.item(), rewards.mean().item()

    def save(self, counter):
        self.q_eval.save(counter)

    def load(self, name, counter):
        self.q_eval.load(name, counter)
