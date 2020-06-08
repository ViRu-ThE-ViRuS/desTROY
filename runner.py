from agents.ac_pg_agent import Agent
from troy_env import TroyEnv
import numpy as np

if __name__ == '__main__':
    env = TroyEnv()
    player1 = Agent(env.observation_space, env.action_space, 1.0)
    player2 = Agent(env.observation_space, env.action_space, 1.0)

    episodes = 1000
    rewards = []
    losses1, losses2 = [], []
    for episode in range(episodes):
        done = False
        total_reward = 0
        state = env.reset()

        while not done:
            action1 = player1.move(state)
            action2 = player2.move(state)

            state_, reward, done, _ = env.step(action1, action2)

            loss1 = player1.learn(state, state_, reward, done)
            loss2 = player2.learn(state, state_, -reward, done)

            state = state_
            total_reward += reward

            losses1.append(loss1)
            losses2.append(loss2)

        rewards.append(total_reward)
        print(f'{episode:5d} : {np.mean(losses1):5.2f}'
              f' : {np.mean(losses2):5.2f} : {np.mean(rewards):5f}')

env = TroyEnv(manual=True)
done = False
rewards = 0
state = env.reset()
env.render()
while not done:
    action1 = player1.move(state)
    state_, reward, done, _ = env.step(action1, action2)

    loss1 = player1.learn(state, state_, reward, done)
    loss2 = player2.learn(state, state_, -reward, done)

    state = state_
    rewards += reward
