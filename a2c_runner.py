from agents.pg_ac_agent import Agent as AgentPG
from troy_env import TroyEnv
import numpy as np

if __name__ == '__main__':
    env = TroyEnv()
    rider = AgentPG(env.observation_space, env.action_space, 1.0, 'pg0')

    episodes = 50000
    rewards, steps = [], []
    losses = []

    try:
        rider.load('pg0', -1)
    except FileNotFoundError:
        print('no save loaded...')

    current_stage = 2500

    print('episode : loss_rider1_avg'
          ' : reward_rider1_avg'
          ' : game_steps_avg')

    for episode in range(episodes+1):
        done = False
        total_reward = np.zeros(2)
        episode_steps = 0
        state = env.reset()[0]

        while not done:
            action, actionprobs = rider.move(state)

            state_, reward, done, _ = env.step(action,
                                               # episode % 10 == 0)
                                               True)

            state_ = state_[0]
            loss, reward = rider.learn(state, action, state_, reward[0], done, actionprobs)

            state = state_
            episode_steps += 1

            rewards.append(reward)

        steps.append(episode_steps)
        losses.append(loss)

        print(f'{episode:5d} : {np.mean(losses):5.2f}'
              f' : {np.array(rewards).mean():5f}'
              f' : {np.array(steps).mean():5f}')

        if episode % 500 == 0 and episode != 0:
            rider.save(episode)
            print('saving model...')

            rewards = []
            steps = []
            losses = []
