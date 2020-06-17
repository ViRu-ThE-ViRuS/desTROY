from agents.dqn_agent import Agent as AgentDQN
from troy_env import TroyEnv
import numpy as np

if __name__ == '__main__':
    env = TroyEnv()
    rider = AgentDQN(env.observation_space, env.action_space, 1.0, 'dqn0')

    episodes = 50000
    rewards, steps = [], []
    losses = []

    try:
        rider.load('dqn0', 12500)
    except FileNotFoundError:
        pass

    current_stage = 2500

    print('episode : loss_rider1_avg'
          ' : reward_rider1_avg'
          ' : game_steps_avg')

    for episode in range(episodes+1):
        done = False
        total_reward = np.zeros(2)
        episode_steps = 0
        state1 = env.reset()[0]

        while not done:
            action1 = rider.move(state1)

            state_, reward, done, _ = env.step(action1,
                                               # episode % 10 == 0)
                                               True)
            state_ = state_[0]
            loss, reward = rider.learn(state1, action1, state_, reward[0], done)

            state1 = state_
            episode_steps += 1

            if loss:
                # print('trainstep with batch of data...', rider1.threshold)
                if episode > current_stage:
                    rider.threshold += 50
                    current_stage += 500

                rewards.append(reward)
                losses.append(loss)
                steps = []
        steps.append(episode_steps)

        print(f'{episode:5d} : {np.mean(losses):5.2f}'
              f' : {np.array(rewards).mean():5f}'
              f' : {np.array(steps).mean():5f}')

        if episode % 500 == 0 and episode != 0:
            rider.save(episode)
            print('saving model...')
