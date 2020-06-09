from agents.pg_ac_agent import Agent
from troy_env import TroyEnv
import numpy as np

if __name__ == '__main__':
    env = TroyEnv()
    rider1 = Agent(env.observation_space, env.action_space, 1.0, 'v0')
    rider2 = Agent(env.observation_space, env.action_space, 1.0)

    episodes = 10000
    rewards, steps = [], []
    losses1, losses2 = [], []

    print('episode : loss_rider1_avg'
          # ' : loss_rider2_avg'
          ' : reward_rider1_avg'
          ' : reward_rider2_avg : game_steps_avg')
    for episode in range(episodes):
        done = False
        total_reward = np.zeros(2)
        episode_steps = 0
        (state1, state2) = env.reset()

        while not done:
            action1 = rider1.move(state1)
            action2 = rider2.move(state2)

            (state1_, state2_), reward, done, _ = env.step(action1, action2,
                                                           episode % 15 == 0)

            loss1 = rider1.learn(state1, state1_, reward[0], done)
            # loss2 = rider2.learn(state2, state2_, reward[1], done)

            state1 = state1_
            state2 = state2_
            total_reward += reward

            episode_steps += 1
            losses1.append(loss1)
            # losses2.append(loss2)

        rewards.append(total_reward)
        steps.append(episode_steps)
        print(f'{episode:5d} : {np.mean(losses1):5.2f}'
              # f' : {np.mean(losses2):5.2f}'
              f' : {np.array(rewards)[:,0].mean():5f}'
              f' : {np.array(rewards)[:,1].mean():5f}'
              f' : {np.array(steps).mean():5f}')

        if episode % 1000 == 0:
            rider1.save(episode)
            rider2.load(rider1.actorcritic.model_name, episode)
