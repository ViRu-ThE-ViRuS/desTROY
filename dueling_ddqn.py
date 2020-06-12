from agents.pg_ac_agent import Agent as AgentPG
from agents.dqn_agent import Agent as AgentDQN
from troy_env import TroyEnv
import numpy as np

if __name__ == '__main__':
    env = TroyEnv()
    rider1 = AgentDQN(env.observation_space, env.action_space, 1.0, 'dqn0')
    rider2 = AgentDQN(env.observation_space, env.action_space, 1.0)

    episodes = 50000
    rewards, steps = [], []
    losses1, losses2 = [], []

    try:
        rider1.load('dqn0', 12500)
        rider2.load('dqn0', 12500)
    except FileNotFoundError:
        pass

    current_stage = 2500

    print('episode : loss_rider1_avg'
          # ' : loss_rider2_avg'
          ' : reward_rider1_avg'
          ' : game_steps_avg')

    for episode in range(episodes+1):
        done = False
        total_reward = np.zeros(2)
        episode_steps = 0
        (state1, state2) = env.reset()

        while not done:
            action1 = rider1.move(state1)
            action2 = rider2.move(state2)

            (state1_, state2_), reward, done, _ = env.step(action1, action2,
                                                           # episode % 10 == 0)
                                                           True)

            loss1, reward = rider1.learn(state1, action1, state1_, reward[0], done)
            # loss2 = rider2.learn(state2, state2_, reward[1], done, actionprobs2)

            state1 = state1_
            state2 = state2_
            episode_steps += 1

            if loss1:
                # print('trainstep with batch of data...', rider1.threshold)
                if episode > current_stage:
                    rider1.threshold += 50
                    current_stage += 500

                rewards.append(reward)
                losses1.append(loss1)
                steps = []
        steps.append(episode_steps)

        print(f'{episode:5d} : {np.mean(losses1):5.2f}'
              # f' : {np.mean(losses2):5.2f}'
              f' : {np.array(rewards).mean():5f}'
              f' : {np.array(steps).mean():5f}')

        if episode % 500 == 0 and episode != 0:
            rider1.save(episode)
            rider2.load(rider1.q_eval.model_name, episode)
            print('saving model...')