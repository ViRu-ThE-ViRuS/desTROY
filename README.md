# desTROY
Learning to play the TROY game using reinforcement learning methods like policy
gradients actor critic models, Deep Q Learning, Deep Deterministic Actor Critic,
and Advanced Advantage Actor Critic, and technologies like Pytorch and Cython.

### results
#### gen1
after 3 hours of training, the model shows significant recognition of game
boundaries and tries to avoid itself by going in circles, using the Advantage
Actor Critic model

- the leads are promising, but frankly, the training time for this algorithm is
  too slow for me, so i will be coming back to this problem to explore A3C,
  DDPG learning methods later

#### gen2
after 1 hour of training on dqn, results are much better even though each
episode is taking longer to run (due to bigger batches in the replay buffer)

- trying to speed up learning by giving incentive rewards for survival and
  strategic cutting off of opponent

![](res/training.gif)

#### future work
- cythonize the code
- add more algorithms
- explore evolutionary strategy in this particular game (integrate my Evolve
  project with desTROY)

#### updates
- added batching in train steps
- added missing `.eval()` when using model to choose an action
