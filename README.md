# desTROY
Learning to play the TROY game using reinforcement learning methods like policy
gradients actor critic models, Deep Q Learning, Deep Deterministic Actor Critic,
and Advanced Advantage Actor Critic, and technologies like Pytorch and Cython.

### results
#### gen0
starting training on Actor Critic model
<br><img src='res/training.gif' width="250" height="250" /><br>

#### gen1
after 3 hours of training, the model shows significant recognition of game
boundaries and tries to avoid itself by going in circles, using the *actor critic* model

- the leads are promising, but frankly, the training time for this algorithm is
  too slow for me, so i will be coming back to this problem to explore A3C,
  DDPG learning methods later

#### gen2
after 1 hour of training on *dueling double deep q network*, results are much better even though each
episode is taking longer to run (due to bigger batches in the replay buffer)

- trying to speed up learning by giving incentive rewards for survival and
  strategic cutting off of opponent
- this gif is from training phase where i removed the opponent, to see if model
  converges faster... since im using self play, both players show this learned
  knowledge when in dueling mode
<br><img src='res/traniing_2.gif' width="250" height="250" /><br>

#### future work
- add more algorithms
- explore evolutionary strategy in this particular game (integrate my Evolve
  project with desTROY)
