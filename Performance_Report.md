# Banana Project Report

## Deep-Q Network Architecture and Optimal Policy Parameters

We implemented the DQL algorithm as originally formulated in the Nature [paper](https://www.nature.com/articles/nature14236). We used a combination of fully connected layers followed by RELU activation functions. The details of the implementation consists of an input layer with 37 nodes (coresponding to the state size) followed by two 64 nodes hidden layers and an output layer with 4 nodes (coresponding to the action size).

During this project there was not a systematic search for an optimal set of hyperparameters. Nevertheless, the following parameters used in computing an optimal policy (provided by default in the Udacity notebook) seem to reproduce the agent's wanted behavior. 

-  `n_episodes (int) = 5000`: The default maximum number of training episodes (Notice that a succesful implementation requires `n_episodes <= 1800` ).
- `max_t (int) = 1000`: maximum number of timesteps per episode.
- `eps_start (float) = 1.0`: starting value of epsilon, for epsilon-greedy action selection.
- `eps_end (float) = 0.01`: minimum value of epsilon.
- `eps_decay (float) = 0.995`: multiplicative factor (per episode) for decreasing epsilon. The decay rule follows the formula `eps = max(eps_end, eps_decay*eps)`.

It is possible that sometimes, depending on some particular random realizations, the agent does not reach an average score of 13 when we average over 100 episodes. One possible explanation for this unwanted behavior is that we break from the training loop as soon as the target score is reached. In order to minimize this, we include an extra parameter to the implementation labeled as `critical_episode`. It is defined as twice the number of episodes it takes to reach an average score of 13. The reason for this is to let the agent become more stable once it reaches the wanted score. Given that the number of training episodes is still less 1800 (after implementing this extra control step), the additional training steps can be seen as valid attempt to make the agent become more succcesful. 

## Deep-Q Learning Parameters

The DQL parameters used to run the `navigation_bananas_dqn.ipynb` notebook are as follows:

- `BUFFER_SIZE = int(1e5)`:  replay buffer size (used for experience replay).
- `BATCH_SIZE = 64`: minibatch size (this is a subset of previous agent experiences chosen randomly during the learning steps).
- `GAMMA = 0.99`: discount factor.
- `TAU = 1e-3`: for soft update of target parameters.
- `LR = 5e-4` : learning rate. 
- `UPDATE_EVERY = 4`: how often to update the network (this is used in order to avoid very correlated experience samples).


## Training Results
The following printout is the result of training the agent with the architecture and paramaters previously defined. Notice that it took around ~500 episodes to reach an average score of 13. The additional steps correspond to the stabilizing steps discussed before.

![results](/Score_vs_Episodes.png)

```
Episode 100    Average Score: 0.36
Episode 200    Average Score: 2.84
Episode 300    Average Score: 7.48
Episode 400    Average Score: 9.92
Episode 500    Average Score: 13.40
Episode 600    Average Score: 14.59
Episode 700    Average Score: 16.06
Episode 800    Average Score: 14.25
Episode 900    Average Score: 14.84
Episode 982    Average Score: 14.32
Environment solved in 982 episodes!    Average Score: 14.32
```

### Untrained (Random) agent
This is the performance of an untrained agent. At this stage, the actions are basically chosen randomly.
![untrained](Gifs/random_agent.gif)

### Trained (Smart) agent
This is an agent performing after the training process.
![trained](Gifs/smart_agent.gif)

## Future Perspectives

A couple potential optimizations that could not only speed up the training process but also improve the overall performance of the agent are either a complete implementation of the [RAINBOW](https://arxiv.org/pdf/1710.02298.pdf) algorithm or perform a systematic parameter search in order to find the optimal set of learning parameters. Both optimizations can be used simultaneously of course and by no means are supposed to be exhaustive given the rapidly evolving state of the art of the field.
