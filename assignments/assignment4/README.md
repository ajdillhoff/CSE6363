# Assignment 4: Reinforcement Learning

## (40 points) Part 1: Q-Learning and Policy Iteration on the Frozen Lake Environment

In this part of the assignment, you will implement a basic version of Q-Learning on the Frozen Lake environment, following the tutorial provided [here](https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/).
The main purpose is to familiarize yourself with using the OpenAI Gym library. In part 2, we will be using a more complex RL method.

**Objective**: Implement a Q-Learning agent to solve the Frozen Lake environment.

**Tasks**:

1. Familiarize yourself with the Frozen Lake environment and its dynamics.
2. Implement the Q-Learning algorithm using the tutorial as a guide.
3. Train your Q-Learning agent on the Frozen Lake environment.
4. Evaluate the performance of your agent and analyze the impact of hyperparameters on the learning process. Specifically, verify the impact of the following hyperparameters:
    - `alpha`: learning rate
    - `gamma`: discount factor
    - `epsilon`: exploration rate
    Test at least 3 different values for each hyperparameter and explain the effect of each hyperparameter on the learning process.
5. Implement the policy iteration algorithm and compare its performance to Q-Learning.

## (60 points) Part 2: Proximal Policy Optimization on an Atari Game Environment

In this part of the assignment, you will adapt the Q-Learning code from Part 1 to an Atari game environment of your choosing. Additionally, you will implement the Proximal Policy Optimization (PPO) algorithm and evaluate it on the same Atari game environment.

**Objective**: Adapt the Q-Learning code to an Atari game environment and implement the PPO algorithm for comparison.

**Tasks**:

1. Choose an Atari game environment from the OpenAI Gym library[1].
2. Adapt the Q-Learning code from Part 1 to work with the chosen Atari game environment.
3. Train your Q-Learning agent on the Atari game environment.
4. Implement the PPO algorithm, following the guidelines provided [here](https://openai.com/research/openai-baselines-ppo/) and [here](https://www.youtube.com/watch?v=MEt6rrxH8W4). **You may use other implementations of PPO as a reference, but you must write your own code. Please cite any references you use.**
5. Train your PPO agent on the same Atari game environment.
6. Compare the performance of the Q-Learning and PPO agents on the chosen Atari game environment by looking at the mean reward over time.