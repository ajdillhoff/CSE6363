# Reinforcement Learning Assignment

## (40 points) Part 1: Q-Learning and Policy Iteration on the Frozen Lake Environment

In this part of the assignment, you will implement a basic version of Q-Learning on the Frozen Lake environment, following the tutorial provided [here](https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/).
The main purpose is to familiarize yourself with using the RL Gym library. In part 2, we will be using an environment of your choosing.

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

## (60 points) Part 2: Deep Q-Learning on an Atari Game Environment

In this part of the assignment, you will adapt the Q-Learning code from Part 1 to Deep Q-Learning in order to play an Atari game environment of your choosing. The primary difference being that the Q-Learning agent will now use a neural network to approximate the Q-values.

**Objective**: Adapt the Q-Learning code to use Deep Q-Learning to an Atari game environment.

**Tasks**:

1. Choose an Atari game environment from the RL Gym library[1].
2. Adapt the Q-Learning code from Part 1 to work with the chosen Atari game environment.
3. Train your Q-Learning agent on the Atari game environment.