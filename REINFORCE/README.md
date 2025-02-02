
# REINFORCE Implementation for CartPole-v1

This repository contains a simple implementation of the REINFORCE algorithm applied to the [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/) environment from OpenAI Gym. Here, the **REINFORCE** algorithm, a classic on-policy policy gradient method, is used to train an agent to balance a pole on a moving cart.

### What is REINFORCE?

REINFORCE is a straightforward policy gradient method that works as follows:
- **Policy Representation**: The agent’s behavior is represented by a parameterized policy, usually modeled with a neural network. This network outputs a probability distribution over possible actions given the current state.
- **Collecting Trajectories**: The agent interacts with the environment following its current policy, collecting trajectories (sequences of states, actions, and rewards).
- **Computing Returns**: For each time step in the trajectory, the algorithm computes the discounted cumulative reward (return), which estimates the future reward.
- **Policy Update**: The policy parameters are updated by moving in the direction of the gradient of the expected reward. The update rule is given by:

  \[
  \theta \leftarrow \theta + \alpha \nabla_\theta \log \pi(a|s;\theta) G_t
  \]

  where:
  - \( \theta \) are the parameters of the policy,
  - \( \alpha \) is the learning rate,
  - \( \pi(a|s;\theta) \) is the probability of taking action \( a \) in state \( s \),
  - \( G_t \) is the discounted return from time \( t \) onward.

This method encourages actions that lead to higher rewards while discouraging those with lower returns.

## Environment: CartPole-v1

The CartPole-v1 environment is a well-known RL benchmark problem where:
- A pole is attached by an unactuated joint to a cart.
- The cart moves along a frictionless track.
- The goal is to keep the pole balanced by applying forces to move the cart left or right.

## Repository Structure

- `reinforce.py`: The main Python script that implements the REINFORCE algorithm for the CartPole-v1 environment.
- `README.md`: This file, which provides an overview of the project, setup instructions, and details on how to run the code.
