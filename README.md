# Breakout AI Game: Q Learning and Approximate Q Learning

This repository contains the implementation of two reinforcement learning techniques, Q-Learning and Approximate Q-Learning, applied to the classic Breakout game. The project aims to explore how an AI agent can learn to play and excel at the Breakout game through trial and error without human intervention.

## Project Overview

The Breakout game environment is custom-built using the Pygame library and integrated into a reinforcement learning framework using OpenAI Gym's environment structures. The agent interacts with the game by observing the state of the environment and taking actions that affect the state.

### Techniques Used

- **Q-Learning**: A model-free reinforcement learning algorithm to learn the value of an action in a particular state.
- **Approximate Q-Learning**: Extends Q-Learning by approximating the Q-value function with a feature-based linear representation.

### Environment

The game consists of:
- A paddle controlled by the AI to hit the ball.
- Bricks arranged in rows on the screen, which the ball needs to break.
- A ball that bounces off the paddle and walls, and the goal is to break as many bricks as possible.

## Installation

Requirements:
- Python 3.8 or higher
- Pygame
- Numpy
- Pandas
- Matplotlib
- OpenAI Gym

To set up the project, clone the repository and install the required packages:

```bash
git clone https://github.com/Rupaljain27/Breakout_QLearning
pip install -r requirements.txt
```

## Running the Code

The main script requires six parameters to run, specified in the following order:

1. **initial_alpha** (float): Initial learning rate for the Q-learning algorithm.
2. **min_alpha** (float): Minimum learning rate after decay.
3. **initial_epsilon** (float): Initial exploration rate for the epsilon-greedy strategy.
4. **min_epsilon** (float): Minimum exploration rate after decay.
5. **discount_factor** (float): Discount factor (gamma), which quantifies the difference in importance between future rewards and immediate rewards.
6. **num_episodes** (int): The total number of episodes to run the training for.

### Example

To run the training script with specific parameters, use the following command in your terminal:


Applying the Q-Learning model:
```bash
python agent.py 0.2 0.001 0.9 0.01 0.99 20000
```

Applying Approximate Q-Learning model:

```bash
python approximate_q_learning.py 0.2 0.001 0.8 0.001 0.99 20000
