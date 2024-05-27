import numpy as np
from breakout import Breakout
import sys
import csv
from plotting import plotting


class Agent():

    """
    This class implements a Q-learning agent that interacts with the Breakout game environment
    to learn optimal policies based on rewards received from the environment.

    Attributes:
        env (Breakout): Instance of the Breakout game environment.
        q_table (np.array): Table to store the Q-values for each state-action pair.
        epsilon (float): Exploration rate, the probability of choosing a random action.
        learning_rate (float): Learning rate or step size used in the Q-learning update rule.
        discount_factor (float): Discount factor for future rewards in the Q-learning update rule.
        episodes (int): Number of episodes to train the agent.
        alpha_decay (float): Decay rate for the learning rate after each episode.
        epsilon_decay (float): Decay rate for the exploration rate after each episode.
        min_epsilon (float): Minimum value for epsilon after decay.
        min_alpha (float): Minimum value for the learning rate after decay.
        steps_history (list): List to track the number of steps taken in each episode.
        rewards_history (list): List to track the rewards obtained in each episode.
        epsilon_history (list): List to track the value of epsilon over episodes.
        alpha_history (list): List to track the value of alpha over episodes.
    """

    """
    *** Hyperparameters for Q-Learning:
    Epsilon (Exploration Rate): Probability of taking a random action
        Purpose: Controls the trade-off between exploration (choosing a random action) and exploitation (choosing actions based on learned values).
        Typical Range: 0.1 to 0.2 for initial exploration, but it should decay over time to allow more exploitation as the agent learns.
        Decay Strategy: Commonly, epsilon starts high (e.g., 0.9) and is multiplied by a decay rate (e.g., 0.995) every episode or step until it reaches a minimum threshold (e.g., 0.01).

    Learning Rate (Alpha)
        Purpose: Determines to what extent newly acquired information overrides old information.
        Typical Range: 0.1 to 0.5, but the optimal value can depend heavily on the specific task and complexity.
        Consideration: A higher learning rate can cause the learning process to converge faster, but it might lead to unstable learning or oscillation. A lower learning rate ensures more stable convergence but at the risk of slowing down the learning process.    

    Discount Factor (Gamma)
        Purpose: The discount factor determines the importance of future rewards. A higher value facilitates consideration of future rewards, useful for strategic long-term planning.
        Typical Range: 0.8 to 0.99. A value too close to 1 can make the learning process slow to converge, particularly in environments with delayed rewards.
        Environment Specific: In environments where the action taken has long-term effects, a higher discount factor is beneficial. In contrast, if the focus is on immediate rewards, a lower value can be more effective.
    """

    def __init__(self):
        """
        Initializes the agent by setting up the environment, defining the state space,
        initializing the Q-table with random values, and setting initial parameters.
        """
        self.env = Breakout(render_mode='human')
        self.env.reset()

        """
        ***** State Space:
        Paddle Position: 10 discrete states
        Ball X Position: 32 discrete states (assuming a discretization of the window width into 20-pixel segments)
        Ball Y Position: 24 discrete states (assuming a discretization of the window height into 20-pixel segments)
        Ball Velocity X: 2 discrete states (left or right)
        Ball Velocity Y: 2 discrete states (up or down)
        This gives a product of states:
        num_states = 10 x 32 x 24 x 2 x 2 = 30720 states
        
        """
        # Define the size of the state space based on environment's settings
        num_states = (10 * 32 * 24 * 2 * 2)

        """
        Some common approaches to initializing Q-tables:
        1. Zero Initialization:
        Approach: All entries in the Q-table are set to zero.
        
        2. Random Initialization:
        Approach: Entries are initialized to random values, typically within a specific range.
        """
        # Initialize Q-table with random values within a specified range
        self.q_table = np.random.uniform(
            low=-0.5, high=0.0, size=(num_states, self.env.action_space.n))

        # Set initial values for learning parameters
        self.epsilon = 0.9
        self.learning_rate = 0.2   # Alpha
        self.discount_factor = 0.99    # Gamma
        self.episodes = 20000
        self.alpha_decay = 0.9994703085993939
        self.epsilon_decay = 0.9995501202592184
        self.min_epsilon = 0.001
        self.min_alpha = 0.01

        # Initialize histories for analysis and plotting
        self.steps_history = []
        self.rewards_history = []
        self.epsilon_history = []
        self.alpha_history = []

    def train(self, file_name="training_data",  show_every=500, render_last_n=5):
        """
        Trains the agent over a number of episodes to learn from interactions with the environment.
        Rewards and steps per episode are recorded, and the agent's policy (Q-table) is updated based on experiences.

        Args:
            show_every (int): Interval to display average results to the console.
            render_last_n (int): Number of episodes from the end to start rendering the environment.
        """

        # Set up CSV files to log training results
        with open(file_name + '_rewards.csv', 'w', newline='') as rewards_file, \
                open(file_name + '_steps.csv', 'w', newline='') as steps_file, \
            open(file_name+'_epsilon.csv', 'w', newline='') as epsilon_file, \
            open(file_name+'_alpha.csv', 'w', newline='') as alpha_file:

            rewards_writer = csv.writer(rewards_file)
            steps_writer = csv.writer(steps_file)
            rewards_writer.writerow(['Episode', 'Reward'])
            steps_writer.writerow(['Episode', 'Steps'])

            epsilon_writer = csv.writer(epsilon_file)
            alpha_writer = csv.writer(alpha_file)
            epsilon_writer.writerow(['Episode', 'Epsilon'])
            alpha_writer.writerow(['Episode', 'Alpha'])

            episode_rewards = []
            steps_per_episode = []
            # Train over the specified number of episodes
            for episode in range(self.episodes):
                # Adjust learning rate and exploration rate based on decay
                self.learning_rate = max(
                    self.min_alpha, self.learning_rate * self.alpha_decay)
                self.epsilon = max(
                    self.min_epsilon, self.epsilon * self.epsilon_decay)

                # Reset environment and initialize variables
                total_reward = 0
                done = False
                steps = 0
                observation = self.env.reset()

                # Record the current values of epsilon and learning rate
                self.alpha_history.append(self.learning_rate)
                self.epsilon_history.append(self.epsilon)
                epsilon_writer.writerow([episode, self.epsilon])
                alpha_writer.writerow([episode, self.learning_rate])

                # Episode loop
                while not done:
                    # Render environment for the last few episodes
                    if episode >= (self.episodes - render_last_n):
                        self.env.render()

                    # Decide action based on exploration or exploitation
                    if np.random.random() < self.epsilon:
                        # print("Exploration")
                        action = self.env.action_space.sample()
                    else:
                        # print("Exploitation")
                        action = np.argmax(self.q_table[observation])

                    # Take action and observe results
                    new_observation, reward, done = self.env.step(action)
                    total_reward += reward

                    # Update Q-table based on the observed transition
                    old_value = np.max(self.q_table[observation])
                    future_optimal_value = np.max(
                        self.q_table[new_observation])
                    new_value = (1 - self.learning_rate) * old_value + self.learning_rate * \
                        (reward + self.discount_factor * future_optimal_value)

                    self.q_table[observation, action] = new_value

                    # Update observation and steps
                    steps += 1
                    observation = new_observation

                # Record results and log to CSV files
                episode_rewards.append(total_reward)
                steps_per_episode.append(steps)
                rewards_writer.writerow([episode, total_reward])
                steps_writer.writerow([episode, steps])

                self.rewards_history.append(total_reward)
                self.steps_history.append(steps)

                # Display average results periodically every 500 episodes
                if episode % show_every == 0:
                    average_reward = sum(
                        episode_rewards[-show_every:]) / show_every
                    average_steps = sum(
                        steps_per_episode[-show_every:]) / show_every

                    print(
                        f"Episode: {episode}, Average Reward: {average_reward}, Average Steps: {average_steps}")

        print("Training Finished...")
        return self.alpha_history, self.epsilon_history, self.rewards_history, self.steps_history

    def test(self, episodes=10):
        """
        Evaluates the trained agent by running a specified number of episodes without further training.
        The agent uses its learned policy to play the game, and performance is reported.

        Args:
            episodes (int): Number of episodes to run for testing.
        """
        for episode in range(episodes):
            observation = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = np.argmax(self.q_table[observation])
                observation, reward, done = self.env.step(action)
                total_reward += reward
                self.env.render()
            print(f"Test Episode: {episode}, Total reward: {total_reward}")


if __name__ == "__main__":

    initial_alpha = float(sys.argv[1])
    min_alpha = float(sys.argv[2])
    initial_epsilon = float(sys.argv[3])
    min_epsilon = float(sys.argv[4])
    discount_factor = float(sys.argv[5])
    num_episodes = int(sys.argv[6])

    """
    *** Tested the agent based on the below range of Hyperparameters for Q-Learning
     initial_alpha in [0.2, 0.1, 0.05, 0.01, 0.005]
     min_alpha in [0.001, 0.0005, 0.0001, 0.00005]
     initial_epsilon in [0.9, 0.8, 0.7, 0.6, 0.5]
     min_epsilon in [0.1, 0.05, 0.01, 0.005, 0.001]
     discount_factor in [0.99, 0.95, 0.9, 0.85, 0.8]
    """

    agent = Agent()

    # Implement the decay logic for alpha and epsilon
    agent.episodes = num_episodes
    agent.alpha_decay = (min_alpha / initial_alpha) ** (1 / agent.episodes)
    agent.epsilon_decay = (
        min_epsilon / initial_epsilon) ** (1 / agent.episodes)
    agent.epsilon = initial_epsilon
    agent.learning_rate = initial_alpha   # Alpha
    agent.discount_factor = discount_factor    # Gamma
    agent.min_epsilon = min_epsilon
    agent.min_alpha = min_alpha

    print(f"Init Alpha: {initial_alpha}, Min Alpha: {min_alpha}, Init Epsilon: {initial_epsilon}, Min Epsilon: {min_epsilon}, Alpha Decay: {agent.alpha_decay}, Epsilon Decay: {agent.epsilon_decay}, Discount Factor: {discount_factor}")

    file_name = 'Q-Learning(' + str(initial_alpha) + ',' + str(min_alpha) + ',' + str(
        initial_epsilon) + ',' + str(min_epsilon) + ',' + str(discount_factor) + ')'

    alpha_history, epsilon_history, reward_history, steps_history = agent.train(file_name=file_name)

    # Plot the learning curves and metrics
    plot = plotting()
    plot.plot_metrics(output_filename=file_name,
                      steps_file=file_name + '_steps.csv', epsilon_file=file_name + '_epsilon.csv', alpha_file=file_name + '_alpha.csv')
    plot.plot_learning_curve(output_filename=file_name + '_Learning_Curve.png',
                             file_name=file_name + '_rewards.csv')

    # test_decision = input("Training completed. Would you like to start testing the agent? (yes/no): ")
    # if test_decision.lower() == 'yes':
    #     agent.test()
    # else:
    #     print("Testing aborted.")
