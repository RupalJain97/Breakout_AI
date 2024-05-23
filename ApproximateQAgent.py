import numpy as np
import random
from breakout import Breakout
import sys
import csv
from plotting import plotting


class ApproximateQAgent:
    """
    This class implements an Approximate Q-Learning agent. This agent uses a function approximator
    to estimate the Q-values, which allows it to handle large or continuous state spaces by generalizing
    from similar states to others using features of those states.

    Attributes:
        num_actions (int): Number of possible actions in the environment.
        learning_rate (float): Rate at which the Q-values are updated during training.
        discount_factor (float): Factor to discount future rewards, reflecting their potential lesser value than immediate rewards.
        epsilon (float): Probability to choose a random action. Facilitates exploration of the action space.
        weights (np.array): Array of weights for the feature extractor's output to compute Q-values.
        feature_extractor (object): Object to transform state and action into a feature vector.
        alpha_history, epsilon_history, rewards_history, steps_history (list): History tracking for analysis.
    """

    def __init__(self, num_actions, learning_rate=0.2, discount_factor=0.99, epsilon=0.9, feature_extractor=None):

        # Set initial values for learning parameters
        self.num_actions = num_actions
        self.episodes = 20000
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.weights = np.zeros(feature_extractor.feature_size())
        self.feature_extractor = feature_extractor
        self.epsilon_decay = 0.9999
        self.alpha_decay = 0.9999

        # History tracking
        self.alpha_history = []
        self.epsilon_history = []
        self.rewards_history = []
        self.steps_history = []

    def predict(self, state, action):
        """
        Predicts the Q-value for a given state and action by computing the dot product
        of the weights and the extracted features of the state-action pair.

        Args:
            state: The current state of the environment.
            action: The action taken in the current state.

        Returns:
            The predicted Q-value.
        """
        features = self.feature_extractor.extract_features(state, action)
        qValue = np.dot(self.weights, features)
        return qValue

    def choose_action(self, state):
        """
        Chooses the next action to take based on the current state using an epsilon-greedy policy.
        With probability epsilon, a random action is chosen (exploration), otherwise, the action
        with the highest predicted Q-value is chosen (exploitation).

        Args:
            state: The current state of the environment.

        Returns:
            The action chosen.
        """
        if random.random() < self.epsilon:  # Explore with probability epsilon
            return random.randint(0, self.num_actions - 1)
        else:  # Exploit best known action
            q_values = [self.predict(state, action)
                        for action in range(self.num_actions)]
            return np.argmax(q_values)

    def update(self, state, action, reward, next_state, done):
        """
        Updates the weights based on the received reward and the temporal difference error.

        Args:
            state: The current state from which the action was taken.
            action: The action taken.
            reward: The reward received after taking the action.
            next_state: The next state reached after the action.
            done: Whether the episode has ended.
        """
        features = self.feature_extractor.extract_features(state, action)
        q_value_current = self.predict(state, action)

        max_q_next = max(self.predict(next_state, a)
                         for a in range(self.num_actions)) if not done else 0

        target = reward + self.discount_factor * max_q_next
        td_error = target - q_value_current

        self.weights += self.learning_rate * td_error * features


class FeatureExtractor:
    """
    Extracts features from the state and action which are used by the Approximate Q-Agent to predict Q-values.
    """

    def __init__(self, env):
        self.env = env

    def feature_size(self):
        """ Returns the size of the feature vector. """
        return 5

    def extract_features(self, state, action):
        """
        Extracts and returns features from the state and action.

        Args:
            state: The current state.
            action: The action taken.

        Returns:
            A feature vector derived from the state and action.
        """
        features = np.zeros(5)
        features[0] = state['paddle_index'] / 10.0
        features[1] = state['ball_x'] / 32.0
        features[2] = state['ball_y'] / 24.0
        features[3] = state['vel_x_discrete']
        features[4] = state['vel_y_discrete']
        return features


def train_agent(env, agent, file_name="trainign_data", show_every=500, render_last_n=5, epsilon_decay=0.9999, alpha_decay=0.9999):
    """
    Trains the Approximate Q-Agent by interacting with the environment.

    Args:
        env: The game environment.
        agent: The Approximate Q-Agent to be trained.
        show_every (int): How often to output training progress.
        epsilon_decay (float): Decay rate for epsilon, to reduce the exploration over time.
        alpha_decay (float): Decay rate for learning rate, to stabilize learning as it progresses.

    Returns:
        Tuple of lists containing the history of alpha values, epsilon values, rewards, and steps.
    """

    with open(file_name+'_rewards.csv', 'w', newline='') as rewards_file, \
            open(file_name+'_steps.csv', 'w', newline='') as steps_file, \
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

        total_rewards = []
        total_steps = []
        # Train over the specified number of episodes
        for episode in range(agent.episodes):
            # Adjust learning rate and exploration rate based on decay
            agent.learning_rate *= alpha_decay
            agent.epsilon *= epsilon_decay

            # Reset environment and initialize variables
            state = env.reset(approx_q=True)
            done = False
            episode_rewards = 0
            steps_per_episode = 0

            # Record the current values of epsilon and learning rate
            agent.alpha_history.append(agent.learning_rate)
            agent.epsilon_history.append(agent.epsilon)
            epsilon_writer.writerow([episode, agent.epsilon])
            alpha_writer.writerow([episode, agent.learning_rate])

            # Episode loop
            while not done:
                # Render environment for the last few episodes
                if episode >= (agent.episodes - render_last_n):
                    env.render()

                # Take action and observe results
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action, True)
                agent.update(state, action, reward, next_state, done)

                state = next_state
                episode_rewards += reward
                steps_per_episode += 1

            # Record results and log to CSV files
            total_rewards.append(episode_rewards)
            total_steps.append(steps_per_episode)
            rewards_writer.writerow([episode, episode_rewards])
            steps_writer.writerow([episode, steps_per_episode])

            agent.rewards_history.append(episode_rewards)
            agent.steps_history.append(steps_per_episode)

            # Display average results periodically every 500 episodes
            if episode % show_every == 0:
                average_reward = sum(total_rewards[-show_every:]) / show_every
                average_steps = sum(total_steps[-show_every:]) / show_every

                print(
                    f"Episode: {episode}, Average Reward: {average_reward}, Average Steps: {average_steps}")

    print("Training Finished... ")
    return agent.alpha_history, agent.epsilon_history, agent.rewards_history, agent.steps_history


def test_agent(env, agent, num_episodes=10):
    """
    Tests the trained Approximate Q-Agent by running a specified number of episodes and tracking performance.

    Args:
        env: The game environment.
        agent: The trained Approximate Q-Agent.
        num_episodes (int): Number of episodes to run for testing.

    Prints:
        Total reward and steps per episode during testing.
    """
    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset(approx_q=True)
        done = False
        total_reward = 0
        total_steps = 0

        while not done:
            action = np.argmax([agent.predict(state, action)
                                for action in range(action_size)])
            next_state, reward, done = env.step(action, approx_q=True)
            env.render()
            state = next_state
            total_reward += reward
            total_steps += 1

        total_rewards.append(total_reward)
        print(
            f"Episode {episode + 1}: Total Reward = {total_reward}, Total Steps = {total_steps}")

    average_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")
    print("Testing Finished... total rewards: ", total_rewards)


if __name__ == "__main__":

    initial_alpha = float(sys.argv[1])
    min_alpha = float(sys.argv[2])
    initial_epsilon = float(sys.argv[3])
    min_epsilon = float(sys.argv[4])
    discount_factor = float(sys.argv[5])
    num_episodes = int(sys.argv[6])

    env = Breakout()
    action_size = env.action_space.n
    feature_extractor = FeatureExtractor(env)
    agent = ApproximateQAgent(num_actions=action_size,
                              feature_extractor=feature_extractor)

    agent.learning_rate = initial_alpha
    agent.epsilon = initial_epsilon
    agent.discount_factor = discount_factor
    agent.episodes = num_episodes
    agent.alpha_decay = (min_alpha / initial_alpha) ** (1 / agent.episodes)
    agent.epsilon_decay = (
        min_epsilon / initial_epsilon) ** (1 / agent.episodes)

    agent.min_epsilon = min_epsilon
    agent.min_alpha = min_alpha
    print(f"Init Alpha: {initial_alpha}, Min Alpha: {min_alpha}, Init Epsilon: {initial_epsilon}, Min Epsilon: {min_epsilon}, Alpha Decay: {agent.alpha_decay}, Epsilon Decay: {agent.epsilon_decay}, Discount Factor: {discount_factor}, Num Episodes: {num_episodes}")

    file_name = 'ApproxQ-Learning(' + str(initial_alpha) + ',' + str(min_alpha) + ',' + str(
        initial_epsilon) + ',' + str(min_epsilon) + ',' + str(discount_factor) + ')'
    alpha_history, epsilon_history, reward_history, steps_history = train_agent( env, agent,file_name=file_name, epsilon_decay=agent.epsilon_decay, alpha_decay=agent.alpha_decay)
    

    plot = plotting()
    plot.plot_metrics(output_filename=file_name,
                      steps_file=file_name + '_steps.csv', epsilon_file=file_name + '_epsilon.csv', alpha_file=file_name + '_alpha.csv')
    plot.plot_learning_curve(output_filename=file_name + '_Learning_Curve.png',file_name=file_name  + '_rewards.csv')
    
    test_decision = input("Training completed. Would you like to start testing the agent? (yes/no): ")
    if test_decision.lower() == 'yes':
        test_agent(env, agent)
    else:
        print("Testing aborted.")
    