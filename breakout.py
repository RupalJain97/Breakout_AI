import random
import gym
from gym import spaces
import numpy as np
import pygame

class Breakout(gym.Env):
    """
    Custom implementation of the Breakout game as a Gym environment for reinforcement learning.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode='human'):
        """
        Initializes the Breakout game environment.
        
        Args:
        render_mode (str): The mode to render the game in ('human' for screen or 'rgb_array' for numpy array).
        """
        super(Breakout, self).__init__()
        self.action_space = spaces.Discrete(3)  # Actions: [0: Stay, 1: Move Left, 2: Move Right]
        
        # Set up game dimensions and color
        self.window_width = 640
        self.window_height = 480
        self.background_color = (0, 0, 0)  

        self.observation_space = [self.window_width, (self.window_height * self.window_width)]

        self.done = False
        self.render_mode = render_mode
        self.initialize_game()
    
    # ************* ENVIRONMENT INITIALIZATION FUNCTIONS *************
    def initialize_game(self):
        """
        Sets up the game by initializing bricks, the paddle, and the ball.
        """
        self.initialize_bricks()
        self.initialize_paddle()
        self.initialize_ball()

        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption('Breakout Game')

    def initialize_bricks(self):
        """
        Initializes bricks in a grid layout at the top of the screen.
        """
        self.brick_rows = 3
        self.brick_cols = 10
        self.brick_width = 62
        self.gap = 2
        self.brick_height = 20
        self.bricks = np.ones((self.brick_rows, self.brick_cols), dtype=bool)
        self.num_bricks = np.sum(self.bricks)

    def initialize_ball(self):
        """
        Sets the initial position and velocity of the ball.
        """
        self.ball_color = [121, 189, 232]  # Blue color
        self.ball_size = (10, 10)  # Width and height of the ball
        self.ball_position = np.array([self.paddle_x_position + self.paddle_width // 2, self.paddle_y_position - self.ball_size[1]], dtype=np.float64)

        # Use starting_direction for both horizontal and vertical movements
        horizontal_direction = random.uniform(-1, 1)  # Random choice for left or right direction
        vertical_direction = -1  # Starting direction of the ball is upwards (towards bricks)
        self.ball_velocity = np.array([horizontal_direction, vertical_direction], dtype=np.float64)

    def initialize_paddle(self):
        """
        Initializes the paddle's position and dimensions.
        """
        self.paddle_color = [255, 255, 255]  # Blue color
        self.paddle_x_position = self.window_width // 2 - 50
        self.paddle_y_position = self.window_height - 20  # Position it 30 pixels from the bottom
        self.paddle_width = 100
        self.paddle_height = 10

    def reset(self, approx_q = False, seed=None ):
        """
        Resets the environment state for a new episode.

        Args:
        approx_q (bool): If True, returns a feature-based state for Approximate Q-Learning.
        seed (int): Seed for random number generator.

        Returns:
        The initial state of the environment.
        """
        super().reset(seed=seed)  
        self.initialize_game()
        self.done = False
        return self._get_obs(approx_q)

    def _get_obs(self, approx_q = False):
        """
        Generates an observation of the current state of the environment.
        
        Args:
        approx_q (bool): If True, returns a feature-based state for Approximate Q-Learning.
        
        Returns:
        Either an integer index or a dictionary of features, based on `approx_q`.
        """

        # Discretizing the paddle x-position
        paddle_index = int((self.paddle_x_position / self.window_width) * 10)  # Normalize and scale

        # Discretizing the ball observation
        ball_x = int((self.ball_position[0] / self.window_width) * 32)
        ball_y = int((self.ball_position[1] / self.window_width) * 24)
        
        # Discretize velocity of the ball
        vel_x_discrete = 0 if self.ball_velocity[0] < 0 else 1
        vel_y_discrete = 0 if self.ball_velocity[1] < 0 else 1

        if approx_q:
            # Return a dictionary of features
            return {
                'paddle_index': paddle_index,
                'ball_x': ball_x,
                'ball_y': ball_y,
                'vel_x_discrete': vel_x_discrete,
                'vel_y_discrete': vel_y_discrete
            }
        
        # Combine discretized positions and velocities into a single integer for Q-table indexing
        observation_index = paddle_index + (ball_x * 10) + (ball_y * 320) + (vel_x_discrete * 7680) + (vel_y_discrete * 15360)

        return observation_index

    def render(self, mode='human'):
        """
        Renders the current game state to the screen or as an RGB array.

        Args:
        mode (str): Rendering mode ('human' or 'rgb_array').

        Returns:
        None or a numpy array of the screen, based on the mode.
        """
        if self.screen is None and self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.screen.fill(self.background_color)
        
        # Draw the paddle
        pygame.draw.rect(self.screen, self.paddle_color, (self.paddle_x_position, self.paddle_y_position, self.paddle_width, self.paddle_height))
        
        # Draw the ball
        ball_radius = self.ball_size[0]
        pygame.draw.circle(self.screen, self.ball_color, (int(self.ball_position[0]), int(self.ball_position[1])), ball_radius)
        
        colors = [(204, 0, 204), (102, 0, 204), (204, 204, 0)]  # Different colors for each row

        # Draw the bricks 
        for i in range(self.bricks.shape[0]):  
            for j in range(self.bricks.shape[1]):  
                if self.bricks[i, j]:
                    brick_x_position = j * (self.brick_width + self.gap)
                    brick_y_position = i * (self.brick_height + self.gap)
                    pygame.draw.rect(self.screen, colors[i % len(colors)], pygame.Rect(brick_x_position, brick_y_position, self.brick_width, self.brick_height))
        
        pygame.display.flip()  # Update the display

        return self.screen if mode == 'rgb_array' else None


    # ************* ENVIRONMENT STEP FUNCTION *************
    def step(self, action, approx_q = False):
        """
        Processes a single timestep within the environment using the given action.

        Args:
        action (int): The action taken by the agent.
        approx_q (bool): If True, uses feature-based state representation.

        Returns:
        tuple: A tuple containing the new state, reward, and a boolean indicating if the episode has ended.
        """
        self.move_paddle(action)
        self.move_ball()
        reward = 0.5    # Survival reward

        """
        Checks for collision between the ball and the paddle, reversing the ball's direction on collision.
        """
        # Calculate positions
        ball_bottom = self.ball_position[1] + self.ball_size[1] // 2
        paddle_top = self.paddle_y_position

        # Calculate horizontal overlap
        ball_center_x = self.ball_position[0]
        paddle_left = self.paddle_x_position
        paddle_right = self.paddle_x_position + self.paddle_width

        # Check if the center of the ball is within the width of the paddle
        if paddle_left <= ball_center_x <= paddle_right:
            # Check if the bottom of the ball is touching or overlapping the top of the paddle
            if ball_bottom >= paddle_top:
                reward += 1
                self.ball_velocity[1] = -self.ball_velocity[1]  # Reverse the vertical direction

                # Adjust the ball position to sit just above the paddle if overlapping
                self.ball_position[1] = paddle_top - self.ball_size[1] // 2

        # Handle collisions with bricks
        reward += self.check_brick_collisions()
        
        if self.num_bricks == 0:
            self.done = True
            reward += 10  # Bonus for clearing all bricks

        # Check if the ball has fallen below the paddle (game over condition)
        if self.ball_position[1] > self.window_height:
            self.done = True
            reward -= 50  # Penalty for losing the ball

        observation = self._get_obs(approx_q)
        return (observation, reward, self.done)

    def move_paddle(self, action):
        """
        Moves the paddle according to the specified action.

        Args:
        action (int): The action to take (0 = stay, 1 = move left, 2 = move right).
        """
        if action == 0:  # No movement
            pass
        elif action == 1 and self.paddle_x_position > 0:  # Move the paddle left
            self.paddle_x_position -= 10
        elif action == 2 and self.paddle_x_position < self.window_width - self.paddle_width:   # Move the paddle right
            self.paddle_x_position += 10

    def move_ball(self):
        """
        Updates the ball's position based on its current velocity.
        """
        # Move the ball by its current velocity
        new_x = self.ball_position[0] + self.ball_velocity[0]
        new_y = self.ball_position[1] + self.ball_velocity[1]

        # Check for collisions with the left or right walls
        if new_x < 0 or new_x > self.window_width - self.ball_size[0]:
            # Reverse the horizontal direction
            self.ball_velocity[0] = -self.ball_velocity[0] 

            # Adjusting x position within bounds
            new_x = max(0, min(new_x, self.window_width - self.ball_size[0]))  

        # Check for collisions with the top wall
        if new_y < 0:
            # Reverse the vertical direction
            self.ball_velocity[1] = -self.ball_velocity[1] 
            new_y = max(0, new_y)  # Adjusting position within bounds

        # Update the ball's position
        self.ball_position[0] = new_x
        self.ball_position[1] = new_y

    def check_brick_collisions(self):
        """
        Checks for collisions between the ball and bricks, updating the state of the bricks and providing additional reward.

        Returns:
        float: The reward obtained from hitting bricks.
        """
        reward = 0
        ball_rect = pygame.Rect(int(self.ball_position[0]), int(self.ball_position[1]), *self.ball_size)

        for i in range(self.bricks.shape[0]):
            for j in range(self.bricks.shape[1]):

                # Check if the brick is still active
                if self.bricks[i, j]:  
                    brick_x = j * (self.brick_width + self.gap)
                    brick_y = i * (self.brick_height + self.gap)
                    brick_rect = pygame.Rect(brick_x, brick_y, self.brick_width, self.brick_height)
                    if ball_rect.colliderect(brick_rect):
                        self.num_bricks -= 1 # Reduce the number of active bricks
                        self.bricks[i, j] = False  # Deactivate the brick
                        reward += 1  # Reward for hitting a brick
        return reward
    