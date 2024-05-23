import numpy as np
import sys
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os

# Adjust matplotlib settings for different platforms, particularly for high DPI displays
os.environ['QT_DEVICE_PIXEL_RATIO'] = '1'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
os.environ['QT_SCREEN_SCALE_FACTORS'] = '1'
os.environ['QT_SCALE_FACTOR'] = '1'

class plotting:
    def __init__(self):
        """
        Initializes the plotting class.
        """
        pass

    def plot_metrics(self, output_filename="training_metrics", steps_file='training_steps.csv', epsilon_file='training_epsilon.csv', alpha_file='training_alpha.csv'):
        """
        Generates and saves plots for training metrics including steps, epsilon values, and alpha values.

        Args:
            output_filename (str): Base name for the output plot files.
            steps_file (str): Filename for the CSV containing step data.
            epsilon_file (str): Filename for the CSV containing epsilon data.
            alpha_file (str): Filename for the CSV containing alpha data.
        """
        
        # Load data from CSV files
        steps_df = pd.read_csv(steps_file)
        epsilon_df = pd.read_csv(epsilon_file)
        alpha_df = pd.read_csv(alpha_file)

        # Create a figure with high DPI
        plt.figure(figsize=(12, 8), dpi=100)

        # Plot alpha values
        plt.subplot(1, 2, 1)
        plt.plot(alpha_df['Episode'], alpha_df['Alpha'], color='c', label='Alpha (Learning Rate)')
        plt.title('Alpha over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Alpha')
        plt.grid(True)
        plt.legend()

        # Plot epsilon values
        plt.subplot(1, 2, 2)
        plt.plot(epsilon_df['Episode'], epsilon_df['Epsilon'], color='c', label='Epsilon (Exploration Rate)')
        plt.title('Epsilon over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_filename + '.png')
        plt.close()

        # Aggregate steps data to reduce noise
        sample_size = 100
        
        # Prepare sampled data
        sampled_steps = steps_df['Steps'].groupby(steps_df.index // sample_size).mean()

        # Plot smoothed steps data
        plt.figure(figsize=(10, 5), dpi=100)
        plt.plot(sampled_steps.index * sample_size, sampled_steps, marker='o', linestyle='-', color='c', label='Average Steps')
        plt.title('Steps Per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Average Steps')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_filename + '-steps.png')
        plt.close()

    def plot_learning_curve(self, output_filename='learning_curve.png', file_name='training_rewards.csv'):
        """
        Plots the learning curve based on the average rewards per episode.

        Args:
            output_filename (str): The filename for the output plot.
            file_name (str): The CSV file containing reward data.
        """
        df = pd.read_csv(file_name)

        # Aggregate steps data to reduce noise
        sample_size = 100
        sampled_steps = df['Reward'].groupby(df.index // sample_size).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(sampled_steps.index * sample_size, sampled_steps, marker='o', linestyle='-', color='c', label='Learning_Curve')
        plt.title('Learning Curve')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_filename)

