import streamlit as st
import gym
from stable_baselines3 import DQN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load the trained DQN model
model = DQN.load("rubiks_cube_model")

# Define the Rubik's Cube Gym environment
class RubiksCubeEnv(gym.Env):
    # Define the environment as shown in the previous example

    # Function to convert cube state to color representation
    def state_to_colors(state):
        # Define the color mapping as shown in the previous example

# Function to solve the Rubik's Cube using the trained model
def solve_cube():
    env = RubiksCubeEnv()
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        steps += 1
        cube_colors = state_to_colors(env.render())
        fig, ax = plt.subplots(3, 3, figsize=(6, 6))
        for i in range(9):
            face_colors = cube_colors[i*6:i*6+9]
            for j, color in enumerate(face_colors):
                row = j // 3
                col = j % 3
                ax[row, col].add_patch(Rectangle((col * 0.3, 0.3 - row * 0.3), 0.3, 0.3, color=color))
                ax[row, col].axis('off')
        st.write(fig)
        plt.close(fig)
        if steps > 100:
            st.write("Exceeded maximum steps")
            break
    st.write("Solved!")

# Streamlit app
st.title("Rubik's Cube Solver")
st.write("Welcome to the Rubik's Cube Solver! Click the button below to see the AI solve the Rubik's Cube.")
if st.button("Solve Rubik's Cube"):
    solve_cube()
