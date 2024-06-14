# import streamlit as st
# import gym
# from stable_baselines3 import DQN
# import numpy as np

# # Custom environment
# class SimpleRubiksCubeEnv(gym.Env):
#     def __init__(self):
#         super(SimpleRubiksCubeEnv, self).__init__()
#         self.action_space = gym.spaces.Discrete(12)
#         self.observation_space = gym.spaces.Box(low=0, high=5, shape=(54,), dtype=int)
#         self.state = np.zeros(54, dtype=int)

#     def reset(self):
#         self.state = np.zeros(54, dtype=int)
#         return self.state

#     def step(self, action):
#         reward = -1
#         done = np.random.rand() > 0.95
#         self.state = np.random.randint(0, 6, 54)
#         return self.state, reward, done, {}

#     def render(self, mode='human'):
#         return self.state

# # Function to convert cube state to color representation
# def state_to_colors(state):
#     colors = {
#         0: "🟥",  # Red
#         1: "🟧",  # Orange
#         2: "🟨",  # Yellow
#         3: "🟩",  # Green
#         4: "🟦",  # Blue
#         5: "⬜"   # White
#     }
#     return ''.join([colors[x] for x in state])

# # Load the model
# model = DQN.load("dqn_rubikscube")

# # Create the environment
# env = SimpleRubiksCubeEnv()

# # Function to solve the Rubik's Cube
# def solve_cube():
#     obs = env.reset()
#     done = False
#     steps = 0
#     st.write("Solving the Rubik's Cube...")
#     while not done:
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         steps += 1
#         st.write(f"Step {steps}: {state_to_colors(env.render())}")
#         if steps > 100:
#             st.write("Exceeded maximum steps")
#             break
#     st.write("Solved!")

# # Streamlit app
# st.title("Rubik's Cube Solver")
# st.write("Welcome to the Rubik's Cube Solver! This app uses a reinforcement learning model to solve a simplified version of the Rubik's Cube. Click the button below to see the AI in action.")
# if st.button("Solve Rubik's Cube"):
#     solve_cube()

# # Additional information about the solver
# st.subheader("About the Solver")
# st.write("This solver is based on a Deep Q-Network (DQN) model trained to find the optimal sequence of moves to solve a simplified Rubik's Cube.")
# st.subheader("How It Works")
# st.write("""
# - **Environment**: The cube is represented as a 54-element array, each element indicating the color of a square.
# - **Actions**: There are 12 possible moves (e.g., rotating a face of the cube).
# - **Rewards**: The model receives a reward for each move, guiding it to find the optimal solution.
# """)
# st.subheader("Model Training")
# st.write("The model was trained using the Stable Baselines3 library, which provides implementations of various reinforcement learning algorithms.")
# st.write("Note: This is a simplified demonstration. Solving a real Rubik's Cube would require a more complex model and state representation.")


import streamlit as st
import gym
import numpy as np

# Custom environment
class SimpleRubiksCubeEnv(gym.Env):
    def __init__(self):
        super(SimpleRubiksCubeEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(54,), dtype=int)
        self.state = np.zeros(54, dtype=int)

    def reset(self):
        self.state = np.zeros(54, dtype=int)
        return self.state

    def step(self, action):
        reward = -1
        done = np.random.rand() > 0.95
        self.state = np.random.randint(0, 6, 54)
        return self.state, reward, done, {}

    def render(self, mode='human'):
        return self.state.reshape((6, 9))

# Load the model
model = None  # Replace with your model loading code

# Create the environment
env = SimpleRubiksCubeEnv()

# Function to convert state to colors
def state_to_colors(state):
    colors = ['W', 'G', 'R', 'B', 'O', 'Y']
    color_state = []
    for i in range(54):
        color_state.append(colors[state[i]])
    return color_state

# Function to solve the Rubik's Cube
def solve_cube():
    obs = env.reset()
    done = False
    steps = 0
    st.write("Solving the Rubik's Cube...")
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        steps += 1
        cube_colors = state_to_colors(env.render())
        st.write(f"Step {steps}: {cube_colors}")
        if steps > 100:
            st.write("Exceeded maximum steps")
            break
    st.write("Solved!")

# Streamlit app
st.title("Rubik's Cube Solver")
if st.button("Solve Rubik's Cube"):
    solve_cube()


# Streamlit app
st.title("Rubik's Cube Solver")
if st.button("Solve Rubik's Cube"):
    solve_cube()
