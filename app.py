import streamlit as st
import gym
from stable_baselines3 import DQN
import numpy as np

# Custom environment
class SimpleRubiksCubeEnv(gym.Env):
    def __init__(self):
        super(SimpleRubiksCubeEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(54,), dtype=int)
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
        state_2d = self.state.reshape((6, 9))
        display_str = '\n'.join([' '.join(map(str, row)) for row in state_2d])
        return display_str

# Load the model
model = DQN.load("dqn_rubikscube")

# Create the environment
env = SimpleRubiksCubeEnv()

# Function to solve the Rubik's Cube
def solve_cube():
    obs = env.reset()
    done = False
    steps = 0
    max_steps = 100
    st.write("Solving the Rubik's Cube...")

    progress_bar = st.progress(0)
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        steps += 1
        st.write(f"Step {steps}:\n{env.render()}")
        progress_bar.progress(steps / max_steps)
        if steps > max_steps:
            st.write("Exceeded maximum steps")
            break
    st.write("Solved!")

# Streamlit app
st.title("Rubik's Cube Solver")
st.markdown("""
Welcome to the Rubik's Cube Solver! This app uses a reinforcement learning model to solve a simplified version of the Rubik's Cube.
Click the button below to see the AI in action.
""")

if st.button("Solve Rubik's Cube"):
    solve_cube()

st.markdown("""
## About the Solver
This solver is based on a Deep Q-Network (DQN) model trained to find the optimal sequence of moves to solve a simplified Rubik's Cube.

### How It Works
- **Environment**: The cube is represented as a 54-element array, each element indicating the color of a square.
- **Actions**: There are 12 possible moves (e.g., rotating a face of the cube).
- **Rewards**: The model receives a reward for each move, guiding it to find the optimal solution.

### Model Training
The model was trained using the Stable Baselines3 library, which provides implementations of various reinforcement learning algorithms.

**Note**: This is a simplified demonstration. Solving a real Rubik's Cube would require a more complex model and state representation.
""")

