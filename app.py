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
        return self.state.reshape((6, 9))

# Load the model
model = DQN.load("dqn_rubikscube")

# Create the environment
env = SimpleRubiksCubeEnv()

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
        st.write(f"Step {steps}: {env.render()}")
        if steps > 100:
            st.write("Exceeded maximum steps")
            break
    st.write("Solved!")

# Streamlit app
st.title("Rubik's Cube Solver")
if st.button("Solve Rubik's Cube"):
    solve_cube()
