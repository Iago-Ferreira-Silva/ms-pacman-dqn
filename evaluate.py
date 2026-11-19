import numpy as np
import gymnasium as gym
import ale_py
from tensorflow import keras

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", render_mode="human")
n_outputs = int(env.action_space.n)

mspacman_color = 210 + 164 + 74

def preprocess_observation(obs):
    img = obs[1:176:2, ::2]
    img = img.sum(axis=2, dtype=np.int32)
    img[img == mspacman_color] = 0
    img = (img // 3 - 128) / 128 - 1
    return img.reshape(88, 80, 1).astype(np.float32)

def build_q_network():
    model = keras.Sequential([
        keras.layers.Input(shape=(88, 80, 1)),
        keras.layers.Conv2D(32, kernel_size=8, strides=4, padding="same", activation="relu"),
        keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same", activation="relu"),
        keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(n_outputs),
    ])
    return model

model = build_q_network()
model.load_weights("dqn_pacman_demo.weights.h5")

n_episodes = 3

for episode in range(1, n_episodes + 1):
    obs, info = env.reset()
    state = preprocess_observation(obs)
    done = False
    total_reward = 0

    while not done:
        q_values = model.predict(state[np.newaxis], verbose=0)[0]
        action = int(np.argmax(q_values))  # sempre a melhor ação (sem exploração)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = preprocess_observation(obs)
        total_reward += reward

    print(f"Episódio {episode}: recompensa total = {total_reward}")

env.close()