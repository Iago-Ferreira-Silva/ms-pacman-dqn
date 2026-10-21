import gymnasium as gym
import ale_py
from tensorflow import keras

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5")
n_outputs = int(env.action_space.n)  # 9 ações
env.close()

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

if __name__ == "__main__":
    online_model = build_q_network()
    online_model.summary()