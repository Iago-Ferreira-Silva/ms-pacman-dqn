import numpy as np
from collections import deque
import gymnasium as gym
import ale_py
import tensorflow as tf
from tensorflow import keras

gym.register_envs(ale_py)

# ---------- 1. Ambiente ----------
env = gym.make("ALE/MsPacman-v5")
n_outputs = int(env.action_space.n)

# ---------- 2. Pré-processamento ----------
mspacman_color = 210 + 164 + 74

def preprocess_observation(obs):
    img = obs[1:176:2, ::2]
    img = img.sum(axis=2, dtype=np.int32)
    img[img == mspacman_color] = 0
    img = (img // 3 - 128) / 128 - 1
    return img.reshape(88, 80, 1).astype(np.float32)

# ---------- 3. Arquitetura da DQN ----------
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

online_model = build_q_network()
target_model = build_q_network()
target_model.set_weights(online_model.get_weights())

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.Huber()

# ---------- 4. Replay memory ----------
replay_memory = deque(maxlen=2000)

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    states, actions, rewards, next_states, continues = [], [], [], [], []
    for idx in indices:
        s, a, r, sp, c = replay_memory[idx]
        states.append(s); actions.append(a); rewards.append(r)
        next_states.append(sp); continues.append(c)
    return (np.array(states), np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states), np.array(continues, dtype=np.float32))

# ---------- 5. Epsilon-greedy ----------
eps_min, eps_max, eps_decay_steps = 0.1, 1.0, 20000

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    return int(np.argmax(q_values))

# ---------- 6. Loop de treino (demonstração curta) ----------
discount_rate = 0.95
batch_size = 32
training_start = 200
training_interval = 4
copy_steps = 200
n_steps_demo = 2000       # demonstração curta (livro usa milhões de passos)
save_path = "dqn_pacman_demo.weights.h5"

obs, info = env.reset()
state = preprocess_observation(obs)
done = False
episode_reward = 0
step = 0

print("Iniciando treino de demonstração (DQN - Ms. Pac-Man)...")
while step < n_steps_demo:
    if done:
        print(f"  Episódio terminou no step {step} | recompensa do episódio: {episode_reward}")
        obs, info = env.reset()
        state = preprocess_observation(obs)
        done = False
        episode_reward = 0

    q_values = online_model.predict(state[np.newaxis], verbose=0)[0]
    action = epsilon_greedy(q_values, step)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    next_state = preprocess_observation(obs)

    replay_memory.append((state, action, reward, next_state, 0.0 if done else 1.0))
    state = next_state
    episode_reward += reward
    step += 1

    if step > training_start and step % training_interval == 0 and len(replay_memory) >= batch_size:
        states, actions, rewards, next_states, continues = sample_memories(batch_size)
        next_q_values = target_model.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1)
        y_targets = rewards + continues * discount_rate * max_next_q_values

        with tf.GradientTape() as tape:
            all_q_values = online_model(states, training=True)
            action_masks = tf.one_hot(actions, n_outputs)
            q_values_pred = tf.reduce_sum(all_q_values * action_masks, axis=1)
            loss = loss_fn(y_targets, q_values_pred)
        grads = tape.gradient(loss, online_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, online_model.trainable_variables))

        if step % 100 == 0:
            print(f"  step={step}  loss={loss.numpy():.4f}  epsilon ativo")

    if step % copy_steps == 0:
        target_model.set_weights(online_model.get_weights())

online_model.save_weights(save_path)
print(f"Treino de demonstração concluído. Passos executados: {step}")
print(f"Pesos salvos em: {save_path}")

env.close()