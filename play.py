"""
play.py — Visualiza o agente de DQN jogando Ms. Pac-Man em tempo real.

Modos disponíveis:
  --mode random   → agente aleatório (sem treino), bom para ver a linha de base
  --mode trained  → carrega os pesos salvos e usa a rede neural treinada

Exemplos de uso:
  py -3.13 play.py --mode random
  py -3.13 play.py --mode trained
  py -3.13 play.py --mode trained --weights dqn_pacman_demo.weights.h5
  py -3.13 play.py --mode trained --episodes 3 --speed 0.02
"""

import argparse
import time
import numpy as np
import gymnasium as gym
import ale_py
import tensorflow as tf
from tensorflow import keras

# ── Configuração ──────────────────────────────────────────────────────────────
gym.register_envs(ale_py)

parser = argparse.ArgumentParser(description="Visualiza o agente DQN jogando Ms. Pac-Man")
parser.add_argument("--mode",     choices=["random", "trained"], default="random",
                    help="'random' = ações aleatórias | 'trained' = usa a rede treinada")
parser.add_argument("--weights",  default="dqn_pacman_demo.weights.h5",
                    help="Caminho para o arquivo de pesos (.weights.h5)")
parser.add_argument("--episodes", type=int, default=2,
                    help="Número de episódios para assistir")
parser.add_argument("--speed",    type=float, default=0.03,
                    help="Pausa em segundos entre frames (0 = máxima velocidade)")
args = parser.parse_args()

# ── Pré-processamento (igual ao train.py) ────────────────────────────────────
mspacman_color = 210 + 164 + 74

def preprocess_observation(obs):
    img = obs[1:176:2, ::2]
    img = img.sum(axis=2, dtype=np.int32)
    img[img == mspacman_color] = 0
    img = (img // 3 - 128) / 128 - 1
    return img.reshape(88, 80, 1).astype(np.float32)

# ── Ambiente com renderização ─────────────────────────────────────────────────
env = gym.make("ALE/MsPacman-v5", render_mode="human")
n_outputs = int(env.action_space.n)

# ── Carrega a rede (apenas no modo 'trained') ─────────────────────────────────
model = None
if args.mode == "trained":
    model = keras.Sequential([
        keras.layers.Input(shape=(88, 80, 1)),
        keras.layers.Conv2D(32, kernel_size=8, strides=4, padding="same", activation="relu"),
        keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same", activation="relu"),
        keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(n_outputs),
    ])
    try:
        model.build(input_shape=(None, 88, 80, 1))
        model.load_weights(args.weights)
        print(f"✅ Pesos carregados de: {args.weights}")
    except Exception as e:
        print(f"❌ Erro ao carregar pesos: {e}")
        print("   Execute train.py primeiro para gerar os pesos, ou use --mode random")
        env.close()
        exit(1)

# ── Loop de episódios ─────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Modo: {'🎲 Aleatório' if args.mode == 'random' else '🧠 Agente Treinado'}")
print(f"  Episódios: {args.episodes}")
print(f"  Feche a janela do jogo para encerrar")
print(f"{'='*50}\n")

for ep in range(1, args.episodes + 1):
    obs, info = env.reset()
    state = preprocess_observation(obs)
    done = False
    total_reward = 0.0
    steps = 0

    print(f"[Episódio {ep}/{args.episodes}] iniciando...")

    while not done:
        env.render()  # atualiza a janela

        if args.mode == "random":
            action = env.action_space.sample()
        else:
            q_values = model.predict(state[np.newaxis], verbose=0)[0]
            action = int(np.argmax(q_values))  # sempre a melhor ação (sem exploração)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = preprocess_observation(obs)
        total_reward += reward
        steps += 1

        if args.speed > 0:
            time.sleep(args.speed)

    print(f"  → Episódio {ep} encerrado | steps: {steps} | recompensa total: {total_reward:.0f}")

print("\n✅ Demonstração concluída.")
env.close()
