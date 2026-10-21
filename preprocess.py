import numpy as np
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt

gym.register_envs(ale_py)

mspacman_color = 210 + 164 + 74

def preprocess_observation(obs):
    img = obs[1:176:2, ::2]                 # recorta e reduz pela metade (amostragem)
    img = img.sum(axis=2, dtype=np.int32)   # converte para escala de cinza (soma dos 3 canais)
    img[img == mspacman_color] = 0          # remove uma cor de fundo específica (melhora contraste)
    img = (img // 3 - 128) / 128 - 1        # normaliza os valores para o intervalo [-1, 1]
    return img.reshape(88, 80, 1).astype(np.float32)

env = gym.make("ALE/MsPacman-v5")
obs, info = env.reset()

processed = preprocess_observation(obs)

print("Formato original:", obs.shape)
print("Formato pré-processado:", processed.shape)
print("Valor mínimo:", processed.min(), "| Valor máximo:", processed.max())

# Visualização: original vs. pré-processada
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(obs)
axes[0].set_title("Original (210x160x3)")
axes[1].imshow(processed.reshape(88, 80), cmap="gray")
axes[1].set_title("Pré-processada (88x80x1)")
plt.show()

env.close()