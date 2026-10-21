import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5")

obs, info = env.reset()

print("Formato da observação (imagem bruta do jogo):", obs.shape)
print("Tipo de dado da observação:", obs.dtype)
print("Espaço de ações:", env.action_space)
print("Número de ações possíveis:", env.action_space.n)
print("Significado das ações:", env.unwrapped.get_action_meanings())

env.close()