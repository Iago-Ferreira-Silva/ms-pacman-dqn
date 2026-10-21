import numpy as np
from collections import deque

replay_memory = deque(maxlen=2000)  # memória pequena, só para demonstração

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    states, actions, rewards, next_states, continues = [], [], [], [], []
    for idx in indices:
        s, a, r, sp, c = replay_memory[idx]
        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(sp)
        continues.append(c)
    return (
        np.array(states),
        np.array(actions),
        np.array(rewards, dtype=np.float32),
        np.array(next_states),
        np.array(continues, dtype=np.float32),
    )

if __name__ == "__main__":
    # Teste com dados falsos (fake), só para verificar se a lógica funciona
    for i in range(10):
        fake_state = np.zeros((88, 80, 1), dtype=np.float32)
        fake_action = np.random.randint(9)
        fake_reward = float(np.random.randint(0, 10))
        fake_next_state = np.zeros((88, 80, 1), dtype=np.float32)
        fake_continue = 1.0
        replay_memory.append((fake_state, fake_action, fake_reward, fake_next_state, fake_continue))

    print("Itens na memória:", len(replay_memory))

    states, actions, rewards, next_states, continues = sample_memories(batch_size=4)
    print("Formato de states:", states.shape)
    print("Formato de actions:", actions.shape)
    print("Formato de rewards:", rewards.shape)
    print("Recompensas sorteadas:", rewards)