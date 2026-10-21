import numpy as np

eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 20000

def epsilon_greedy(q_values, step, n_outputs):
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs), epsilon    # explora: ação aleatória
    return int(np.argmax(q_values)), epsilon            # aproveita: melhor ação segundo a rede

if __name__ == "__main__":
    fake_q_values = np.array([1.2, 3.5, 0.1, 2.2, -1.0, 0.8, 4.1, 2.9, 0.5])
    n_outputs = 9

    for step in [0, 5000, 10000, 15000, 20000, 25000]:
        action, epsilon = epsilon_greedy(fake_q_values, step, n_outputs)
        print(f"step={step:6d}  epsilon={epsilon:.3f}  ação escolhida={action}")