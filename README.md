# Ms. Pac-Man DQN — Deep Q-Network em TensorFlow/Keras

Implementação de um agente de **Aprendizado por Reforço Profundo (Deep Reinforcement Learning)** para jogar **Ms. Pac-Man**, usando o algoritmo **Deep Q-Network (DQN)**.

Este projeto foi desenvolvido como atividade prática baseada no **Capítulo 16 — "Aprendizado por Reforço"** do livro *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn & TensorFlow*, de Aurélien Géron, para a disciplina de **Tópicos Especiais em Sistemas de Informação** (Prof. Robson Gonçalves Fechine Feitosa), no curso de Sistemas de Informação do **IFCE — Campus Crato**.

> **Autor:** Iago Ferreira Silva

---

## Índice

- [Sobre o projeto](#sobre-o-projeto)
- [O que é um DQN, em resumo](#o-que-é-um-dqn-em-resumo)
- [Modernização em relação ao livro](#modernização-em-relação-ao-livro)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Explicação detalhada de cada arquivo](#explicação-detalhada-de-cada-arquivo)
  - [`explore_env.py`](#explore_envpy)
  - [`preprocess.py`](#preprocesspy)
  - [`dqn_model.py`](#dqn_modelpy)
  - [`replay_memory.py`](#replay_memorypy)
  - [`epsilon_greedy.py`](#epsilon_greedypy)
  - [`train.py`](#trainpy)
  - [`evaluate.py`](#evaluatepy)
  - [`play.py`](#playpy)
  - [`requirements.txt`](#requirementstxt)
  - [`.gitignore`](#gitignore)
- [Como rodar o projeto do zero](#como-rodar-o-projeto-do-zero)
- [Resultados obtidos](#resultados-obtidos)
- [Limitações deste projeto](#limitações-deste-projeto)
- [Referências](#referências)

---

## Sobre o projeto

O objetivo foi reproduzir, de forma fiel à arquitetura e à lógica descritas no livro, um agente de Aprendizado por Reforço capaz de jogar Ms. Pac-Man a partir de imagens brutas da tela do jogo — sem nenhuma informação além dos pixels e da recompensa (pontuação).

O projeto foi construído **passo a passo**, em etapas incrementais, cada uma validando uma parte específica do pipeline antes de seguir para a próxima:

1. Exploração do ambiente de jogo
2. Pré-processamento das imagens
3. Construção da rede neural (arquitetura DQN)
4. Memória de repetição (replay memory)
5. Política de exploração (epsilon-greedy)
6. Loop de treino completo
7. Avaliação do agente treinado

## O que é um DQN, em resumo

Um **Deep Q-Network** é uma técnica de Aprendizado por Reforço que usa uma rede neural para estimar os **Q-values** — o retorno total esperado (recompensa futura descontada) de se tomar cada ação possível em um determinado estado (no nosso caso, uma imagem do jogo).

Em vez de guardar uma tabela de valores para cada combinação possível de estado/ação (inviável quando o estado é uma imagem), a rede neural **aprende a aproximar** esses valores. O agente então escolhe, a cada passo, a ação com maior Q-value estimado — ou, durante o treino, às vezes uma ação aleatória, para continuar explorando o ambiente (ver [`epsilon_greedy.py`](#epsilon_greedypy)).

Dois mecanismos do algoritmo, usados neste projeto, merecem destaque por resolverem problemas práticos de estabilidade do treino:

- **Replay memory:** evita treinar com experiências muito correlacionadas (frames consecutivos do jogo), guardando experiências passadas e sorteando lotes aleatórios delas para treinar.
- **Rede online + rede alvo (target network):** duas cópias da mesma rede. A rede *online* é a que treina a cada passo; a rede *alvo* fornece os valores de referência (alvos) do treino e só é atualizada periodicamente, evitando que o alvo do treino mude a cada passo e desestabilize o aprendizado.

## Modernização em relação ao livro

O livro foi publicado em 2019 e usa bibliotecas hoje obsoletas: **TensorFlow 1.x** (com `placeholder`, sessões e `tf.contrib`) e a biblioteca `gym` original da OpenAI (hoje descontinuada). Para rodar em um ambiente atual, o projeto foi reescrito com bibliotecas modernas, **mantendo a mesma arquitetura e a mesma lógica do algoritmo** descritas no livro:

| Livro (2019) | Este projeto (atual) |
|---|---|
| TensorFlow 1.x (`placeholder`, `Session`) | TensorFlow 2 / Keras (`keras.Sequential`, `tf.GradientTape`) |
| `gym` (OpenAI, descontinuado) | `gymnasium` (sucessor oficial, mantido pela Farama Foundation) |
| Ambiente Atari via `gym[atari]` | `ale-py` (Arcade Learning Environment), ambiente `ALE/MsPacman-v5` |

## Estrutura do projeto

```
ms_pacman_dqn/
├── venv/                        # ambiente virtual Python (não versionado)
├── explore_env.py                # Etapa 2 — exploração do ambiente
├── preprocess.py                 # Etapa 3 — pré-processamento das imagens
├── dqn_model.py                  # Etapa 4 — arquitetura da rede neural (CNN)
├── replay_memory.py               # Etapa 5 — memória de repetição
├── epsilon_greedy.py              # Etapa 6 — política de exploração
├── train.py                       # Etapa 7 — loop de treino completo
├── evaluate.py                    # Etapa 8 — avaliação do agente treinado
├── play.py                        # extra — assiste ao agente jogando, aleatório ou treinado, via linha de comando
├── dqn_pacman_demo.weights.h5      # pesos treinados (não versionado)
├── requirements.txt                # dependências do projeto
├── .gitignore
└── README.md
```

## Explicação detalhada de cada arquivo

### `explore_env.py`

**Objetivo:** entender o ambiente de jogo antes de desenvolver qualquer lógica de IA.

Cria o ambiente `ALE/MsPacman-v5` (via `gymnasium` + `ale_py`) e inspeciona:
- **Formato da observação:** a tela do jogo é devolvida como uma imagem RGB de `(210, 160, 3)` — 210 pixels de altura, 160 de largura, 3 canais de cor.
- **Espaço de ações:** `Discrete(9)` — 9 ações possíveis (`NOOP`, `UP`, `RIGHT`, `LEFT`, `DOWN` e as 4 diagonais).

Esse script não faz parte do pipeline de treino em si; serve apenas como etapa de reconhecimento, confirmando os formatos de dados que as próximas etapas precisam tratar.

### `preprocess.py`

**Objetivo:** transformar a imagem bruta do jogo em algo mais leve e adequado para alimentar a rede neural.

A função principal, `preprocess_observation(obs)`, faz quatro transformações, nessa ordem:

1. **Recorte e subamostragem** (`obs[1:176:2, ::2]`): corta partes irrelevantes da imagem (como o placar) e reduz a resolução pela metade, pegando 1 a cada 2 pixels em cada eixo.
2. **Conversão para escala de cinza** (`img.sum(axis=2, dtype=np.int32)`): soma os 3 canais de cor (R+G+B) em um único valor por pixel, eliminando a dimensão de cor. É importante forçar o tipo `int32` aqui — sem isso, o NumPy mantém o resultado como inteiro sem sinal (`uint64`), o que causa um erro sutil de *overflow* (estouro) na etapa seguinte.
3. **Melhora de contraste:** zera uma cor de fundo específica do jogo (`mspacman_color`), destacando os elementos relevantes (Pac-Man, fantasmas, paredes, pastilhas).
4. **Normalização:** reescala os valores de pixel para o intervalo `[-1, 1]`, o que ajuda a rede neural a treinar de forma mais estável.

O resultado é uma imagem de `(88, 80, 1)` — bem menor que o original `(210, 160, 3)` e sem informação de cor — pronta para ser usada como entrada da rede neural.

Esse arquivo também inclui um bloco de teste (`if __name__ == "__main__"`) que gera um gráfico comparando a imagem original com a pré-processada, usando `matplotlib`.

### `dqn_model.py`

**Objetivo:** definir a arquitetura da rede neural convolucional (CNN) que estima os Q-values.

A função `build_q_network()` monta a rede com `keras.Sequential`:

| Camada | Função |
|---|---|
| `Conv2D(32, kernel_size=8, strides=4)` | Primeira camada convolucional — extrai padrões visuais básicos da imagem pré-processada |
| `Conv2D(64, kernel_size=4, strides=2)` | Segunda convolução — combina os padrões anteriores em características mais complexas |
| `Conv2D(64, kernel_size=3, strides=1)` | Terceira convolução — refina ainda mais as características extraídas |
| `Flatten()` | Achata o resultado 2D das convoluções em um vetor 1D |
| `Dense(512, activation="relu")` | Camada totalmente conectada, combina todas as características extraídas |
| `Dense(n_outputs)` | Camada de saída — um neurônio por ação possível (9 no total), cada um estimando o Q-value daquela ação |

Ao todo, essa rede tem **3.681.449 parâmetros treináveis**. A entrada é a imagem pré-processada `(88, 80, 1)`, e a saída são 9 valores numéricos (os Q-values de cada ação — sem função de ativação na última camada, já que são valores livres, não probabilidades).

Essa mesma função é reutilizada mais tarde (em `train.py` e `evaluate.py`) para criar **duas cópias independentes** da rede: a rede *online* (que treina) e a rede *alvo* (que fornece valores de referência estáveis).

### `replay_memory.py`

**Objetivo:** implementar a memória de repetição (replay memory), que resolve o problema de dados sequenciais muito correlacionados.

- `replay_memory = deque(maxlen=2000)`: uma fila de tamanho fixo. Quando atinge o limite, as experiências mais antigas são descartadas automaticamente para dar lugar às novas.

> **Nota:** este arquivo é um exemplo isolado, só para validar a lógica da replay memory (daí o `maxlen` pequeno, de `2000`). O `train.py` tem sua **própria cópia** dessa mesma lógica embutida nele (não importa este arquivo), e nela o `maxlen` foi ampliado para `50000` — ver a seção de [`train.py`](#trainpy).
- `sample_memories(batch_size)`: sorteia aleatoriamente um lote (`batch`) de experiências armazenadas, cada uma no formato `(estado, ação, recompensa, próximo_estado, continua?)`, e organiza tudo em arrays NumPy prontos para o treino.

Esse arquivo inclui um teste standalone com dados falsos (`fake_state`, `fake_action`, etc.) só para validar que a lógica de armazenamento e sorteio funciona corretamente, sem precisar rodar o jogo de verdade.

### `epsilon_greedy.py`

**Objetivo:** implementar a política de ação do agente, equilibrando exploração e aproveitamento.

A função `epsilon_greedy(q_values, step, n_outputs)` decide a ação do agente:
- Com probabilidade `epsilon`, escolhe uma ação **aleatória** (explora o ambiente, descobrindo novas estratégias).
- Caso contrário, escolhe a ação de **maior Q-value** segundo a rede (aproveita o que já foi aprendido).

O valor de `epsilon` começa em `1.0` (100% de exploração) e decai linearmente até `0.1` (10%) ao longo de `eps_decay_steps = 20000` passos, permanecendo nesse mínimo depois disso. Essa é a estratégia clássica de **exploration vs. exploitation**: no início do treino, o agente explora bastante (ainda não sabe nada); com o tempo, passa a confiar mais na rede, mas nunca para de explorar completamente.

### `train.py`

**Objetivo:** o arquivo mais importante do projeto — junta todos os componentes anteriores em um loop de treino completo.

Principais blocos do script, na ordem em que aparecem:

1. **Ambiente** — cria o `ALE/MsPacman-v5`.
2. **Pré-processamento** — reaproveita a função de `preprocess.py`.
3. **Redes neurais** — cria `online_model` e `target_model` (duas instâncias de `build_q_network()`), e sincroniza os pesos da rede alvo com os da rede online no início (`target_model.set_weights(online_model.get_weights())`).
4. **Replay memory** — mesma lógica de `replay_memory.py` (implementada como uma cópia própria dentro deste arquivo, não uma importação), mas com `maxlen=50000` em vez de `2000`, para suportar o treino mais longo.
5. **Epsilon-greedy** — reaproveita a lógica de `epsilon_greedy.py`.
6. **Loop de treino**, que a cada passo:
   - Observa o estado atual (imagem pré-processada) e escolhe uma ação via epsilon-greedy.
   - Executa a ação no ambiente, observa a recompensa e o próximo estado.
   - Guarda essa experiência na replay memory.
   - A cada `training_interval = 4` passos (depois de `training_start = 200` passos iniciais, para já ter experiências suficientes na memória), sorteia um lote da replay memory e treina a rede online:
     - Calcula os **Q-values alvo** usando a rede *alvo*: `y_targets = rewards + continues * discount_rate * max_next_q_values` (a fórmula de Bellman, com `discount_rate = 0.95`).
     - Usa `tf.GradientTape()` para calcular o gradiente da função de perda (`Huber loss`, mais robusta a valores extremos que o erro quadrático médio) entre os Q-values previstos pela rede online e os alvos calculados.
     - Aplica o gradiente com o otimizador Adam, atualizando os pesos da rede online.
   - A cada `copy_steps = 200` passos, copia os pesos da rede online para a rede alvo — é isso que estabiliza o treino, evitando que o "alvo" do aprendizado fique mudando a cada passo.
7. Ao final, salva os pesos treinados em `dqn_pacman_demo.weights.h5`.

> **Nota sobre o escopo do treino:** o livro descreve um treino de **milhões de passos**, levando dias para rodar. Este projeto roda `n_steps_demo = 500000` passos (meio milhão) — bem menor que o do livro, mas já suficiente para o agente sair do comportamento quase aleatório inicial e desenvolver uma estratégia de jogo consistente (ver [Resultados obtidos](#resultados-obtidos)). Esse valor começou em `2000` passos (uma validação rápida do pipeline completo) e foi ampliado para `500000`, junto com o aumento da replay memory, para produzir um agente mais capaz. Com o hardware usado, esse treino levou aproximadamente 30 horas.

### `evaluate.py`

**Objetivo:** carregar os pesos treinados e observar o agente jogando, usando apenas o que foi aprendido (sem exploração aleatória).

- Recria a mesma arquitetura de `dqn_model.py` e carrega os pesos salvos por `train.py` (`model.load_weights("dqn_pacman_demo.weights.h5")`).
- Roda `n_episodes = 3` episódios completos, em cada um escolhendo sempre a ação de maior Q-value (`np.argmax(q_values)`, sem `epsilon`).
- Usa `render_mode="human"`, que abre uma janela mostrando o jogo rodando em tempo real.
- Imprime a recompensa total de cada episódio ao final.

Com o treino de 500.000 passos, o agente carregado aqui já joga de forma bastante mais consistente do que no treino inicial de validação (ver [Resultados obtidos](#resultados-obtidos)).

### `play.py`

**Objetivo:** uma versão mais flexível do `evaluate.py` para assistir ao agente jogando, criada pelo colega de equipe. Permite comparar lado a lado o comportamento **antes e depois do treino**, direto pela linha de comando.

Principais diferenças em relação ao `evaluate.py`:

- **Dois modos**, escolhidos pelo argumento `--mode`:
  - `random` (padrão): o agente escolhe ações aleatórias (`env.action_space.sample()`) — serve como "linha de base", mostrando como o jogo se comporta sem nenhum aprendizado.
  - `trained`: carrega os pesos salvos (`model.load_weights(...)`) e usa a rede treinada, escolhendo sempre a ação de maior Q-value — igual ao `evaluate.py`.
- **Argumentos de linha de comando** (via `argparse`), que tornam o script reutilizável sem precisar editar o código:
  - `--weights`: caminho do arquivo de pesos (padrão: `dqn_pacman_demo.weights.h5`).
  - `--episodes`: quantos episódios assistir (padrão: 2).
  - `--speed`: pausa (em segundos) entre cada frame, útil para deixar a visualização mais lenta e fácil de acompanhar (padrão: 0.03s; `0` roda na velocidade máxima).
- **Tratamento de erro ao carregar os pesos:** se o arquivo de pesos não existir (por exemplo, antes de rodar o `train.py`), o script avisa com uma mensagem clara em vez de travar com um erro genérico.
- Chama `env.render()` explicitamente a cada passo do loop, garantindo que a janela do jogo seja atualizada quadro a quadro.

Exemplos de uso:
```powershell
python play.py --mode random                                    # ve o agente sem treino nenhum
python play.py --mode trained                                   # ve o agente treinado
python play.py --mode trained --episodes 3 --speed 0.02          # 3 episodios, mais rapido
```

Esse script é especialmente útil para apresentações: rodar primeiro em `--mode random` e depois em `--mode trained` evidencia visualmente o quanto o agente aprendeu.

### `requirements.txt`

Lista de todas as dependências Python do projeto (geradas com `pip freeze`), incluindo as bibliotecas principais:

- `tensorflow` — framework de redes neurais (Keras incluso)
- `gymnasium` — biblioteca de ambientes de Aprendizado por Reforço (sucessora do `gym`)
- `ale-py` — Arcade Learning Environment, fornece os jogos Atari (incluindo Ms. Pac-Man)
- `numpy` — operações numéricas e manipulação de arrays
- `matplotlib` — visualização (usado em `preprocess.py` para comparar imagens)

Permite que qualquer pessoa recrie o ambiente exato do projeto com:
```powershell
pip install -r requirements.txt
```

### `.gitignore`

Define o que **não** deve ser versionado no Git:

```
venv/
__pycache__/
*.pyc
checkpoints/
*.h5
*.weights.h5
```

- `venv/` — o ambiente virtual Python não deve ir para o repositório (é específico de cada máquina; quem for rodar o projeto recria o seu próprio com `requirements.txt`).
- `__pycache__/`, `*.pyc` — arquivos de cache gerados automaticamente pelo Python.
- `*.h5`, `*.weights.h5`, `checkpoints/` — os pesos treinados são arquivos binários grandes, não fazem sentido versionados em um repositório de código.

## Como rodar o projeto do zero

```powershell
# 1. Criar e ativar o ambiente virtual
python -m venv venv
venv\Scripts\Activate.ps1

# 2. Instalar as dependências
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. (Opcional) Explorar o ambiente
python explore_env.py

# 4. (Opcional) Visualizar o pré-processamento
python preprocess.py

# 5. (Opcional) Ver o resumo da arquitetura da rede
python dqn_model.py

# 6. Treinar o agente (500.000 passos — pode levar muitas horas)
python train.py

# 7. Avaliar o agente treinado (abre uma janela com o jogo)
python evaluate.py
```

## Resultados obtidos

- O treino completo (500.000 passos) rodou por aproximadamente 30 horas. Ao longo do treino, as recompensas por episódio foram, de forma geral, crescendo e ficando mais consistentes: nos primeiros passos os episódios terminavam rápido com pontuações baixas, enquanto próximo do final já apareciam episódios de **800, 900, 990 e até 1530 pontos**.
- Na avaliação final (`evaluate.py`, agente já treinado, sem exploração aleatória), o agente obteve **470, 490 e 870 pontos** nos 3 episódios — valores bem mais altos e consistentes do que o treino inicial de validação (que obtinha 70 pontos, repetidos de forma idêntica nos 3 episódios). Os resultados variam um pouco entre episódios porque o ambiente `ALE/MsPacman-v5` tem uma pequena aleatoriedade interna por padrão (*sticky actions*: às vezes repete a ação anterior em vez da escolhida pela política), então mesmo uma política puramente gulosa (greedy) não repete exatamente a mesma partida a cada vez.

### Histórico: da validação rápida ao treino completo

O projeto passou por duas fases de treino:

1. **Validação do pipeline (2000 passos, replay memory de 2000):** usada para confirmar rapidamente que todo o pipeline (ambiente → pré-processamento → rede → memória → treino) funcionava sem erros, antes de investir tempo em um treino longo. Resultado: 3 episódios com recompensas de 230, 210 e 240 pontos durante o treino, e 70 pontos (repetidos) na avaliação — o agente mal havia começado a aprender.
2. **Treino completo (500.000 passos, replay memory de 50000):** parâmetros ampliados para dar ao agente tempo e memória suficientes para de fato aprender uma estratégia de jogo, com os resultados descritos acima.

## Limitações deste projeto

- **Treino menor que o do livro:** o livro treina o agente por milhões de passos, ao longo de dias; aqui foram usados 500.000 passos (~30 horas), por restrição de tempo/recursos. O agente aprendeu uma estratégia real, mas ainda aquém do que um treino de milhões de passos produziria.
- **Sem checkpoints intermediários:** o livro salva o progresso periodicamente para poder retomar um treino longo; aqui os pesos só são salvos ao final do loop, então uma interrupção no meio do treino perde todo o progresso.

Essas limitações foram uma escolha deliberada: o foco do projeto foi reproduzir **corretamente a arquitetura e a lógica do algoritmo DQN** descrito no livro, validando cada etapa do pipeline antes de escalar para um treino mais longo.

## Referências

- GÉRON, Aurélien. *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn & TensorFlow*. Capítulo 16 — Aprendizado por Reforço. O'Reilly / Alta Books, 2019.
- [Gymnasium (Farama Foundation)](https://gymnasium.farama.org/)
- [Arcade Learning Environment (ALE)](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
- [TensorFlow / Keras](https://www.tensorflow.org/)