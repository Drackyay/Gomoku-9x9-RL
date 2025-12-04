# 🎮 Gomoku 9×9 AI with AlphaZero

A reinforcement learning project that trains an AI to play Gomoku (Five in a Row) using the **AlphaZero algorithm** (MCTS + Neural Network).

## 📋 Project Structure

```
gomoku-9x9-ppo/
├── env/
│   ├── __init__.py
│   ├── gomoku_env.py      # Gymnasium environment
│   └── test_env.py        # Environment tests
├── rl/
│   ├── __init__.py
│   ├── alphazero_train.py # AlphaZero training (MCTS + NN)
│   └── mcts_ai.py         # Monte Carlo Tree Search
├── gui/
│   ├── __init__.py
│   └── play_gui.py        # Streamlit GUI
├── models/                 # Saved models
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Play Against AI (MCTS)

```bash
streamlit run gui/play_gui.py
```

Open http://localhost:8501 in your browser.

### 3. Train AlphaZero Model

```bash
python rl/alphazero_train.py --iterations 30 --games 50 --simulations 100
```

## 🧠 How It Works

### AlphaZero Algorithm

This project implements the **AlphaZero** approach:

```
┌─────────────────────────────────────────┐
│           ALPHAZERO TRAINING            │
├─────────────────────────────────────────┤
│                                         │
│  1. SELF-PLAY                          │
│     MCTS (guided by NN) plays games    │
│     against itself                      │
│                                         │
│  2. TRAINING                           │
│     Neural network learns to predict:   │
│     - Move probabilities (policy)       │
│     - Position value (who's winning)    │
│                                         │
│  3. REPEAT                             │
│     Better NN → Better MCTS → Better   │
│     training data → Even better NN     │
│                                         │
└─────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **Neural Network** | ResNet that evaluates positions and suggests moves |
| **MCTS** | Monte Carlo Tree Search for move selection |
| **Self-Play** | Generate training data by playing against itself |

### Difficulty Levels

| Level | MCTS Simulations | Thinking Time |
|-------|------------------|---------------|
| Easy | 50 | ~0.5s |
| Medium | 200 | ~2s |
| Hard | 500 | ~4s |

## 🎯 Game Rules

- **Board**: 9×9 grid
- **Players**: Black (⚫) vs White (⚪)
- **Objective**: Get 5 stones in a row (horizontal, vertical, or diagonal)
- **Black moves first**

## 📊 Training Details

### AlphaZero Configuration

```python
GomokuNet:
  - 5 Residual Blocks
  - 128 channels
  - Policy head: predicts move probabilities
  - Value head: predicts game outcome

MCTS:
  - 100 simulations per move
  - UCB exploration constant: 1.5
  - Temperature: 1.0 (early game), 0.1 (late game)
```

### Training Loop

1. **Self-play**: 50 games per iteration
2. **Training**: Update neural network
3. **Evaluation**: Test against heuristic opponent
4. **Repeat**: 30 iterations total

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+ (CUDA recommended)
- Streamlit
- NumPy

## 📝 School Project

**TAC450** - Reinforcement Learning Project

This project demonstrates the AlphaZero algorithm applied to Gomoku, showcasing how combining neural networks with Monte Carlo Tree Search creates a powerful game-playing AI.

## 📚 References

- [AlphaGo Zero Paper](https://www.nature.com/articles/nature24270)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)

## License

MIT License - For educational purposes
