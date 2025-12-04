# 🎮 Gomoku 9×9 AI with AlphaZero

A reinforcement learning project that trains an AI to play Gomoku (Five in a Row) using an **AlphaZero-style algorithm** (Monte Carlo Tree Search + Residual Neural Network), with a modern **React web UI** and a **Flask API backend**.

## 📋 Project Structure

```bash
Gomoku-9x9-RL/
├── env/
│   ├── __init__.py
│   ├── gomoku_env.py      # Gymnasium environment
│   └── test_env.py        # Environment tests
├── rl/
│   ├── __init__.py
│   ├── alphazero_train.py # AlphaZero-style training (MCTS + NN)
│   └── mcts_ai.py         # Heuristic MCTS baseline (no NN)
├── api/
│   ├── __init__.py
│   └── app.py             # Flask API serving the trained model (AlphaZero or MCTS)
├── frontend/
│   ├── public/
│   └── src/               # React single-page app (Gomoku board UI)
├── models/                # Saved PyTorch models (alphazero_latest.pth / alphazero_best.pth)
├── requirements.txt       # Core Python deps (RL, env, etc.)
└── README.md
```

## 🚀 Quick Start

### 1. Python Environment

```bash
cd Gomoku-9x9-RL
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# install API dependencies
cd api
pip install -r requirements.txt
cd ..
```

### 2. Start the AlphaZero / MCTS API (backend)

```bash
cd api
conda activate tf210  # or your env with PyTorch + CUDA
python app.py
```

The backend will start on `http://localhost:5000` and load `models/alphazero_latest.pth`
or `models/alphazero_best.pth`. If no trained model exists, it will fall back to the heuristic
MCTS bot.

### 3. Start the React Web UI (frontend)

In a **separate terminal**:

```bash
cd Gomoku-9x9-RL/frontend
npm install        # first time only
npm start
```

Open `http://localhost:3000` in your browser to play against the AI on a 9×9 Gomoku board.

### 4. Train / Continue the AlphaZero Model

From the project root:

```bash
conda activate tf210  # or your PyTorch env
cd Gomoku-9x9-RL
python rl/alphazero_train.py --iterations 100 --games 50 --simulations 100 --lr 5e-4
```

This will:

- run curriculum self-play (vs itself + vs heuristic opponent)
- train the neural network with policy + value loss
- periodically evaluate vs the heuristic bot
- save checkpoints to `models/alphazero_latest.pth` and the best model to `models/alphazero_best.pth`

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
  - 3 Residual Blocks
  - 64 channels
  - Policy head: predicts move probabilities
  - Value head: predicts game outcome

MCTS:
  - 25 → 50 → 100+ simulations per move (adaptive)
  - UCB exploration constant: 1.5
  - Temperature: 1.0 (early game), 0.1 (late game)
  - Dirichlet noise at root for exploration
```

### Training Loop

1. **Self-play**  
   - Early iterations: mostly self-play (model vs itself)  
   - Mid iterations: mixed games vs heuristic opponent (curriculum learning)  
   - Late iterations: self-play heavy to refine strong policies
2. **Training**  
   - Sample mini-batches from a replay buffer (up to 50k positions)  
   - Optimize combined loss: policy cross-entropy + value MSE  
   - Use Adam optimizer with gradient clipping
3. **Evaluation**  
   - Play against a heuristic Gomoku bot  
   - Track win rate and save the best-performing model
4. **Repeat** for `num_iterations` (default 50–100)

## 🔧 Requirements

- **Python** 3.8+
- **PyTorch** 2.0+ (CUDA recommended, e.g. RTX 30xx)
- **Gymnasium** (for the training environment)
- **Flask** + **flask-cors** (API backend)
- **Node.js** + **npm** (for the React frontend)
- **NumPy**, **tqdm** and other utilities (see `requirements.txt` and `api/requirements.txt`)

## 📝 School Project

**TAC450** – Reinforcement Learning Project  
Weixi Chen · Hongyu Wei · Minhao Li  
University of Southern California

This project demonstrates an AlphaZero-style algorithm applied to 9×9 Gomoku, showing how
combining a deep residual network with Monte Carlo Tree Search can produce a strong game-playing AI,
and how to deploy it as a modern web application (Flask API + React UI).

## 📚 References

- [AlphaGo Zero Paper](https://www.nature.com/articles/nature24270)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)

## License

MIT License - For educational purposes
