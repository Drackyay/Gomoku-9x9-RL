"""
AlphaZero-style Training for Gomoku

Combines MCTS + Neural Network:
1. Neural Network provides move probabilities (policy) and position value
2. MCTS uses NN to guide search
3. MCTS results train the NN to be better
4. Iterate: better NN → better MCTS → better training data → better NN

This is how AlphaGo Zero achieved superhuman play!
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from datetime import datetime
from tqdm import tqdm
import math


# ============== NEURAL NETWORK ==============

class GomokuNet(nn.Module):
    """
    Neural network for Gomoku (AlphaZero style).
    
    Input: Board state (3, 9, 9)
    Output: 
        - Policy: Probability distribution over 81 moves
        - Value: Expected outcome [-1, 1]
    """
    
    def __init__(self, board_size=9, num_channels=64, num_res_blocks=3):
        super().__init__()
        self.board_size = board_size
        
        # Initial convolution
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, board_size * board_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Input processing
        out = self.conv_input(x)
        
        # Residual blocks
        for block in self.res_blocks:
            out = block(out)
        
        # Heads
        policy = self.policy_head(out)
        value = self.value_head(out)
        
        return policy, value
    
    def predict(self, board, device="cuda"):
        """Get policy and value for a board position."""
        self.eval()
        with torch.no_grad():
            if isinstance(board, np.ndarray):
                board = torch.FloatTensor(board)
            if board.dim() == 3:
                board = board.unsqueeze(0)
            board = board.to(device)
            
            policy, value = self(board)
            policy = F.softmax(policy, dim=1)
            
            return policy.cpu().numpy()[0], value.cpu().item()


class ResBlock(nn.Module):
    """Residual block."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


# ============== MCTS WITH NEURAL NETWORK ==============

class MCTSNode:
    """MCTS node for AlphaZero."""
    
    def __init__(self, prior=1.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False
    
    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visits, c_puct=1.5):
        """UCB score with neural network prior."""
        exploitation = self.value
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def select_child(self, c_puct=1.5):
        """Select child with highest UCB."""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child


class AlphaZeroMCTS:
    """MCTS guided by neural network."""
    
    def __init__(self, model, device="cuda", num_simulations=100, c_puct=1.5):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def search(self, board, current_player, temperature=1.0, add_noise=True):
        """
        Run MCTS search and return action probabilities.
        
        Args:
            board: Current board state
            current_player: Player to move (1 or 2)
            temperature: Controls exploration (higher = more random)
        
        Returns:
            action_probs: Probability distribution over actions
        """
        root = MCTSNode()
        
        state = self._get_state(board, current_player)
        policy, _ = self.model.predict(state, self.device)
        
        valid_actions = self._get_valid_actions(board)
        self._expand_node(root, policy, valid_actions)
        
        if add_noise and valid_actions:
            noise = np.random.dirichlet([0.03] * len(valid_actions))
            epsilon = 0.25
            for a, n in zip(valid_actions, noise):
                child = root.children[a]
                child.prior = float((1 - epsilon) * child.prior + epsilon * n)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_board = board.copy()
            sim_player = current_player
            path = [node]
            actions = []
            
            # Selection: traverse to leaf
            while node.is_expanded:
                action, child = node.select_child(self.c_puct)
                if child is None:
                    break
                
                row, col = action // 9, action % 9
                sim_board[row, col] = sim_player
                sim_player = 3 - sim_player
                
                node = child
                path.append(node)
                actions.append(action)
                
                # Check terminal
                if self._check_terminal(sim_board, row, col):
                    break
            
            # Expansion and evaluation
            last_action = actions[-1] if actions else None
            if last_action is not None:
                row, col = last_action // 9, last_action % 9
                terminal, winner = self._is_terminal(sim_board, row, col)
            else:
                terminal, winner = False, None
            
            if terminal:
                if winner == 0 or winner is None:
                    value = 0.0
                elif winner == current_player:
                    value = 1.0
                else:
                    value = -1.0
            else:
                state = self._get_state(sim_board, sim_player)
                policy, value = self.model.predict(state, self.device)
                
                if sim_player != current_player:
                    value = -value
                
                valid = self._get_valid_actions(sim_board)
                if valid:
                    self._expand_node(node, policy, valid)
            
            v = value
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += v
                v = -v
        
        # Get action probabilities from visit counts
        action_probs = np.zeros(81)
        
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        
        if action_probs.sum() > 0:
            if temperature == 0:
                # Deterministic: pick best
                best = np.argmax(action_probs)
                action_probs = np.zeros(81)
                action_probs[best] = 1.0
            else:
                # Apply temperature
                action_probs = action_probs ** (1.0 / temperature)
                action_probs = action_probs / action_probs.sum()
        
        return action_probs
    
    def _get_state(self, board, player):
        """Convert board to neural network input."""
        state = np.zeros((3, 9, 9), dtype=np.float32)
        state[0] = (board == player).astype(np.float32)
        state[1] = (board == (3 - player)).astype(np.float32)
        if player == 1:
            state[2] = 1.0
        return state
    
    def _get_valid_actions(self, board):
        """Get valid actions."""
        valid = []
        for r in range(9):
            for c in range(9):
                if board[r, c] == 0:
                    valid.append(r * 9 + c)
        return valid
    
    def _expand_node(self, node, policy, valid_actions):
        """Expand node with children."""
        if not valid_actions:
            return
        priors = np.array([policy[a] for a in valid_actions], dtype=np.float32)
        s = priors.sum()
        if s > 0:
            priors /= s
        else:
            priors = np.ones(len(valid_actions), dtype=np.float32) / len(valid_actions)
        for action, p in zip(valid_actions, priors):
            if action not in node.children:
                node.children[action] = MCTSNode(prior=float(p))
        node.is_expanded = True
    
    def _check_terminal(self, board, row, col):
        """Quick terminal check."""
        player = board[row, col]
        if player == 0:
            return False
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for d in [1, -1]:
                r, c = row + d*dr, col + d*dc
                while 0 <= r < 9 and 0 <= c < 9 and board[r, c] == player:
                    count += 1
                    r += d*dr
                    c += d*dc
            if count >= 5:
                return True
        return False
    
    def _is_terminal(self, board, row, col):
        """Check if game ended."""
        if row is None:
            return False, None
        
        player = board[row, col]
        if player == 0:
            return False, None
        
        # Check win
        if self._check_terminal(board, row, col):
            return True, player
        
        # Check draw
        if np.all(board != 0):
            return True, 0
        
        return False, None


# ============== SELF-PLAY DATA GENERATION ==============

class ReplayBuffer:
    """Store training examples."""
    
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return (
            np.array(states),
            np.array(policies),
            np.array(values)
        )
    
    def __len__(self):
        return len(self.buffer)


def play_against_opponent(mcts: AlphaZeroMCTS, opponent_move_fn, model_plays_first=True, temperature_threshold=15):
    """
    Play one game against an external opponent (e.g., heuristic AI).
    
    Args:
        mcts: AlphaZero MCTS for the model
        opponent_move_fn: Function that takes (board, player) and returns action
        model_plays_first: If True, model is player 1, else player 2
        temperature_threshold: Temperature threshold for exploration
    
    Returns:
        List of (state, policy, result) tuples for model's moves only
    """
    board = np.zeros((9, 9), dtype=np.int8)
    model_player = 1 if model_plays_first else 2
    opponent_player = 3 - model_player
    current_player = 1
    game_history = []
    move_count = 0
    
    while True:
        if current_player == model_player:
            state = np.zeros((3, 9, 9), dtype=np.float32)
            state[0] = (board == model_player).astype(np.float32)
            state[1] = (board == opponent_player).astype(np.float32)
            if model_player == 1:
                state[2] = 1.0
            
            temp = 1.0 if move_count < temperature_threshold else 0.1
            action_probs = mcts.search(board, model_player, temperature=temp, add_noise=True)
            game_history.append((state.copy(), action_probs.copy(), model_player))
            
            action = np.random.choice(81, p=action_probs)
        else:
            action = opponent_move_fn(board.copy(), opponent_player)
        
        row, col = action // 9, action % 9
        board[row, col] = current_player
        move_count += 1
        
        winner = check_winner(board, row, col, current_player)
        if winner is not None or move_count >= 81:
            results = []
            for state, policy, player in game_history:
                if winner == 0 or winner is None:
                    value = 0.0
                elif winner == model_player:
                    value = 1.0
                else:
                    value = -1.0
                results.append((state, policy, value))
            return results
        
        current_player = 3 - current_player


def self_play_game(mcts: AlphaZeroMCTS, temperature_threshold=15):
    """
    Play one game of self-play.
    
    Returns list of (state, policy, result) tuples.
    """
    board = np.zeros((9, 9), dtype=np.int8)
    current_player = 1
    game_history = []
    move_count = 0
    
    while True:
        state = np.zeros((3, 9, 9), dtype=np.float32)
        state[0] = (board == current_player).astype(np.float32)
        state[1] = (board == (3 - current_player)).astype(np.float32)
        if current_player == 1:
            state[2] = 1.0
        
        temp = 1.0 if move_count < temperature_threshold else 0.1
        action_probs = mcts.search(board, current_player, temperature=temp, add_noise=True)
        game_history.append((state.copy(), action_probs.copy(), current_player))
        
        action = np.random.choice(81, p=action_probs)
        row, col = action // 9, action % 9
        
        board[row, col] = current_player
        move_count += 1
        
        winner = check_winner(board, row, col, current_player)
        if winner is not None or move_count >= 81:
            results = []
            for state, policy, player in game_history:
                if winner == 0 or winner is None:
                    value = 0.0
                elif winner == player:
                    value = 1.0
                else:
                    value = -1.0
                results.append((state, policy, value))
            return results
        
        current_player = 3 - current_player


def check_winner(board, row, col, player):
    """Check if player won. Returns winner or None."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dr, dc in directions:
        count = 1
        for d in [1, -1]:
            r, c = row + d*dr, col + d*dc
            while 0 <= r < 9 and 0 <= c < 9 and board[r, c] == player:
                count += 1
                r += d*dr
                c += d*dc
        if count >= 5:
            return player
    
    if np.all(board != 0):
        return 0  # Draw
    
    return None


# ============== TRAINING ==============

def evaluate_loss(model, replay_buffer, batch_size=256, device="cuda"):
    """Evaluate model loss on replay buffer without training."""
    if len(replay_buffer) < batch_size:
        return 0.0
    
    model.eval()
    actual_batch_size = min(batch_size, len(replay_buffer))
    
    with torch.no_grad():
        states, policies, values = replay_buffer.sample(actual_batch_size)
        
        states_t = torch.FloatTensor(states).to(device)
        policies_t = torch.FloatTensor(policies).to(device)
        values_t = torch.FloatTensor(values).unsqueeze(1).to(device)
        
        pred_policies, pred_values = model(states_t)
        
        policy_loss = -torch.mean(torch.sum(policies_t * F.log_softmax(pred_policies, dim=1), dim=1))
        value_loss = F.mse_loss(pred_values, values_t)
        
        loss = policy_loss + value_loss
    
    return loss.item()


def train_network(model, replay_buffer, optimizer, batch_size=256, epochs=5, device="cuda"):
    """Train neural network on self-play data."""
    
    if len(replay_buffer) < batch_size:
        return 0.0
    
    model.train()
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    num_batches = 0
    
    actual_batch_size = min(batch_size, len(replay_buffer))
    
    for _ in range(epochs):
        states, policies, values = replay_buffer.sample(actual_batch_size)
        
        states_t = torch.FloatTensor(states).to(device)
        policies_t = torch.FloatTensor(policies).to(device)
        values_t = torch.FloatTensor(values).unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        
        pred_policies, pred_values = model(states_t)
        
        policy_loss = -torch.mean(torch.sum(policies_t * F.log_softmax(pred_policies, dim=1), dim=1))
        value_loss = F.mse_loss(pred_values, values_t)
        
        loss = policy_loss + value_loss
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_policy_loss = total_policy_loss / max(num_batches, 1)
    avg_value_loss = total_value_loss / max(num_batches, 1)
    
    return avg_loss


def simple_heuristic_move(board, player):
    """Simple heuristic for evaluation."""
    size = 9
    opponent = 3 - player
    
    def score(r, c):
        if board[r, c] != 0:
            return -1e9
        s = 0
        dirs = [(0,1), (1,0), (1,1), (1,-1)]
        
        for dr, dc in dirs:
            # Count our line
            cnt, ops = 1, 0
            nr, nc = r + dr, c + dc
            while 0 <= nr < size and 0 <= nc < size and board[nr, nc] == player:
                cnt += 1
                nr += dr
                nc += dc
            if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == 0:
                ops += 1
            nr, nc = r - dr, c - dc
            while 0 <= nr < size and 0 <= nc < size and board[nr, nc] == player:
                cnt += 1
                nr -= dr
                nc -= dc
            if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == 0:
                ops += 1
            
            if cnt >= 5: s += 100000
            elif cnt == 4 and ops >= 1: s += 10000
            elif cnt == 3 and ops == 2: s += 1000
            
            # Count opponent line (blocking)
            cnt2, ops2 = 1, 0
            board[r, c] = opponent
            nr, nc = r + dr, c + dc
            while 0 <= nr < size and 0 <= nc < size and board[nr, nc] == opponent:
                cnt2 += 1
                nr += dr
                nc += dc
            nr, nc = r - dr, c - dc
            while 0 <= nr < size and 0 <= nc < size and board[nr, nc] == opponent:
                cnt2 += 1
                nr -= dr
                nc -= dc
            board[r, c] = 0
            
            if cnt2 >= 5: s += 50000
            elif cnt2 == 4: s += 5000
            elif cnt2 == 3: s += 500
        
        s += (4 - abs(r - 4)) * 3 + (4 - abs(c - 4)) * 3
        return s
    
    best = -1e9
    best_moves = []
    for r in range(size):
        for c in range(size):
            sc = score(r, c)
            if sc > best:
                best = sc
                best_moves = [(r, c)]
            elif sc == best:
                best_moves.append((r, c))
    
    if best_moves:
        r, c = best_moves[np.random.randint(len(best_moves))]
        return r * 9 + c
    return 40


def evaluate_model(model, device="cuda", num_games=20):
    """Evaluate model against simple heuristic."""
    wins = 0
    
    mcts = AlphaZeroMCTS(model, device=device, num_simulations=100)
    
    for _ in range(num_games):
        board = np.zeros((9, 9), dtype=np.int8)
        current = 1
        
        while True:
            if current == 1:
                probs = mcts.search(board, 1, temperature=0.1, add_noise=False)
                action = np.argmax(probs)
            else:
                # Heuristic plays
                action = simple_heuristic_move(board.copy(), 2)
            
            row, col = action // 9, action % 9
            board[row, col] = current
            
            winner = check_winner(board, row, col, current)
            if winner is not None:
                if winner == 1:
                    wins += 1
                break
            
            if np.all(board != 0):
                break
            
            current = 3 - current
    
    return wins / num_games


# ============== MAIN TRAINING LOOP ==============

def train_alphazero(
    num_iterations=50,
    games_per_iteration=50,
    mcts_simulations=100,
    batch_size=256,
    train_epochs=10,
    lr=5e-4,
    resume=True,
    adaptive_simulations=True,
    use_curriculum=True,
    opponent_ratio=0.5,
):
    """
    AlphaZero training loop.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize
    model = GomokuNet(board_size=9, num_channels=64, num_res_blocks=3).to(device)
    replay_buffer = ReplayBuffer(max_size=50000)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Try to resume from checkpoint
    start_iteration = 0
    models_dir = project_root / "models"
    checkpoint_path = models_dir / "alphazero_latest.pth"
    
    if resume and checkpoint_path.exists():
        try:
            model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
            print(f"✓ Resumed from checkpoint: {checkpoint_path}")
            # Note: We can't restore iteration number, so we start fresh but with trained weights
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
    
    mcts = AlphaZeroMCTS(model, device=device, num_simulations=mcts_simulations)
    
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print("ALPHAZERO TRAINING")
    print(f"{'='*60}")
    print(f"Iterations: {num_iterations}")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"MCTS simulations: {mcts_simulations}")
    print(f"{'='*60}\n")
    
    best_win_rate = 0.0
    current_simulations = mcts_simulations
    
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
        
        if adaptive_simulations:
            if iteration < 3:
                current_simulations = 25
            elif iteration < 10:
                current_simulations = 50
            elif iteration < 20:
                current_simulations = 100
            else:
                current_simulations = mcts_simulations
            mcts.num_simulations = current_simulations
            print(f"Using {current_simulations} MCTS simulations")
        
        if use_curriculum:
            if iteration < num_iterations * 0.3:
                opponent_games = 0
                self_play_games = games_per_iteration
                training_mode = "self-play (warm-up)"
            elif iteration < num_iterations * 0.8:
                opponent_games = int(games_per_iteration * opponent_ratio)
                self_play_games = games_per_iteration - opponent_games
                training_mode = "mixed (self-play + heuristic)"
            else:
                opponent_games = int(games_per_iteration * (opponent_ratio * 0.5))
                self_play_games = games_per_iteration - opponent_games
                training_mode = "self-play-heavy"
        else:
            opponent_games = int(games_per_iteration * opponent_ratio)
            self_play_games = games_per_iteration - opponent_games
            training_mode = "mixed (fixed ratio)"
        
        print(f"Training mode: {training_mode}")
        print(f"  Self-play games: {self_play_games}, Opponent games: {opponent_games}")
        
        total_samples = 0
        for game_idx in tqdm(range(games_per_iteration)):
            if game_idx < opponent_games:
                model_first = (game_idx % 2 == 0)
                game_data = play_against_opponent(
                    mcts, 
                    simple_heuristic_move,
                    model_plays_first=model_first
                )
            else:
                game_data = self_play_game(mcts)
            
            for state, policy, value in game_data:
                replay_buffer.add(state, policy, value)
                total_samples += 1
        
        print(f"Replay buffer size: {len(replay_buffer)}")
        print(f"New samples this iteration: {total_samples}")
        
        if len(replay_buffer) < batch_size:
            print(f"Warning: Replay buffer ({len(replay_buffer)}) < batch_size ({batch_size})")
            print("Skipping training this iteration...")
            continue
        
        print("Training neural network...")
        
        initial_loss = evaluate_loss(model, replay_buffer, batch_size, device=device)
        print(f"  Initial loss (before training): {initial_loss:.4f}")
        
        for epoch in range(train_epochs):
            loss = train_network(model, replay_buffer, optimizer, batch_size, epochs=1, device=device)
            if epoch == 0 or (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch + 1}/{train_epochs}: Loss = {loss:.4f}")
        
        final_loss = evaluate_loss(model, replay_buffer, batch_size, device=device)
        loss_reduction = initial_loss - final_loss
        loss_reduction_pct = (loss_reduction / initial_loss) * 100 if initial_loss > 0 else 0
        
        print(f"Loss: {initial_loss:.4f} → {final_loss:.4f} (reduced by {loss_reduction:.4f}, {loss_reduction_pct:.1f}%)")
        
        if loss_reduction < 0.01 or loss_reduction_pct < 1.0:
            print(" Warning: Loss reduction is very small!")
            print("   This suggests the model is not learning effectively.")
            print("   Consider:")
            print("     - The model may need more training iterations")
            print("     - Try reducing learning rate further")
            print("     - Check if training data quality is good")
        
        mcts.model = model
        
        # Evaluation every 5 iterations
        if (iteration + 1) % 5 == 0:
            print("Evaluating...")
            win_rate = evaluate_model(model, device=device, num_games=20)
            print(f"Win rate vs heuristic: {win_rate:.0%}")
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(model.state_dict(), str(models_dir / "alphazero_best.pth"))
                print(f"New best model saved!")
        
        # Save checkpoint
        torch.save(model.state_dict(), str(models_dir / "alphazero_latest.pth"))
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Best win rate: {best_win_rate:.0%}")
    print(f"{'='*60}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't load checkpoint")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive simulations")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--opponent-ratio", type=float, default=0.5, help="Ratio of games vs opponent (0.0-1.0)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()
    
    train_alphazero(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        mcts_simulations=args.simulations,
        resume=not args.no_resume,
        adaptive_simulations=not args.no_adaptive,
        use_curriculum=not args.no_curriculum,
        opponent_ratio=args.opponent_ratio,
        lr=args.lr,
    )

