"""
Monte Carlo Tree Search (MCTS) for Gomoku

This is how AlphaGo actually works (simplified version).
MCTS doesn't need training - it searches for the best move in real-time.
"""

import numpy as np
import math
from typing import Optional, List, Tuple
from dataclasses import dataclass
import time


@dataclass
class MCTSConfig:
    """MCTS configuration."""
    num_simulations: int = 800      # More = stronger but slower
    c_puct: float = 1.4             # Exploration constant
    max_time: float = 5.0           # Max seconds per move
    use_heuristic: bool = True      # Use heuristic for evaluation


class Node:
    """MCTS Tree Node."""
    
    def __init__(self, parent=None, prior: float = 1.0, action: int = -1):
        self.parent = parent
        self.action = action        # Action that led to this node
        self.prior = prior          # Prior probability
        self.children = {}          # action -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float) -> float:
        """Upper Confidence Bound score."""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.value
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def select_child(self, c_puct: float) -> 'Node':
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, valid_actions: List[int], priors: Optional[np.ndarray] = None):
        """Expand node with children for valid actions."""
        if priors is None:
            priors = np.ones(len(valid_actions)) / len(valid_actions)
        
        for i, action in enumerate(valid_actions):
            if action not in self.children:
                self.children[action] = Node(
                    parent=self, 
                    prior=priors[i] if i < len(priors) else 1.0/len(valid_actions),
                    action=action
                )
        
        self.is_expanded = True
    
    def backup(self, value: float):
        """Backpropagate value up the tree."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent
            node = node.parent


class GomokuState:
    """Game state for MCTS."""
    
    def __init__(self, board: np.ndarray = None, current_player: int = 1):
        if board is None:
            self.board = np.zeros((9, 9), dtype=np.int8)
        else:
            self.board = board.copy()
        self.current_player = current_player
        self.winner = None
        self.last_move = None
    
    def copy(self) -> 'GomokuState':
        state = GomokuState(self.board, self.current_player)
        state.winner = self.winner
        state.last_move = self.last_move
        return state
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions."""
        valid = []
        for r in range(9):
            for c in range(9):
                if self.board[r, c] == 0:
                    valid.append(r * 9 + c)
        return valid
    
    def make_move(self, action: int) -> bool:
        """Make a move. Returns True if game ended."""
        row, col = action // 9, action % 9
        
        if self.board[row, col] != 0:
            return True  # Invalid move
        
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        
        # Check win
        if self._check_win(row, col, self.current_player):
            self.winner = self.current_player
            return True
        
        # Check draw
        if len(self.get_valid_actions()) == 0:
            self.winner = 0  # Draw
            return True
        
        self.current_player = 3 - self.current_player
        return False
    
    def _check_win(self, row: int, col: int, player: int) -> bool:
        """Check if player won."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            
            # Positive direction
            r, c = row + dr, col + dc
            while 0 <= r < 9 and 0 <= c < 9 and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            
            # Negative direction
            r, c = row - dr, col - dc
            while 0 <= r < 9 and 0 <= c < 9 and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                return True
        
        return False
    
    def is_terminal(self) -> bool:
        return self.winner is not None
    
    def get_result(self, player: int) -> float:
        """Get result from player's perspective."""
        if self.winner == 0:
            return 0.0
        elif self.winner == player:
            return 1.0
        else:
            return -1.0


def evaluate_position(state: GomokuState) -> float:
    """
    Heuristic evaluation of board position.
    Returns value from current player's perspective.
    """
    if state.is_terminal():
        return state.get_result(state.current_player)
    
    player = state.current_player
    opponent = 3 - player
    board = state.board
    
    def count_patterns(p):
        """Count threatening patterns for player p."""
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for r in range(9):
            for c in range(9):
                if board[r, c] != p:
                    continue
                
                for dr, dc in directions:
                    count = 1
                    open_ends = 0
                    
                    # Count forward
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < 9 and 0 <= nc < 9 and board[nr, nc] == p:
                        count += 1
                        nr += dr
                        nc += dc
                    if 0 <= nr < 9 and 0 <= nc < 9 and board[nr, nc] == 0:
                        open_ends += 1
                    
                    # Only count once per line (check if this is the start)
                    pr, pc = r - dr, c - dc
                    if 0 <= pr < 9 and 0 <= pc < 9 and board[pr, pc] == p:
                        continue
                    
                    # Check backward for open end
                    if 0 <= pr < 9 and 0 <= pc < 9 and board[pr, pc] == 0:
                        open_ends += 1
                    
                    # Score based on pattern
                    if count >= 5:
                        score += 100000
                    elif count == 4:
                        if open_ends == 2:
                            score += 10000  # Open four = win
                        elif open_ends == 1:
                            score += 1000   # Half-open four
                    elif count == 3:
                        if open_ends == 2:
                            score += 500    # Open three
                        elif open_ends == 1:
                            score += 50     # Half-open three
                    elif count == 2 and open_ends == 2:
                        score += 10         # Open two
        
        return score
    
    my_score = count_patterns(player)
    opp_score = count_patterns(opponent)
    
    # Normalize to [-1, 1]
    diff = my_score - opp_score
    return np.tanh(diff / 1000)


def get_action_priors(state: GomokuState) -> Tuple[List[int], np.ndarray]:
    """
    Get prior probabilities for actions using heuristic.
    Higher scores = higher probability.
    """
    valid_actions = state.get_valid_actions()
    scores = np.zeros(len(valid_actions))
    
    player = state.current_player
    opponent = 3 - player
    board = state.board
    
    for i, action in enumerate(valid_actions):
        row, col = action // 9, action % 9
        score = 0
        
        # Check what happens if we play here
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            # Our patterns
            count, opens = count_line(board, row, col, dr, dc, player)
            if count >= 4:
                score += 10000
            elif count == 3 and opens >= 1:
                score += 1000
            elif count == 2 and opens == 2:
                score += 100
            
            # Blocking opponent
            count, opens = count_line(board, row, col, dr, dc, opponent)
            if count >= 4:
                score += 5000
            elif count == 3 and opens == 2:
                score += 500
        
        # Center preference
        score += (4 - abs(row - 4)) * 10 + (4 - abs(col - 4)) * 10
        
        scores[i] = score
    
    # Convert to probabilities
    if scores.max() > scores.min():
        probs = scores - scores.min()
        probs = probs / probs.sum()
    else:
        probs = np.ones(len(valid_actions)) / len(valid_actions)
    
    return valid_actions, probs


def count_line(board, row, col, dr, dc, player):
    """Count consecutive stones if we place at (row, col)."""
    count = 1
    opens = 0
    
    # Forward
    r, c = row + dr, col + dc
    while 0 <= r < 9 and 0 <= c < 9 and board[r, c] == player:
        count += 1
        r += dr
        c += dc
    if 0 <= r < 9 and 0 <= c < 9 and board[r, c] == 0:
        opens += 1
    
    # Backward
    r, c = row - dr, col - dc
    while 0 <= r < 9 and 0 <= c < 9 and board[r, c] == player:
        count += 1
        r -= dr
        c -= dc
    if 0 <= r < 9 and 0 <= c < 9 and board[r, c] == 0:
        opens += 1
    
    return count, opens


def random_rollout(state: GomokuState, max_moves: int = 50) -> float:
    """
    Random rollout from current state.
    Returns result from original player's perspective.
    """
    original_player = state.current_player
    state = state.copy()
    
    for _ in range(max_moves):
        if state.is_terminal():
            break
        
        valid = state.get_valid_actions()
        if not valid:
            break
        
        # Semi-random: prefer good moves
        _, priors = get_action_priors(state)
        action = np.random.choice(valid, p=priors)
        state.make_move(action)
    
    return state.get_result(original_player)


class MCTS:
    """Monte Carlo Tree Search."""
    
    def __init__(self, config: MCTSConfig = None):
        self.config = config or MCTSConfig()
    
    def search(self, state: GomokuState) -> int:
        """
        Run MCTS and return best action.
        """
        root = Node()
        
        # Expand root
        valid_actions, priors = get_action_priors(state)
        root.expand(valid_actions, priors)
        
        start_time = time.time()
        simulations = 0
        
        while simulations < self.config.num_simulations:
            # Check time limit
            if time.time() - start_time > self.config.max_time:
                break
            
            # Selection: traverse tree to leaf
            node = root
            sim_state = state.copy()
            
            while node.is_expanded and not sim_state.is_terminal():
                node = node.select_child(self.config.c_puct)
                if node is None:
                    break
                sim_state.make_move(node.action)
            
            # Expansion
            if not sim_state.is_terminal() and node is not None:
                valid, priors = get_action_priors(sim_state)
                if valid:
                    node.expand(valid, priors)
            
            # Evaluation
            if sim_state.is_terminal():
                value = sim_state.get_result(state.current_player)
            elif self.config.use_heuristic:
                # Use heuristic + short rollout
                value = 0.7 * evaluate_position(sim_state) + 0.3 * random_rollout(sim_state, max_moves=20)
            else:
                value = random_rollout(sim_state)
            
            # Backup
            if node is not None:
                node.backup(value)
            
            simulations += 1
        
        # Select most visited action
        best_action = None
        best_visits = -1
        
        for action, child in root.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action
        
        return best_action
    
    def get_move(self, board: np.ndarray, player: int) -> int:
        """
        Get best move for given board position.
        
        Args:
            board: 9x9 numpy array (0=empty, 1=black, 2=white)
            player: Current player (1 or 2)
        
        Returns:
            Best action (0-80)
        """
        state = GomokuState(board, player)
        
        # Quick checks for immediate wins/blocks
        valid_actions = state.get_valid_actions()
        
        for action in valid_actions:
            row, col = action // 9, action % 9
            
            # Check if we can win
            test_state = state.copy()
            test_state.make_move(action)
            if test_state.winner == player:
                return action
        
        for action in valid_actions:
            row, col = action // 9, action % 9
            
            # Check if opponent can win (must block)
            test_board = board.copy()
            test_board[row, col] = 3 - player
            test_state = GomokuState(test_board, player)
            if test_state._check_win(row, col, 3 - player):
                return action
        
        # Run full MCTS
        return self.search(state)


# Easy-to-use function
def mcts_move(board: np.ndarray, player: int, 
              simulations: int = 500, max_time: float = 3.0) -> int:
    """
    Get MCTS move for given position.
    
    Args:
        board: 9x9 numpy array
        player: Current player (1 or 2)
        simulations: Number of MCTS simulations (more = stronger)
        max_time: Maximum time in seconds
    
    Returns:
        Best action (0-80)
    """
    config = MCTSConfig(
        num_simulations=simulations,
        max_time=max_time,
        use_heuristic=True
    )
    mcts = MCTS(config)
    return mcts.get_move(board, player)


# Test
if __name__ == "__main__":
    print("Testing MCTS...")
    
    # Create test position
    board = np.zeros((9, 9), dtype=np.int8)
    board[4, 4] = 1  # Black center
    board[4, 5] = 2  # White
    board[3, 4] = 1  # Black
    board[3, 5] = 2  # White
    board[5, 4] = 1  # Black threatens 3 in a row
    
    print("Test board:")
    symbols = {0: ".", 1: "X", 2: "O"}
    for r in range(9):
        print(" ".join(symbols[board[r, c]] for c in range(9)))
    
    print("\nRunning MCTS (White to play, should block)...")
    action = mcts_move(board, player=2, simulations=300, max_time=2.0)
    row, col = action // 9, action % 9
    print(f"MCTS chose: ({row}, {col})")
    
    # Verify it blocks
    if row == 2 and col == 4:
        print("✓ Correct! MCTS blocked the threat.")
    elif row == 6 and col == 4:
        print("✓ Correct! MCTS blocked the other side.")
    else:
        print(f"Move at ({row}, {col})")
    
    print("\nMCTS is ready to use!")

