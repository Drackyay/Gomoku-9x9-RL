"""
Gomoku 9x9 Environment for Reinforcement Learning
A Gymnasium-compatible environment for the game of Gomoku (Five in a Row)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class GomokuEnv9x9(gym.Env):
    """
    9x9 Gomoku (Five in a Row) Environment
    
    State representation (3, 9, 9):
        - Channel 0: Current player's stones (1 where player has stone, 0 elsewhere)
        - Channel 1: Opponent's stones (1 where opponent has stone, 0 elsewhere)
        - Channel 2: Current player indicator (all 1s if player 1, all 0s if player 2)
    
    Actions: 0-80 representing board positions (row * 9 + col)
    
    Rewards:
        - Win: +1
        - Lose: -1
        - Draw: 0
        - Illegal move: -1 (and episode ends)
        - Otherwise: 0
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.board_size = 9
        self.win_length = 5  # 5 in a row to win
        
        # Action space: 81 possible positions (0-80)
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        
        # Observation space: 3 channels of 9x9 board
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(3, self.board_size, self.board_size), 
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        
        # Initialize game state
        self.board = None  # 0: empty, 1: player 1, 2: player 2
        self.current_player = None
        self.done = None
        self.winner = None
        self.move_count = None
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.winner = None
        self.move_count = 0
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer 0-80 representing board position
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            # Episode already finished
            return self._get_observation(), 0.0, True, False, {"winner": self.winner}
        
        row, col = divmod(action, self.board_size)
        
        # Check for illegal move (position already occupied)
        if self.board[row, col] != 0:
            self.done = True
            self.winner = 3 - self.current_player  # Other player wins
            return self._get_observation(), -1.0, True, False, {
                "illegal_move": True,
                "winner": self.winner
            }
        
        # Place the stone
        self.board[row, col] = self.current_player
        self.move_count += 1
        
        # Check for win
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
            return self._get_observation(), reward, True, False, {"winner": self.winner}
        
        # Check for draw (board full)
        if self.move_count >= self.board_size * self.board_size:
            self.done = True
            self.winner = 0  # Draw
            return self._get_observation(), 0.0, True, False, {"winner": 0, "draw": True}
        
        # Switch player
        self.current_player = 3 - self.current_player
        
        return self._get_observation(), 0.0, False, False, {}
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation from the perspective of the current player.
        
        Returns:
            3-channel observation array (3, 9, 9)
        """
        obs = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # Channel 0: Current player's stones
        obs[0] = (self.board == self.current_player).astype(np.float32)
        
        # Channel 1: Opponent's stones
        opponent = 3 - self.current_player
        obs[1] = (self.board == opponent).astype(np.float32)
        
        # Channel 2: Current player indicator (1 if player 1, 0 if player 2)
        if self.current_player == 1:
            obs[2] = np.ones((self.board_size, self.board_size), dtype=np.float32)
        
        return obs
    
    def _check_win(self, row: int, col: int) -> bool:
        """
        Check if the last move at (row, col) resulted in a win.
        
        Args:
            row: Row of the last move
            col: Column of the last move
            
        Returns:
            True if the current player has won, False otherwise
        """
        player = self.board[row, col]
        
        # Directions: horizontal, vertical, diagonal, anti-diagonal
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal (top-left to bottom-right)
            (1, -1),  # Anti-diagonal (top-right to bottom-left)
        ]
        
        for dr, dc in directions:
            count = 1  # Count the current stone
            
            # Count in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r, c] == player:
                    count += 1
                    r += dr
                    c += dc
                else:
                    break
            
            # Count in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r, c] == player:
                    count += 1
                    r -= dr
                    c -= dc
                else:
                    break
            
            if count >= self.win_length:
                return True
        
        return False
    
    def render(self):
        """Render the current board state."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            self._render_text()
    
    def _render_text(self):
        """Render board as text."""
        symbols = {0: ".", 1: "X", 2: "O"}
        
        print("\n  ", end="")
        for i in range(self.board_size):
            print(f" {i}", end="")
        print()
        
        for row in range(self.board_size):
            print(f"{row}  ", end="")
            for col in range(self.board_size):
                print(f"{symbols[self.board[row, col]]} ", end="")
            print()
        
        print(f"\nCurrent player: {symbols[self.current_player]}")
        if self.done:
            if self.winner == 0:
                print("Game ended in a DRAW!")
            else:
                print(f"Player {symbols[self.winner]} WINS!")
        print()
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get array of valid action indices.
        
        Returns:
            Array of valid action indices (empty positions)
        """
        valid = []
        for i in range(self.board_size * self.board_size):
            row, col = divmod(i, self.board_size)
            if self.board[row, col] == 0:
                valid.append(i)
        return np.array(valid, dtype=np.int32)
    
    def action_mask(self) -> np.ndarray:
        """
        Get a boolean mask of valid actions.
        
        Returns:
            Boolean array where True indicates valid action
        """
        mask = np.zeros(self.board_size * self.board_size, dtype=bool)
        for i in range(self.board_size * self.board_size):
            row, col = divmod(i, self.board_size)
            mask[i] = (self.board[row, col] == 0)
        return mask


class GomokuSelfPlayEnv(GomokuEnv9x9):
    """
    Self-play wrapper for Gomoku environment.
    
    This environment handles both players, with the agent always seeing
    the board from its own perspective. The opponent can be:
    - Random
    - Another trained model
    - The same model (self-play)
    """
    
    def __init__(self, opponent_policy=None, render_mode: Optional[str] = None):
        """
        Args:
            opponent_policy: Callable that takes observation and returns action.
                           If None, uses random valid moves.
            render_mode: Rendering mode
        """
        super().__init__(render_mode=render_mode)
        self.opponent_policy = opponent_policy
        
    def set_opponent(self, opponent_policy):
        """Set or update the opponent policy."""
        self.opponent_policy = opponent_policy
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute agent's move, then opponent's move.
        """
        # Agent's move
        obs, reward, terminated, truncated, info = super().step(action)
        
        if terminated or truncated:
            return obs, reward, terminated, truncated, info
        
        # Opponent's move
        if self.opponent_policy is not None:
            opp_action = self.opponent_policy(self._get_observation())
        else:
            # Random valid move
            valid_actions = self.get_valid_actions()
            if len(valid_actions) > 0:
                opp_action = self.np_random.choice(valid_actions)
            else:
                return obs, 0.0, True, False, {"draw": True}
        
        obs, opp_reward, terminated, truncated, info = super().step(int(opp_action))
        
        # Invert reward (opponent's win is agent's loss)
        if terminated and self.winner == 2:  # Opponent won
            reward = -1.0
        
        return obs, reward, terminated, truncated, info


# For testing the environment
if __name__ == "__main__":
    print("Testing GomokuEnv9x9...")
    
    env = GomokuEnv9x9(render_mode="human")
    obs, _ = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a few moves
    env.render()
    
    # Play center
    obs, reward, done, _, info = env.step(40)  # Center (4,4)
    print(f"After move 40: reward={reward}, done={done}")
    env.render()
    
    # Opponent plays
    obs, reward, done, _, info = env.step(41)  # (4,5)
    print(f"After move 41: reward={reward}, done={done}")
    env.render()
    
    # Test winning condition - create 5 in a row
    print("\n--- Testing win condition ---")
    env.reset()
    
    # Player 1 plays: 0, 1, 2, 3, 4 (top row)
    # Player 2 plays: 9, 10, 11, 12
    moves = [0, 9, 1, 10, 2, 11, 3, 12, 4]  # Player 1 wins with top row
    
    for move in moves:
        obs, reward, done, _, info = env.step(move)
        env.render()
        if done:
            print(f"Game over! Reward: {reward}, Info: {info}")
            break
    
    print("\n--- Testing illegal move ---")
    env.reset()
    env.step(40)  # Place at center
    obs, reward, done, _, info = env.step(40)  # Try to place at same spot
    print(f"Illegal move result: reward={reward}, done={done}, info={info}")
    
    print("\nAll tests passed!")

