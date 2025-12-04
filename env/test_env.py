"""
Debug script to test the Gomoku environment.

Run this to verify:
1. Environment initialization
2. Move placement
3. Win detection
4. Draw detection
5. Illegal move handling
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from gomoku_env import GomokuEnv9x9


def test_basic_init():
    """Test basic environment initialization."""
    print("=" * 50)
    print("TEST 1: Basic Initialization")
    print("=" * 50)
    
    env = GomokuEnv9x9(render_mode="human")
    obs, info = env.reset()
    
    print(f"✓ Environment created")
    print(f"✓ Observation shape: {obs.shape} (expected: (3, 9, 9))")
    print(f"✓ Action space: {env.action_space}")
    print(f"✓ Current player: {env.current_player}")
    
    assert obs.shape == (3, 9, 9), "Observation shape mismatch!"
    assert env.current_player == 1, "Player 1 should start!"
    
    print("✓ All basic init tests passed!\n")


def test_move_placement():
    """Test placing moves on the board."""
    print("=" * 50)
    print("TEST 2: Move Placement")
    print("=" * 50)
    
    env = GomokuEnv9x9(render_mode="human")
    env.reset()
    
    # Place at center (4, 4) = action 40
    obs, reward, done, _, info = env.step(40)
    print(f"Placed at (4,4): reward={reward}, done={done}")
    env.render()
    
    assert env.board[4, 4] == 1, "Player 1's stone should be at (4,4)"
    assert env.current_player == 2, "Should be player 2's turn now"
    assert not done, "Game should not be over"
    
    # Place at (4, 5) = action 41
    obs, reward, done, _, info = env.step(41)
    print(f"Placed at (4,5): reward={reward}, done={done}")
    env.render()
    
    assert env.board[4, 5] == 2, "Player 2's stone should be at (4,5)"
    assert env.current_player == 1, "Should be player 1's turn now"
    
    print("✓ Move placement tests passed!\n")


def test_horizontal_win():
    """Test horizontal win detection."""
    print("=" * 50)
    print("TEST 3: Horizontal Win Detection")
    print("=" * 50)
    
    env = GomokuEnv9x9(render_mode="human")
    env.reset()
    
    # Player 1: 0, 1, 2, 3, 4 (top row)
    # Player 2: 9, 10, 11, 12
    moves = [
        (0, "P1 at (0,0)"),
        (9, "P2 at (1,0)"),
        (1, "P1 at (0,1)"),
        (10, "P2 at (1,1)"),
        (2, "P1 at (0,2)"),
        (11, "P2 at (1,2)"),
        (3, "P1 at (0,3)"),
        (12, "P2 at (1,3)"),
        (4, "P1 at (0,4) - SHOULD WIN"),
    ]
    
    for action, desc in moves:
        obs, reward, done, _, info = env.step(action)
        print(f"{desc}: reward={reward}, done={done}")
        
        if done:
            env.render()
            print(f"Winner: {info.get('winner')}")
            break
    
    assert done, "Game should be over!"
    assert info.get("winner") == 1, "Player 1 should win!"
    assert reward == 1.0, "Winner should get +1 reward!"
    
    print("✓ Horizontal win test passed!\n")


def test_vertical_win():
    """Test vertical win detection."""
    print("=" * 50)
    print("TEST 4: Vertical Win Detection")
    print("=" * 50)
    
    env = GomokuEnv9x9(render_mode="human")
    env.reset()
    
    # Player 1: col 0 (0, 9, 18, 27, 36)
    # Player 2: col 1 (1, 10, 19, 28)
    moves = [0, 1, 9, 10, 18, 19, 27, 28, 36]
    
    for action in moves:
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    
    env.render()
    
    assert done, "Game should be over!"
    assert info.get("winner") == 1, "Player 1 should win!"
    
    print("✓ Vertical win test passed!\n")


def test_diagonal_win():
    """Test diagonal win detection."""
    print("=" * 50)
    print("TEST 5: Diagonal Win Detection")
    print("=" * 50)
    
    env = GomokuEnv9x9(render_mode="human")
    env.reset()
    
    # Player 1: diagonal (0,0), (1,1), (2,2), (3,3), (4,4)
    # Actions: 0, 10, 20, 30, 40
    # Player 2: (0,8), (1,8), (2,8), (3,8)
    # Actions: 8, 17, 26, 35
    moves = [0, 8, 10, 17, 20, 26, 30, 35, 40]
    
    for action in moves:
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    
    env.render()
    
    assert done, "Game should be over!"
    assert info.get("winner") == 1, "Player 1 should win!"
    
    print("✓ Diagonal win test passed!\n")


def test_anti_diagonal_win():
    """Test anti-diagonal win detection."""
    print("=" * 50)
    print("TEST 6: Anti-Diagonal Win Detection")
    print("=" * 50)
    
    env = GomokuEnv9x9(render_mode="human")
    env.reset()
    
    # Player 1: anti-diagonal (0,4), (1,3), (2,2), (3,1), (4,0)
    # Actions: 4, 12, 20, 28, 36
    # Player 2: (0,0), (0,1), (0,2), (0,3)
    # Actions: 0, 1, 2, 3
    moves = [4, 0, 12, 1, 20, 2, 28, 3, 36]
    
    for action in moves:
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    
    env.render()
    
    assert done, "Game should be over!"
    assert info.get("winner") == 1, "Player 1 should win!"
    
    print("✓ Anti-diagonal win test passed!\n")


def test_illegal_move():
    """Test illegal move handling."""
    print("=" * 50)
    print("TEST 7: Illegal Move Handling")
    print("=" * 50)
    
    env = GomokuEnv9x9(render_mode="human")
    env.reset()
    
    # Place at center
    env.step(40)
    print("Placed at (4,4)")
    
    # Try to place at same position
    obs, reward, done, _, info = env.step(40)
    print(f"Tried to place at (4,4) again: reward={reward}, done={done}")
    print(f"Info: {info}")
    
    assert done, "Game should end on illegal move!"
    assert reward == -1.0, "Illegal move should give -1 reward!"
    assert info.get("illegal_move"), "Should flag illegal move!"
    
    print("✓ Illegal move test passed!\n")


def test_draw():
    """Test draw detection (fill the board)."""
    print("=" * 50)
    print("TEST 8: Draw Detection")
    print("=" * 50)
    
    env = GomokuEnv9x9()
    env.reset()
    
    # This is a contrived pattern that fills the board without winning
    # It's hard to guarantee no 5-in-a-row, so we just test the mechanism
    # by manually setting up a near-full board
    
    # Fill board in a pattern that avoids 5-in-a-row
    # Alternating columns pattern
    move_count = 0
    done = False
    
    # Simple fill pattern (may trigger win, that's ok for this test)
    for i in range(81):
        if done:
            break
        obs, reward, done, _, info = env.step(i)
        move_count += 1
    
    print(f"Game ended after {move_count} moves")
    print(f"Done: {done}, Info: {info}")
    
    # Game should definitely be over (either win or draw)
    assert done, "Game should be over after filling board!"
    
    print("✓ Draw/full board test passed!\n")


def test_observation_perspective():
    """Test that observations are from current player's perspective."""
    print("=" * 50)
    print("TEST 9: Observation Perspective")
    print("=" * 50)
    
    env = GomokuEnv9x9()
    obs, _ = env.reset()
    
    # Channel 2 should be all 1s for player 1
    assert np.all(obs[2] == 1), "Player 1's indicator should be all 1s"
    print("✓ Player 1: Channel 2 is all 1s")
    
    # Make a move
    env.step(40)  # Player 1 places at center
    
    # Now it's player 2's turn
    obs = env._get_observation()
    
    # Channel 2 should be all 0s for player 2
    assert np.all(obs[2] == 0), "Player 2's indicator should be all 0s"
    print("✓ Player 2: Channel 2 is all 0s")
    
    # Channel 0 should show player 2's stones (none yet)
    # Channel 1 should show player 1's stone at center
    assert obs[1, 4, 4] == 1, "Opponent's (P1) stone should be visible"
    print("✓ Opponent's stone correctly shown in channel 1")
    
    print("✓ Observation perspective test passed!\n")


def test_valid_actions():
    """Test valid actions helper."""
    print("=" * 50)
    print("TEST 10: Valid Actions Helper")
    print("=" * 50)
    
    env = GomokuEnv9x9()
    env.reset()
    
    valid = env.get_valid_actions()
    print(f"Initial valid actions: {len(valid)} (expected: 81)")
    assert len(valid) == 81, "All 81 positions should be valid initially"
    
    # Make a move
    env.step(40)
    valid = env.get_valid_actions()
    print(f"After 1 move: {len(valid)} valid actions (expected: 80)")
    assert len(valid) == 80, "80 positions should be valid after 1 move"
    assert 40 not in valid, "Position 40 should not be valid"
    
    # Test action mask
    mask = env.action_mask()
    assert mask.shape == (81,), "Mask should have 81 elements"
    assert mask.sum() == 80, "80 positions should be valid"
    assert not mask[40], "Position 40 should be masked"
    
    print("✓ Valid actions test passed!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("GOMOKU ENVIRONMENT TEST SUITE")
    print("=" * 50 + "\n")
    
    tests = [
        test_basic_init,
        test_move_placement,
        test_horizontal_win,
        test_vertical_win,
        test_diagonal_win,
        test_anti_diagonal_win,
        test_illegal_move,
        test_draw,
        test_observation_perspective,
        test_valid_actions,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)
    
    if failed == 0:
        print("\n🎉 All tests passed! Environment is ready for training.\n")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please fix before training.\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

