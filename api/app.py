from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl.alphazero_train import GomokuNet, AlphaZeroMCTS
from rl.mcts_ai import mcts_move

app = Flask(__name__)
CORS(app)

BOARD_SIZE = 9
MCTS_SIMULATIONS = 100
USE_ALPHAZERO = True
FALLBACK_TO_MCTS = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = None
mcts = None
model_loaded = False

def load_model():
    global model, mcts, model_loaded
    try:
        model_path = project_root / "models" / "alphazero_latest.pth"
        if not model_path.exists():
            model_path = project_root / "models" / "alphazero_best.pth"
        
        if not model_path.exists():
            print(f"Warning: No model found in {project_root / 'models'}")
            print("Will use fallback MCTS if enabled")
            return False
        
        model = GomokuNet(board_size=9, num_channels=64, num_res_blocks=3).to(device)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.eval()
        
        mcts = AlphaZeroMCTS(model, device=device, num_simulations=MCTS_SIMULATIONS)
        model_loaded = True
        print(f"✓ AlphaZero model loaded from: {model_path}")
        print(f"  Using {MCTS_SIMULATIONS} MCTS simulations")
        return True
    except Exception as e:
        print(f"✗ Error loading AlphaZero model: {e}")
        print("Will use fallback MCTS if enabled")
        return False

if USE_ALPHAZERO:
    load_ok = load_model()
    if not load_ok and not FALLBACK_TO_MCTS:
        print("AlphaZero model not loaded and fallback disabled, API will return errors for moves")
else:
    print("AlphaZero disabled, using standard MCTS")

def check_winner(board, row, col, player):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == player:
            count += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == player:
            count += 1
            r -= dr
            c -= dc
        if count >= 5:
            return True
    return False

@app.route('/api/move', methods=['POST'])
def get_ai_move():
    try:
        data = request.json
        board = np.array(data['board'], dtype=np.int8)
        player = int(data['player'])
        
        if USE_ALPHAZERO and model_loaded and mcts is not None:
            try:
                action_probs = mcts.search(board.copy(), player, temperature=0.1, add_noise=False)
                action = np.argmax(action_probs)
            except Exception as e:
                print(f"AlphaZero MCTS error: {e}, falling back to standard MCTS")
                if FALLBACK_TO_MCTS:
                    action = mcts_move(board.copy(), player, simulations=500, max_time=3.0)
                else:
                    raise
        else:
            if FALLBACK_TO_MCTS:
                action = mcts_move(board.copy(), player, simulations=500, max_time=3.0)
            else:
                return jsonify({'error': 'No AlphaZero model available'}), 500
        
        row = int(action // 9)
        col = int(action % 9)
        
        if board[row, col] != 0:
            return jsonify({'error': f'Invalid move: position ({row}, {col}) is already occupied'}), 400
        
        board[row, col] = player
        
        winner = None
        game_over = False
        
        if check_winner(board, row, col, player):
            winner = player
            game_over = True
        elif np.all(board != 0):
            winner = 0
            game_over = True
        
        board_list = board.tolist()
        
        return jsonify({
            'row': row,
            'col': col,
            'board': board_list,
            'winner': winner,
            'gameOver': game_over,
            'aiType': 'AlphaZero' if (USE_ALPHAZERO and model_loaded) else 'MCTS'
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/api/check', methods=['POST'])
def check_game_state():
    try:
        data = request.json
        board = np.array(data['board'], dtype=np.int8)
        row = int(data['row'])
        col = int(data['col'])
        player = int(data['player'])
        
        winner = None
        game_over = False
        
        if check_winner(board, row, col, player):
            winner = player
            game_over = True
        elif np.all(board != 0):
            winner = 0
            game_over = True
        
        return jsonify({
            'winner': winner,
            'gameOver': game_over
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    ai_type = "AlphaZero" if (USE_ALPHAZERO and model_loaded) else "MCTS (fallback)"
    return jsonify({
        'status': 'ok',
        'aiType': ai_type,
        'modelLoaded': model_loaded,
        'device': device,
        'simulations': MCTS_SIMULATIONS if model_loaded else 500
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
