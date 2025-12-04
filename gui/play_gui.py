"""
Streamlit GUI for playing Gomoku against MCTS AI

Run with: streamlit run gui/play_gui.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import numpy as np

from rl.mcts_ai import mcts_move

st.set_page_config(page_title="Gomoku AI", page_icon="⚫", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
    
    .stApp {
        background: linear-gradient(160deg, #0d0d0d 0%, #1a1a2e 40%, #16213e 100%);
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        width: 48px !important;
        height: 48px !important;
        border-radius: 6px !important;
        font-size: 24px !important;
        border: none !important;
        padding: 0 !important;
        margin: 1px !important;
        background: linear-gradient(145deg, #c4a35a, #b8956a) !important;
    }
    
    .stButton > button:hover:not(:disabled) {
        background: linear-gradient(145deg, #d4b36a, #c8a57a) !important;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

BOARD_SIZE = 9
MCTS_SIMULATIONS = 300  # Fixed difficulty


def check_win(board, row, col, player):
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


def init_game():
    st.session_state.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.message = "Your turn — click any cell"
    st.session_state.waiting_for_ai = False
    st.session_state.last_ai_move = None
    
    if "human_wins" not in st.session_state:
        st.session_state.human_wins = 0
    if "ai_wins" not in st.session_state:
        st.session_state.ai_wins = 0
    if "draws" not in st.session_state:
        st.session_state.draws = 0


def make_human_move(row, col):
    board = st.session_state.board
    board[row, col] = 1
    
    if check_win(board, row, col, 1):
        st.session_state.game_over = True
        st.session_state.winner = "human"
        st.session_state.message = "🎉 You win!"
        st.session_state.human_wins += 1
        return
    
    if np.all(board != 0):
        st.session_state.game_over = True
        st.session_state.winner = "draw"
        st.session_state.message = "Draw!"
        st.session_state.draws += 1
        return
    
    st.session_state.waiting_for_ai = True
    st.session_state.message = "🤔 AI thinking..."


def make_ai_move():
    board = st.session_state.board
    
    action = mcts_move(board, player=2, simulations=MCTS_SIMULATIONS, max_time=3.0)
    row, col = action // 9, action % 9
    
    board[row, col] = 2
    st.session_state.last_ai_move = (row, col)
    
    if check_win(board, row, col, 2):
        st.session_state.game_over = True
        st.session_state.winner = "ai"
        st.session_state.message = "🤖 AI wins!"
        st.session_state.ai_wins += 1
        return
    
    if np.all(board != 0):
        st.session_state.game_over = True
        st.session_state.winner = "draw"
        st.session_state.message = "Draw!"
        st.session_state.draws += 1
        return
    
    st.session_state.waiting_for_ai = False
    st.session_state.message = "Your turn — click any cell"


def main():
    st.markdown('<h1 class="main-title">GOMOKU AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Monte Carlo Tree Search • 9×9 Board</p>', unsafe_allow_html=True)
    
    if "board" not in st.session_state:
        init_game()
    
    # Process AI move
    if st.session_state.get("waiting_for_ai") and not st.session_state.game_over:
        make_ai_move()
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("You", st.session_state.human_wins)
    with col2:
        st.metric("Draws", st.session_state.draws)
    with col3:
        st.metric("AI", st.session_state.ai_wins)
    
    st.divider()
    
    # Status
    if st.session_state.game_over:
        if st.session_state.winner == "human":
            st.success(st.session_state.message)
        elif st.session_state.winner == "ai":
            st.error(st.session_state.message)
        else:
            st.info(st.session_state.message)
    else:
        st.info(f"⚫ {st.session_state.message}")
    
    # Board
    board = st.session_state.board
    last_move = st.session_state.get("last_ai_move")
    
    def get_symbol(r, c):
        val = board[r, c]
        if val == 0:
            return "·"
        elif val == 1:
            return "⚫"
        else:
            if last_move and last_move == (r, c):
                return "⭕"
            return "⚪"
    
    board_cols = st.columns([1, 6, 1])
    
    with board_cols[1]:
        for row in range(BOARD_SIZE):
            cols = st.columns(BOARD_SIZE)
            for col in range(BOARD_SIZE):
                cell_value = board[row, col]
                with cols[col]:
                    is_disabled = st.session_state.game_over or cell_value != 0
                    if st.button(
                        get_symbol(row, col),
                        key=f"cell_{row}_{col}",
                        disabled=is_disabled,
                        use_container_width=True
                    ):
                        if cell_value == 0 and not st.session_state.game_over:
                            make_human_move(row, col)
                            st.rerun()
    
    # New Game button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🔄 New Game", use_container_width=True, type="primary"):
            init_game()
            st.rerun()
    
    # How to play
    with st.expander("📖 How to Play"):
        st.markdown("""
        **Gomoku** — Get 5 in a row to win!
        
        - You: ⚫ (Black) — Move first
        - AI: ⚪ (White)
        - Click empty cell to place stone
        - 5 in a row (horizontal/vertical/diagonal) wins!
        
        **About the AI:**
        Uses Monte Carlo Tree Search (MCTS) — the same algorithm used by AlphaGo!
        """)
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Powered by MCTS • TAC450 Project</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
