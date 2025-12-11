import React, { useState, useEffect } from 'react';
import './App.css';

const BOARD_SIZE = 9;
// Use relative path to leverage React's proxy, or 127.0.0.1 to avoid AirPlay conflict
const API_URL = process.env.REACT_APP_API_URL || '';

function App() {
  const [board, setBoard] = useState(Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(0)));
  const [gameOver, setGameOver] = useState(false);
  const [winner, setWinner] = useState(null);
  const [waitingForAI, setWaitingForAI] = useState(false);
  const [lastAIMove, setLastAIMove] = useState(null);
  const [stats, setStats] = useState({ human: 0, ai: 0, draws: 0 });

  useEffect(() => {
    console.log('Board state updated:', board);
  }, [board]);

  const checkWinner = (board, row, col, player) => {
    const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
    
    for (const [dr, dc] of directions) {
      let count = 1;
      
      let r = row + dr, c = col + dc;
      while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] === player) {
        count++;
        r += dr;
        c += dc;
      }
      
      r = row - dr;
      c = col - dc;
      while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] === player) {
        count++;
        r -= dr;
        c -= dc;
      }
      
      if (count >= 5) return true;
    }
    return false;
  };

  const checkDraw = (board) => {
    return board.every(row => row.every(cell => cell !== 0));
  };

  const makeMove = async (row, col) => {
    if (gameOver || waitingForAI || board[row][col] !== 0) return;

    const newBoard = board.map(r => [...r]);
    newBoard[row][col] = 1;
    setBoard(newBoard);

    if (checkWinner(newBoard, row, col, 1)) {
      setGameOver(true);
      setWinner(1);
      setStats(prev => ({ ...prev, human: prev.human + 1 }));
      return;
    }

    if (checkDraw(newBoard)) {
      setGameOver(true);
      setWinner(0);
      setStats(prev => ({ ...prev, draws: prev.draws + 1 }));
      return;
    }

    setWaitingForAI(true);

    try {
      const apiEndpoint = API_URL ? `${API_URL}/api/move` : '/api/move';
      const response = await fetch(apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          board: newBoard,
          player: 2
        }),
      });

      const data = await response.json();

      if (data.error) {
        console.error('API Error:', data.error);
        setWaitingForAI(false);
        return;
      }

      console.log('AI Move Response:', data);
      console.log('AI Board before update:', board);
      
      const aiBoard = data.board;
      console.log('AI Board from API:', aiBoard);
      
      setBoard(aiBoard);
      setLastAIMove({ row: data.row, col: data.col });

      console.log('Board after state update should show:', aiBoard);

      if (data.gameOver) {
        setGameOver(true);
        if (data.winner === 2) {
          setWinner(2);
          setStats(prev => ({ ...prev, ai: prev.ai + 1 }));
        } else if (data.winner === 0) {
          setWinner(0);
          setStats(prev => ({ ...prev, draws: prev.draws + 1 }));
        }
      }
    } catch (error) {
      console.error('Error making AI move:', error);
    } finally {
      setWaitingForAI(false);
    }
  };

  const resetGame = () => {
    setBoard(Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(0)));
    setGameOver(false);
    setWinner(null);
    setWaitingForAI(false);
    setLastAIMove(null);
  };

  const getCellSymbol = (row, col) => {
    const cell = board[row][col];
    if (cell === 0) return '·';
    if (cell === 1) return '●';
    if (cell === 2) return '●';
    return '·';
  };

  const getCellClass = (row, col) => {
    const cell = board[row][col];
    let classes = 'cell';
    if (cell === 1) classes += ' black';
    if (cell === 2) {
      if (lastAIMove && lastAIMove.row === row && lastAIMove.col === col) {
        classes += ' white last-move';
      } else {
        classes += ' white';
      }
    }
    return classes;
  };

  return (
    <div className="App">
      <div className="container">
        <h1 className="title">GOMOKU AI</h1>
        <p className="subtitle">Monte Carlo Tree Search • 9×9 Board</p>
        <p className="credits">Weixi Chen · Hongyu Wei · Minhao Li — University of Southern California</p>

        <div className="stats">
          <div className="stat">
            <div className="stat-label">You</div>
            <div className="stat-value">{stats.human}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Draws</div>
            <div className="stat-value">{stats.draws}</div>
          </div>
          <div className="stat">
            <div className="stat-label">AI</div>
            <div className="stat-value">{stats.ai}</div>
          </div>
        </div>

        <div className="divider"></div>

        <div className="status">
          {gameOver ? (
            winner === 1 ? (
              <div className="status-message success">🎉 You win!</div>
            ) : winner === 2 ? (
              <div className="status-message error">🤖 AI wins!</div>
            ) : (
              <div className="status-message info">Draw!</div>
            )
          ) : waitingForAI ? (
            <div className="status-message info">🤔 AI thinking...</div>
          ) : (
            <div className="status-message info">⚫ Your turn — click any cell</div>
          )}
        </div>

        <div className="board-container">
          <div className="board">
            {board.map((row, rowIdx) => (
              <div key={rowIdx} className="board-row">
                {row.map((cell, colIdx) => (
                  <button
                    key={colIdx}
                    className={getCellClass(rowIdx, colIdx)}
                    onClick={() => makeMove(rowIdx, colIdx)}
                    disabled={gameOver || cell !== 0 || waitingForAI}
                  >
                    {getCellSymbol(rowIdx, colIdx)}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </div>

        <div className="controls">
          <button className="new-game-btn" onClick={resetGame}>
            🔄 New Game
          </button>
        </div>

        <div className="info">
          <div className="info-title">📖 How to Play</div>
          <div className="info-content">
            <p className="info-heading">Gomoku — Get 5 in a row to win!</p>
            <ul className="info-list">
              <li><span className="info-bullet info-bullet-black">●</span> You play as <strong>Black</strong> and move first.</li>
              <li><span className="info-bullet info-bullet-white">●</span> AI plays as <strong>White</strong>.</li>
              <li>Click any empty cell to place your stone.</li>
              <li>Connect <strong>5 in a row</strong> horizontally, vertically, or diagonally to win.</li>
            </ul>
            <p className="info-heading">About the AI</p>
            <p className="info-text">
              The AI uses an <strong>AlphaZero-style neural network</strong> combined with
              <strong> Monte Carlo Tree Search (MCTS)</strong> to evaluate board positions and choose moves.
            </p>
          </div>
        </div>

        <div className="footer">
          <p>Powered by MCTS • TAC450 Project</p>
        </div>
      </div>
    </div>
  );
}

export default App;

