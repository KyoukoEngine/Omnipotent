import chess
import chess.pgn
import chess.engine
import joblib
from mcts import MCTS  # Your MCTS implementation
from chess_model import ChessModel, board_to_array  # Your model and utility function
import numpy as np
import os

# Load your model
model = joblib.load('stockfishKiller2.2.pkl')

#model = chess.engine.SimpleEngine.popen_uci("stockfish")

# Define chess puzzles for various ratings.
# Format: {'fen': FEN_string, 'best_move': move_in_uci_format}
puzzles = {
    100: {'fen': '8/3R4/8/8/8/8/3k4/3R4 b - - 0 1', 'best_move': 'd1d7'},
    300: {'fen': '8/8/8/8/8/5N2/6p1/7K b - - 0 1', 'best_move': 'g2g1q'},
    500: {'fen': '8/3B4/8/8/8/8/3k4/3R4 b - - 0 1', 'best_move': 'd1d7'},
    700: {'fen': '8/8/8/2K5/8/8/6q1/7R b - - 0 1', 'best_move': 'g2c2'},
    900: {'fen': '8/8/8/8/8/5N2/6p1/5K2 b - - 0 1', 'best_move': 'g2g1q'},
    1100: {'fen': '8/8/8/8/8/8/3q4/3R3K b - - 0 1', 'best_move': 'd2d1'},
    1300: {'fen': '8/8/8/8/8/8/3q4/3R4 b - - 0 1', 'best_move': 'd2d1'},
    1500: {'fen': '8/8/8/8/8/8/8/3R1k1K b - - 0 1', 'best_move': 'f1f2'},
    1700: {'fen': '8/8/8/8/8/5k2/8/5K2 b - - 0 1', 'best_move': 'f3f2'},
    1900: {'fen': '8/8/8/8/8/8/3q4/3R4 b - - 0 1', 'best_move': 'd2d1'},
    2000: {'fen': '8/8/8/8/8/8/6p1/5K2 b - - 0 1', 'best_move': 'g2g1q'},
}

def evaluate_puzzle(board, best_move_uci):
    best_move_engine = MCTS(board, 1000, model)  # Adjust the number of iterations
    return best_move_engine.uci() == best_move_uci

def benchmark(model, puzzles):
    scores = {}
    for rating, puzzle in puzzles.items():
        board = chess.Board(puzzle['fen'])
        move_chosen = MCTS(board, 1000, model)  # Adjust the number of iterations
        scores[rating] = {'correct': move_chosen.uci() == puzzle['best_move'],
                          'chosen_move': move_chosen.uci()}
    return scores

if __name__ == "__main__":
    scores = benchmark(model, puzzles)
    total_solved = sum([score['correct'] for score in scores.values()])
    print(f"Solved {total_solved} out of {len(puzzles)} puzzles.")
    for rating, score in scores.items():
        print(f"Rating {rating}: {'Solved' if score['correct'] else 'Failed'}; Move chosen: {score['chosen_move']}")
