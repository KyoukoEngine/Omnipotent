import chess
import chess.pgn
import joblib
from mcts import MCTS  # Your MCTS implementation
from chess_model import ChessModel, board_to_array  # Your model and utility function
import numpy as np
import os

# Initialize counters for results
omnipotent_wins = 0
stockfishkiller_wins = 0
draws = 0

# Create directory if it doesn't exist
if not os.path.exists("PGN/SFKiller_vs_omnipotent"):
    os.makedirs("PGN/SFKiller_vs_omnipotent")

# Initialize or load your models
try:
    omnipotent = joblib.load('omnipotent.pkl')
    SFKiller = joblib.load('stockfishKiller1.3.pkl')
except FileNotFoundError:
    print("Models not found!")
    exit()

def evaluate_terminal(board):
    if board.is_checkmate():
        return 1 if board.turn == chess.BLACK else -1
    return 0  # Draw or stalemate

def train_model(model, data):
    X, y = zip(*data)
    X = np.array(X)
    y = np.array(y)
    model.train(X, y)
    print("Training complete.")

# Initialize game variables
num_games = 50
data = []

for game_number in range(1, num_games + 1):
    board = chess.Board()
    game = chess.pgn.Game()

    is_omnipotent_turn = bool(np.random.randint(0, 2))  # Randomly decide who goes first

    if is_omnipotent_turn:
        game.headers["White"] = "Omnipotent"
        game.headers["Black"] = "SFKiller"
    else:
        game.headers["White"] = "SFKiller"
        game.headers["Black"] = "Omnipotent"

    node = game
    game_data = []

    print(f"Starting game {game_number}. Omnipotent is white: {is_omnipotent_turn}")

    while not board.is_game_over():
        if is_omnipotent_turn:
            best_move = MCTS(board, 50, omnipotent)
        else:
            best_move = MCTS(board, 1, SFKiller)
        
        board.push(best_move)
        game_data.append((board_to_array(board), evaluate_terminal(board)))
        node = node.add_main_variation(best_move)
        
        is_omnipotent_turn = not is_omnipotent_turn

    outcome = evaluate_terminal(board)
    data.extend([(state, outcome) for state, _ in game_data])

    # Count results based on outcome and color
    if outcome == 1:
        if game.headers["White"] == "Omnipotent":
            omnipotent_wins += 1
        else:
            stockfishkiller_wins += 1
    elif outcome == -1:
        if game.headers["Black"] == "Omnipotent":
            omnipotent_wins += 1
        else:
            stockfishkiller_wins += 1
    else:
        draws += 1

    node.comment = f"Result: {outcome}"
    game.headers["Result"] = str(outcome)

    # Save the game to a PGN file
    with open(f"PGN/SFKiller_vs_omnipotent/game_{game_number}.pgn", "w") as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)

    print(f"Game {game_number} completed. Current Score: Omnipotent {omnipotent_wins} - SFKiller {stockfishkiller_wins} - Draws {draws}")

# Save results to a text file
with open("results.txt", "w") as f:
    f.write(f"Omnipotent: Wins {omnipotent_wins}, Draws {draws}, Losses {num_games - omnipotent_wins - draws}\n")
    f.write(f"StockfishKiller: Wins {stockfishkiller_wins}, Draws {draws}, Losses {num_games - stockfishkiller_wins - draws}")

print("Results saved to 'results.txt'")
