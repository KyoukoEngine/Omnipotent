import chess
import chess.pgn
import chess.engine
import joblib
from mcts import MCTS  # Your MCTS implementation
from chess_model import ChessModel, board_to_array  # Your model and utility function
import numpy as np
import os

# Initialize counters for results
stockfish_wins = 0
my_model_wins = 0
draws = 0

# Create directory if it doesn't exist
if not os.path.exists("PGN/omnipotent1-beta"):
    os.makedirs("PGN/omnipotent1-beta")

# Initialize or load your model
try:
    model = joblib.load('omnipotent.pkl')
    old_data = joblib.load('omnipotent_data.pkl')  # Load previous training data
except FileNotFoundError:
    model = ChessModel()
    old_data = []

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

# Initialize Stockfish
engine = chess.engine.SimpleEngine.popen_uci("stockfish")
engine.configure({"Skill Level": 0, "Threads": 1, "Hash": 8})

# Initialize game variables
num_games = 5
data = []

for game_number in range(1, num_games + 1):
    board = chess.Board()
    game = chess.pgn.Game()

    is_stockfish_turn = bool(np.random.randint(0, 2))  # Randomly decide who goes first

    if is_stockfish_turn:
        game.headers["White"] = "Stockfish"
        game.headers["Black"] = "MyModel"
    else:
        game.headers["White"] = "MyModel"
        game.headers["Black"] = "Stockfish"

    node = game
    game_data = []

    print(f"Starting game {game_number}. Stockfish is white: {is_stockfish_turn}")

    while not board.is_game_over():
        if is_stockfish_turn:
            result = engine.play(board, chess.engine.Limit(time=0.1, depth=1))
            board.push(result.move)
        else:
            best_move = MCTS(board, 10000, model)
            board.push(best_move)
        
        game_data.append((board_to_array(board), evaluate_terminal(board)))
        node = node.add_main_variation(result.move if is_stockfish_turn else best_move)

        is_stockfish_turn = not is_stockfish_turn

    outcome = evaluate_terminal(board)
    data.extend([(state, outcome) for state, _ in game_data])

    # Count results based on outcome and color
    if outcome == 1:
        if game.headers["White"] == "MyModel":
            my_model_wins += 1
        else:
            stockfish_wins += 1
    elif outcome == -1:
        if game.headers["Black"] == "MyModel":
            my_model_wins += 1
        else:
            stockfish_wins += 1
    else:
        draws += 1

    node.comment = f"Result: {outcome}"
    game.headers["Result"] = str(outcome)

    # Save the game to a PGN file
    with open(f"PGN/omnipotent1-beta/game_{game_number}.pgn", "w") as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)

    print(f"Game {game_number} completed. Current Score: Stockfish {stockfish_wins} - MyModel {my_model_wins} - Draws {draws}")

# Shut down Stockfish Engine
engine.quit()

# Train the model on the new data
print("Training model...")
data.extend(old_data)  # Add old data to new data
train_model(model, data)

# Save the trained model
joblib.dump(model, 'omnipotent1.pkl')
joblib.dump(data, 'omnipotent1_data.pkl')  # Save new training data
print("Training complete. Model saved as 'omnipotent1.pkl'")

# Save results to a text file
with open("results.txt", "w") as f:
    f.write(f"Stockfish: Wins {stockfish_wins}, Draws {draws}, Losses {num_games - stockfish_wins - draws}\n")
    f.write(f"MyModel: Wins {my_model_wins}, Draws {draws}, Losses {num_games - my_model_wins - draws}")

print("Results saved to 'results.txt'")
