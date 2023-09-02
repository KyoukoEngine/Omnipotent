import chess
import joblib
from mcts import MCTS  # Your MCTS implementation
from chess_model import ChessModel  # Your model

# Initialize or load your model
try:
    model = joblib.load('stockfishKiller2.1.pkl')
except FileNotFoundError:
    model = ChessModel()

# Function to analyze a board position given its FEN and turn
def analyze_position(fen, turn):
    board = chess.Board(fen)
    if turn == 'black':
        board.turn = chess.BLACK
    elif turn == 'white':
        board.turn = chess.WHITE
    else:
        return "Invalid turn input. Please enter 'white' or 'black'."

    best_move = MCTS(board, 10000, model)  # You can adjust the number of iterations (20s for 10000)
    
    # If a best move is found, return it
    if best_move is not None:
        return best_move.uci()
    else:
        return "No legal moves"

# Input for FEN string
fen_input = input("Enter the FEN string of the board position you'd like to analyze: ")

# Input for turn
turn_input = input("Is it white's or black's turn to play? (Enter 'white' or 'black'): ").strip().lower()

# Analyze the position
best_move = analyze_position(fen_input, turn_input)

# Output the best move
print(f"The best move according to the model is: {best_move}")


# mate in 1 rnb1k1nr/pppppppp/8/8/5bP1/P1P1q2B/RBP1P2P/3K2NR b kq - 0 1

#knight takes queen r1bq1k1r/ppp3pp/3p1n2/3Bn3/2Q1P3/8/PPPP1PPP/RNB1K2R b KQ - 0 1