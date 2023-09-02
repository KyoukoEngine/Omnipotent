import chess
import math
import random
from collections import defaultdict
from chess_model import board_to_array
import joblib
from functools import lru_cache
#@lru_cache(maxsize=TRANSPOSITION_TABLE_SIZE)


C_PUCT = 1.0 
TRANSPOSITION_TABLE_SIZE = 1000000  # Adjust as needed
neural_network = joblib.load('stockfishKiller2.2.pkl')
transposition_table = defaultdict(float)


class Node:
    def __init__(self, state, move=None, parent=None):
        self.state = state  # state is a chess.Board object
        self.move = move  # the move that led to this node
        self.parent = parent  # parent Node
        self.children = []  # child Nodes
        self.visits = 0  # number of visits
        self.value = 0.0  # value, to be updated during backpropagation

def MCTS(root_state, iterations, neural_network, explore=False):
    root = Node(root_state)
    for _ in range(iterations):
        leaf = traverse(root, neural_network)
        simulation_result = evaluate(leaf.state, neural_network)
        backpropagate(leaf, simulation_result)
    return select_best_move(root, explore=explore)

def traverse(node, neural_network):
    while node.children:
        node = select_ucb(node)
    if node.visits == 0:
        node.value = evaluate(node.state, neural_network)
    expand(node)
    return node

def select_ucb(node):
    best_value = -float("inf")
    best_node = None
    for child in node.children:
        adjusted_exploration = C_PUCT * math.sqrt(math.log(node.visits + 1) / (child.visits + 1))
        ucb_value = (child.value / (child.visits + 1e-7)) + adjusted_exploration
        if ucb_value > best_value:
            best_value = ucb_value
            best_node = child
    return best_node

def move_value_heuristic(board, move):
    value = 0
    
    # High value for capturing an opponent's piece
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece is not None:
            piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 100}
            value += 10 * piece_values.get(captured_piece.symbol().upper(), 0)
    
    # Extra points for putting the opponent in check
    if board.gives_check(move):
        value += 50
    
    # Even higher if it's a checkmate
    board.push(move)
    if board.is_checkmate():
        value += 1000000
    board.pop()
    
    # Additional points for castling (king safety)
    if board.is_castling(move):
        value += 30

    # Slight incentive for central control
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    if move.to_square in center_squares:
        value += 5

    return value




def expand(node):
    legal_moves = list(node.state.legal_moves)
    sorted_moves = sorted(legal_moves, key=lambda move: move_value_heuristic(node.state, move), reverse=True)
    
    for move in sorted_moves:
        new_state = node.state.copy()
        new_state.push(move)
        child_node = Node(new_state, move=move, parent=node)
        node.children.append(child_node)


def evaluate(state, model):
    fen = state.fen()
    
    # Implement size limit for transposition table
    if len(transposition_table) > TRANSPOSITION_TABLE_SIZE:
        # Remove random item (consider better strategies like LRU)
        transposition_table.popitem()
    
    if fen in transposition_table:
        return transposition_table[fen]
    
    board_array = board_to_array(state)
    value = model.predict(board_array)
    
    transposition_table[fen] = value
    return value
    


def backpropagate(node, result):
    while node:
        node.visits += 1
        node.value += result  # Accumulate values over the tree
        node = node.parent

def select_best_move(node, explore=False):
    if explore:
        top_moves = sorted(node.children, key=lambda x: x.visits, reverse=True)[:3]
        return random.choice(top_moves).move
    else:
        most_visits = -1
        best_move = None
        for child in node.children:
            if child.visits > most_visits:
                most_visits = child.visits
                best_move = child.move
        return best_move



if __name__ == "__main__":
    # Example usage
    board = chess.Board()
    neural_network = joblib.load('stockfishKiller2.2.pkl') # Placeholder for a neural network
    best_move = MCTS(board, 10, neural_network)
    print(f"Best move is {best_move}")
