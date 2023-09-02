from sklearn.svm import SVR
import numpy as np
import chess

class ChessModel:
    def __init__(self):
        self.model = SVR()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict([X])[0]

def board_to_array(board):
    piece_to_int = {
        'p': 1, 'P': -1,
        'n': 2, 'N': -2,
        'b': 3, 'B': -3,
        'r': 4, 'R': -4,
        'q': 5, 'Q': -5,
        'k': 6, 'K': -6,
        '.': 0
    }

    board_str = board.fen().split(' ')[0]
    board_list = []
    for char in board_str:
        if char.isdigit():
            board_list.extend([0] * int(char))
        elif char != '/':
            board_list.append(piece_to_int[char])
    board_array = np.array(board_list)
    return board_array
