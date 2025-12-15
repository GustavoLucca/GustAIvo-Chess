import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, pgn_file):
        self.moves = []
        self.boards = []
        self.move_to_int = {}
        self.int_to_move = {}
        
        # Load data immediately
        self._load_pgn(pgn_file)
        
    def _load_pgn(self, pgn_file):
        print(f"Parsing {pgn_file}...")
        pgn = open(pgn_file, encoding="utf-8")
        
        count = 0
        while True:
            try:
                game = chess.pgn.read_game(pgn)
            except:
                break
            if game is None: break
            
            # Simple heuristic: We train on ALL moves in the file
            board = game.board()
            for move in game.mainline_moves():
                # Store the board state BEFORE the move
                self.boards.append(self._board_to_tensor(board))
                
                # Store the move taken
                uci = move.uci()
                if uci not in self.move_to_int:
                    idx = len(self.move_to_int)
                    self.move_to_int[uci] = idx
                    self.int_to_move[idx] = uci
                
                self.moves.append(self.move_to_int[uci])
                board.push(move)
            
            count += 1
            if count % 100 == 0: print(f"Parsed {count} games...", end="\r")
            
        print(f"\nFinished. Loaded {len(self.moves)} positions.")
        print(f"Vocabulary Size: {len(self.move_to_int)} unique moves.")

    def _board_to_tensor(self, board):
        # 12x8x8 Tensor: 6 White pieces, 6 Black pieces
        matrix = np.zeros((12, 8, 8), dtype=np.float32)
        piece_map = board.piece_map()
        
        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            # Channel 0-5 (White), 6-11 (Black)
            channel = piece.piece_type - 1
            if piece.color == chess.BLACK:
                channel += 6
            matrix[channel, 7-row, col] = 1
            
        return matrix

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.boards[idx]), 
            torch.tensor(self.moves[idx], dtype=torch.long)
        )