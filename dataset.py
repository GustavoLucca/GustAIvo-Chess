import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

class ChessDataset(Dataset):
    """Dataset of (board_tensor, move_idx) samples built from PGN games.

    Notes:
    - We keep samples grouped per-game so we can split train/val by *game*,
      which avoids leakage from the same game appearing in both splits.
    - `move_to_int` / `int_to_move` are built from the loaded PGN.
    """

    def __init__(
        self,
        pgn_file: str,
        *,
        indices: Optional[List[Tuple[int, int]]] = None,
        move_to_int: Optional[dict] = None,
        int_to_move: Optional[dict] = None,
    ):
        # Flat storage for the selected subset (via indices)
        self.moves: List[int] = []
        self.boards: List[np.ndarray] = []
        self.plys: List[int] = []
        self.piece_counts: List[int] = []
        self.fens: List[str] = []
        # Per-position: list of move indices (into vocab) that are legal from that position.
        self.legal_move_idxs: List[List[int]] = []

        # Vocab (shared across splits)
        self.move_to_int = move_to_int if move_to_int is not None else {}
        self.int_to_move = int_to_move if int_to_move is not None else {}

        # Full, per-game storage (only populated when we load from PGN)
        self._games_boards: List[List[np.ndarray]] = []
        self._games_moves: List[List[int]] = []
        self._games_plys: List[List[int]] = []
        self._games_piece_counts: List[List[int]] = []
        self._games_fens: List[List[str]] = []
        self._games_legal_move_idxs: List[List[List[int]]] = []

        # Optional: populated for split datasets
        self.game_ids: Optional[List[int]] = None

        # Load data immediately
        self._load_pgn(pgn_file)

        # If a subset is requested (train/val split), materialize it now.
        if indices is None:
            indices = [(g, i) for g, gm in enumerate(self._games_moves) for i in range(len(gm))]
        self._materialize(indices)

    @classmethod
    def from_parsed(
        cls,
        *,
        games_boards: List[List[np.ndarray]],
        games_moves: List[List[int]],
        games_plys: List[List[int]],
        games_piece_counts: List[List[int]],
        games_fens: List[List[str]],
        games_legal_move_idxs: List[List[List[int]]],
        indices: List[Tuple[int, int]],
        move_to_int: dict,
        int_to_move: dict,
    ):
        """Create a dataset reusing already-parsed PGN structures."""
        self = cls.__new__(cls)
        Dataset.__init__(self)

        self.moves = []
        self.boards = []
        self.plys = []
        self.piece_counts = []
        self.fens = []
        self.legal_move_idxs = []

        self.move_to_int = move_to_int
        self.int_to_move = int_to_move

        self._games_boards = games_boards
        self._games_moves = games_moves
        self._games_plys = games_plys
        self._games_piece_counts = games_piece_counts
        self._games_fens = games_fens
        self._games_legal_move_idxs = games_legal_move_idxs

        self.game_ids = sorted({g for g, _ in indices})

        self._materialize(indices)
        return self
        
    def _load_pgn(self, pgn_file: str):
        print(f"Parsing {pgn_file}...")
        with open(pgn_file, encoding="utf-8") as pgn:
            count = 0
            while True:
                try:
                    game = chess.pgn.read_game(pgn)
                except Exception:
                    break
                if game is None:
                    break

                game_boards: List[np.ndarray] = []
                game_moves: List[int] = []
                game_plys: List[int] = []
                game_piece_counts: List[int] = []
                game_fens: List[str] = []
                game_legal_move_idxs: List[List[int]] = []

                board = game.board()
                ply = 0
                for move in game.mainline_moves():
                    # Store the board state BEFORE the move
                    game_boards.append(self._board_to_tensor(board))
                    game_plys.append(ply)
                    game_piece_counts.append(len(board.piece_map()))
                    game_fens.append(board.fen())

                    # Store the move taken
                    uci = move.uci()
                    if uci not in self.move_to_int:
                        idx = len(self.move_to_int)
                        self.move_to_int[uci] = idx
                        self.int_to_move[idx] = uci

                    game_moves.append(self.move_to_int[uci])

                    # Precompute legal move indices under the (updated) vocab.
                    # Note: moves not in vocab are ignored; mask is best-effort.
                    legal_idxs: List[int] = []
                    for lm in board.legal_moves:
                        mi = self.move_to_int.get(lm.uci())
                        if mi is not None:
                            legal_idxs.append(mi)
                    game_legal_move_idxs.append(legal_idxs)

                    board.push(move)
                    ply += 1

                # Keep per-game samples
                self._games_boards.append(game_boards)
                self._games_moves.append(game_moves)
                self._games_plys.append(game_plys)
                self._games_piece_counts.append(game_piece_counts)
                self._games_fens.append(game_fens)
                self._games_legal_move_idxs.append(game_legal_move_idxs)

                count += 1
                if count % 100 == 0:
                    print(f"Parsed {count} games...", end="\r")

        total_positions = sum(len(g) for g in self._games_moves)
        print(f"\nFinished. Loaded {total_positions} positions.")
        print(f"Vocabulary Size: {len(self.move_to_int)} unique moves.")

    def _materialize(self, indices: List[Tuple[int, int]]):
        self.boards = []
        self.moves = []
        self.plys = []
        self.piece_counts = []
        self.fens = []
        self.legal_move_idxs = []
        for game_idx, pos_idx in indices:
            self.boards.append(self._games_boards[game_idx][pos_idx])
            self.moves.append(self._games_moves[game_idx][pos_idx])
            self.plys.append(self._games_plys[game_idx][pos_idx])
            self.piece_counts.append(self._games_piece_counts[game_idx][pos_idx])
            self.fens.append(self._games_fens[game_idx][pos_idx])
            self.legal_move_idxs.append(self._games_legal_move_idxs[game_idx][pos_idx])

    @property
    def num_games(self) -> int:
        return len(self._games_moves)

    @property
    def num_positions(self) -> int:
        return len(self.moves)

    def make_splits(self, *, val_fraction: float = 0.1, seed: int = 42):
        """Return (train_dataset, val_dataset) split by game.

        This avoids position-level leakage from the same game appearing in both sets.
        """
        if not (0.0 < val_fraction < 1.0):
            raise ValueError("val_fraction must be between 0 and 1")

        rng = np.random.default_rng(seed)
        game_indices = np.arange(self.num_games)
        rng.shuffle(game_indices)

        n_val = max(1, int(round(self.num_games * val_fraction)))
        val_games = set(game_indices[:n_val].tolist())

        train_games = sorted([g for g in range(self.num_games) if g not in val_games])
        val_games_sorted = sorted(list(val_games))

        train_idx: List[Tuple[int, int]] = []
        val_idx: List[Tuple[int, int]] = []
        for g in range(self.num_games):
            target = val_idx if g in val_games else train_idx
            for i in range(len(self._games_moves[g])):
                target.append((g, i))

        train_ds = ChessDataset.from_parsed(
            games_boards=self._games_boards,
            games_moves=self._games_moves,
            games_plys=self._games_plys,
            games_piece_counts=self._games_piece_counts,
            games_fens=self._games_fens,
            games_legal_move_idxs=self._games_legal_move_idxs,
            indices=train_idx,
            move_to_int=self.move_to_int,
            int_to_move=self.int_to_move,
        )
        val_ds = ChessDataset.from_parsed(
            games_boards=self._games_boards,
            games_moves=self._games_moves,
            games_plys=self._games_plys,
            games_piece_counts=self._games_piece_counts,
            games_fens=self._games_fens,
            games_legal_move_idxs=self._games_legal_move_idxs,
            indices=val_idx,
            move_to_int=self.move_to_int,
            int_to_move=self.int_to_move,
        )

        # Persist split game ids for reporting
        train_ds.game_ids = train_games
        val_ds.game_ids = val_games_sorted

        return train_ds, val_ds

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
        board_t = torch.tensor(self.boards[idx])
        move_t = torch.tensor(self.moves[idx], dtype=torch.long)

        # If we have piece counts + FENs + legal move indices, return everything.
        if (
            hasattr(self, "piece_counts")
            and self.piece_counts is not None
            and hasattr(self, "fens")
            and self.fens is not None
            and hasattr(self, "legal_move_idxs")
            and self.legal_move_idxs is not None
        ):
            pc = int(self.piece_counts[idx])
            fen = self.fens[idx]
            legal = self.legal_move_idxs[idx]
            return board_t, move_t, pc, fen, legal

        # If only piece counts exist, return it.
        if hasattr(self, "piece_counts") and self.piece_counts is not None:
            pc = int(self.piece_counts[idx])
            return board_t, move_t, pc

        return board_t, move_t