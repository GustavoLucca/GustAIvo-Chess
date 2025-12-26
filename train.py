import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessNet
import chess

# --- CONFIG ---
PGN_FILE = "my_games.pgn"
BATCH_SIZE = 32
EPOCHS = 30 # Raised number of epochs since wasn't plateauing
LEARNING_RATE = 0.001
VAL_FRACTION = 0.1
SEED = 42


@torch.no_grad()
def _topk_correct(outputs: torch.Tensor, labels: torch.Tensor, ks=(1, 3, 5)):
    """Return number of correct predictions for each k in ks."""
    max_k = max(ks)
    # topk_indices: (batch, max_k)
    topk_indices = torch.topk(outputs, k=max_k, dim=1).indices
    # labels: (batch,) -> (batch, 1) for broadcasting
    labels = labels.view(-1, 1)
    correct_matrix = topk_indices.eq(labels)  # (batch, max_k)

    correct_counts = {}
    for k in ks:
        # correct in top-k if any of first k positions matches
        correct_counts[k] = correct_matrix[:, :k].any(dim=1).sum().item()
    return correct_counts


def _piece_bucket(piece_count: int) -> str:
    """Coarse phase buckets based on the number of pieces on the board.

    - 25-32: opening-ish (most pieces still on)
    - 13-24: middlegame-ish
    - 3-12 : endgame-ish
    """
    if piece_count >= 25:
        return "opening"
    if piece_count >= 13:
        return "middlegame"
    return "endgame"


def _collate_batch(batch):
    """Custom collate function to support variable-length fields (like legal move lists).

    Returns:
      - (boards, labels) OR
      - (boards, labels, piece_counts) OR
      - (boards, labels, piece_counts, fens) OR
      - (boards, labels, piece_counts, fens, legal_idxs)
    where boards/labels are tensors and the rest are Python lists.
    """
    first = batch[0]
    if not isinstance(first, (list, tuple)):
        raise TypeError("Unexpected batch element type")

    n = len(first)
    if n < 2:
        raise ValueError("Batch elements must have at least (board, label)")

    boards = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)

    if n == 2:
        return boards, labels

    piece_counts = [int(b[2]) for b in batch]
    if n == 3:
        return boards, labels, piece_counts

    fens = [b[3] for b in batch]
    if n == 4:
        return boards, labels, piece_counts, fens

    legal_idxs = [b[4] for b in batch]
    return boards, labels, piece_counts, fens, legal_idxs


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, *, ks=(1, 3, 5)):
    model.eval()
    total_loss = 0.0
    total = 0
    correct_at_k = {k: 0 for k in ks}

    # Optional: bucketed stats if the dataset provides piece_counts
    bucket_total = {"opening": 0, "middlegame": 0, "endgame": 0}
    bucket_correct = {name: {k: 0 for k in ks} for name in bucket_total}

    has_piece_counts = hasattr(dataloader.dataset, "piece_counts")
    has_fens = hasattr(dataloader.dataset, "fens")
    has_legal_idxs = hasattr(dataloader.dataset, "legal_move_idxs")
    vocab_size = getattr(model, "fc2", None).out_features if getattr(model, "fc2", None) is not None else None

    for batch_idx, batch in enumerate(dataloader):
        # Supported batch shapes:
        # (boards, labels)
        # (boards, labels, piece_counts)
        # (boards, labels, piece_counts, fen)
        piece_counts = None
        fens = None
        legal_idxs = None

        # New batches can include precomputed legal move indices:
        # (boards, labels, piece_counts, fen, legal_idxs)
        if isinstance(batch, (list, tuple)) and len(batch) == 5:
            boards, labels, piece_counts, fens, legal_idxs = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 4:
            boards, labels, piece_counts, fens = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 3 and has_piece_counts:
            boards, labels, piece_counts = batch
        else:
            boards, labels = batch

        boards, labels = boards.to(device), labels.to(device)
        outputs = model(boards)

        # Illegal-move masking (evaluation): prefer precomputed legal indices (fast),
        # otherwise fall back to FEN -> python-chess legal move generation (slow).
        if vocab_size is not None and (legal_idxs is not None or fens is not None):
            legal_mask = torch.zeros((outputs.size(0), vocab_size), dtype=torch.bool, device=outputs.device)

            if legal_idxs is not None:
                # legal_idxs is a list (len=batch) of lists of ints
                for i, idxs in enumerate(legal_idxs):
                    if torch.is_tensor(idxs):
                        idxs = idxs.tolist()
                    for mi in idxs:
                        if 0 <= int(mi) < vocab_size:
                            legal_mask[i, int(mi)] = True
            else:
                if isinstance(fens, str):
                    fens_list = [fens]
                else:
                    fens_list = list(fens)
                for i, fen in enumerate(fens_list):
                    board = chess.Board(fen)
                    for mv in board.legal_moves:
                        mi = dataloader.dataset.move_to_int.get(mv.uci())
                        if mi is not None:
                            legal_mask[i, mi] = True

            any_legal = legal_mask.any(dim=1)
            if not any_legal.all().item():
                legal_mask[~any_legal, :] = True

            outputs = outputs.masked_fill(~legal_mask, -1e9)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        batch_size = labels.size(0)
        total += batch_size
        batch_correct = _topk_correct(outputs, labels, ks=ks)
        for k in ks:
            correct_at_k[k] += batch_correct[k]

        if has_piece_counts and piece_counts is not None:
            # piece_counts is a tensor/list on CPU
            if torch.is_tensor(piece_counts):
                piece_counts_list = piece_counts.tolist()
            else:
                piece_counts_list = list(piece_counts)

            # Per-example bucketed Acc@k
            max_k = max(ks)
            topk_indices = torch.topk(outputs, k=max_k, dim=1).indices
            labels_view = labels.view(-1, 1)
            correct_matrix = topk_indices.eq(labels_view)

            for i, pc in enumerate(piece_counts_list):
                b = _piece_bucket(int(pc))
                bucket_total[b] += 1
                for k in ks:
                    if correct_matrix[i, :k].any().item():
                        bucket_correct[b][k] += 1

    avg_loss = total_loss / max(1, len(dataloader))
    acc_at_k = {k: 100 * correct_at_k[k] / max(1, total) for k in ks}
    bucket_acc = {
        b: {k: (100 * bucket_correct[b][k] / bucket_total[b] if bucket_total[b] else 0.0) for k in ks}
        for b in bucket_total
    }
    return avg_loss, acc_at_k, bucket_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 1. Prepare Data
    full_dataset = ChessDataset(PGN_FILE)
    train_dataset, val_dataset = full_dataset.make_splits(val_fraction=VAL_FRACTION, seed=SEED)

    train_games = len(train_dataset.game_ids) if getattr(train_dataset, "game_ids", None) is not None else "n/a"
    val_games = len(val_dataset.game_ids) if getattr(val_dataset, "game_ids", None) is not None else "n/a"
    print(f"Split by game -> train games: {train_games} | val games: {val_games}")
    print(f"Train positions: {len(train_dataset)} | Val positions: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate_batch)
    
    # 2. Setup Model
    model = ChessNet(num_unique_moves=len(full_dataset.move_to_int)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss_sum = 0.0
        total = 0
        ks = (1, 3, 5)
        train_correct_at_k = {k: 0 for k in ks}
        
        vocab_size = len(full_dataset.move_to_int)

        for batch in train_loader:
            piece_counts = None
            fens = None
            legal_idxs = None
            if isinstance(batch, (list, tuple)) and len(batch) == 5:
                boards, labels, piece_counts, fens, legal_idxs = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 4:
                boards, labels, piece_counts, fens = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                boards, labels, piece_counts = batch
            else:
                boards, labels = batch
            boards, labels = boards.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(boards)

            # Illegal-move masking (training): prefer precomputed legal indices.
            if vocab_size is not None and (legal_idxs is not None or fens is not None):
                legal_mask = torch.zeros((outputs.size(0), vocab_size), dtype=torch.bool, device=outputs.device)
                if legal_idxs is not None:
                    for i, idxs in enumerate(legal_idxs):
                        if torch.is_tensor(idxs):
                            idxs = idxs.tolist()
                        for mi in idxs:
                            if 0 <= int(mi) < vocab_size:
                                legal_mask[i, int(mi)] = True
                else:
                    if isinstance(fens, str):
                        fens_list = [fens]
                    else:
                        fens_list = list(fens)
                    for i, fen in enumerate(fens_list):
                        board = chess.Board(fen)
                        for mv in board.legal_moves:
                            mi = train_loader.dataset.move_to_int.get(mv.uci())
                            if mi is not None:
                                legal_mask[i, mi] = True

                any_legal = legal_mask.any(dim=1)
                if not any_legal.all().item():
                    legal_mask[~any_legal, :] = True

                outputs = outputs.masked_fill(~legal_mask, -1e9)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            batch_size = labels.size(0)
            total_loss_sum += loss.item() * batch_size
            total += batch_size
            batch_correct = _topk_correct(outputs, labels, ks=ks)
            for k in ks:
                train_correct_at_k[k] += batch_correct[k]
            
        train_loss = total_loss_sum / max(1, total)
        train_acc = {k: 100 * train_correct_at_k[k] / max(1, total) for k in ks}

        val_loss, val_acc, val_bucket_acc = evaluate(model, val_loader, criterion, device, ks=ks)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc@1: {train_acc[1]:.2f}% Acc@3: {train_acc[3]:.2f}% Acc@5: {train_acc[5]:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc@1: {val_acc[1]:.2f}% Acc@3: {val_acc[3]:.2f}% Acc@5: {val_acc[5]:.2f}%"
        )

        # Phase-ish breakdown on validation based on piece count
        print(
            "Val by piece-count buckets | "
            f"opening Acc@1: {val_bucket_acc['opening'][1]:.2f}% Acc@3: {val_bucket_acc['opening'][3]:.2f}% Acc@5: {val_bucket_acc['opening'][5]:.2f}% | "
            f"middlegame Acc@1: {val_bucket_acc['middlegame'][1]:.2f}% Acc@3: {val_bucket_acc['middlegame'][3]:.2f}% Acc@5: {val_bucket_acc['middlegame'][5]:.2f}% | "
            f"endgame Acc@1: {val_bucket_acc['endgame'][1]:.2f}% Acc@3: {val_bucket_acc['endgame'][3]:.2f}% Acc@5: {val_bucket_acc['endgame'][5]:.2f}%"
        )

    # 4. Save the Ghost
    print("Saving model...")
    state = {
        'state_dict': model.state_dict(),
        'move_to_int': full_dataset.move_to_int,
        'int_to_move': full_dataset.int_to_move
    }
    torch.save(state, "ghost_bot.pth")
    print("DONE! Your Ghost Bot is ready.")

if __name__ == "__main__":
    main()