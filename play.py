import pygame
import chess
import torch
import numpy as np
import time
from model import ChessNet

# --- CONFIG ---
WIDTH, HEIGHT = 512, 512
SQ_SIZE = WIDTH // 8
MODEL_PATH = "ghost_bot.pth"
WHITE_COLOR = (240, 217, 181)
BLACK_COLOR = (181, 136, 99)
HIGHLIGHT = (186, 202, 68)

# Unicode Pieces
PIECE_UNICODE = {
    'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
    'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚'
}

class GhostBot:
    def __init__(self, model_path):
        print("Loading Ghost Bot...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.move_to_int = checkpoint['move_to_int']
        self.int_to_move = checkpoint['int_to_move']
        
        # Initialize Architecture
        vocab_size = len(self.move_to_int)
        self.model = ChessNet(vocab_size).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval() # Set to evaluation mode
        print("Ghost Bot Loaded!")

    def predict_move(self, board):
        # 1. Convert Board to Tensor
        matrix = self._board_to_tensor(board)
        tensor = torch.tensor(matrix).unsqueeze(0).to(self.device) # Add batch dim
        
        # 2. Run Inference
        with torch.no_grad():
            logits = self.model(tensor)
        
        # 3. Filter for Legal Moves
        # Sort all moves by probability (highest to lowest)
        sorted_indices = torch.argsort(logits, dim=1, descending=True).squeeze()
        
        for idx in sorted_indices:
            move_idx = idx.item()
            if move_idx in self.int_to_move:
                uci_move = self.int_to_move[move_idx]
                move = chess.Move.from_uci(uci_move)
                
                # Verify legality (The bot might hallucinate illegal moves)
                if move in board.legal_moves:
                    return move
        
        # Fallback: Random legal move (if model is confused)
        return list(board.legal_moves)[0]

    def _board_to_tensor(self, board):
        # Must match dataset.py exactly!
        matrix = np.zeros((12, 8, 8), dtype=np.float32)
        piece_map = board.piece_map()
        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            channel = piece.piece_type - 1
            if piece.color == chess.BLACK:
                channel += 6
            matrix[channel, 7-row, col] = 1
        return matrix

def draw_board(screen, board, selected_square=None, last_move=None):
    # Font setup
    try:
        font = pygame.font.SysFont("segoeuisymbol", 50)
    except:
        font = pygame.font.Font(None, 50)
    
    for r in range(8):
        for c in range(8):
            color = WHITE_COLOR if (r + c) % 2 == 0 else BLACK_COLOR
            
            square = chess.square(c, 7-r)
            
            # Highlight selected
            if selected_square == square:
                color = HIGHLIGHT
            # Highlight last move (optional polish)
            elif last_move and square in [last_move.from_square, last_move.to_square]:
                color = (205, 210, 106)

            pygame.draw.rect(screen, color, (c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
            
            piece = board.piece_at(square)
            if piece:
                text = font.render(PIECE_UNICODE[piece.symbol()], True, (0,0,0))
                rect = text.get_rect(center=(c*SQ_SIZE + SQ_SIZE//2, r*SQ_SIZE + SQ_SIZE//2))
                screen.blit(text, rect)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Me vs. My Ghost")
    clock = pygame.time.Clock()
    
    board = chess.Board()
    bot = GhostBot(MODEL_PATH)
    
    selected_square = None
    game_over = False
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            # HUMAN TURN (White)
            if not game_over and board.turn == chess.WHITE and event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col, row = x // SQ_SIZE, y // SQ_SIZE
                clicked = chess.square(col, 7-row)
                
                if selected_square is None:
                    if board.piece_at(clicked) and board.piece_at(clicked).color == chess.WHITE:
                        selected_square = clicked
                else:
                    move = chess.Move(selected_square, clicked)
                    # Auto-promote to Queen
                    if board.piece_at(selected_square).piece_type == chess.PAWN and chess.square_rank(clicked) == 7:
                        move = chess.Move(selected_square, clicked, promotion=chess.QUEEN)
                        
                    if move in board.legal_moves:
                        board.push(move)
                        selected_square = None
                    else:
                        selected_square = clicked if board.piece_at(clicked) and board.piece_at(clicked).color == chess.WHITE else None

        draw_board(screen, board, selected_square, board.peek() if board.move_stack else None)
        pygame.display.flip()
        
        # BOT TURN (Black)
        if not game_over and board.turn == chess.BLACK:
            pygame.event.pump() # Prevent freezing
            time.sleep(0.5)     # Human-like delay
            
            print("Bot is thinking...")
            bot_move = bot.predict_move(board)
            board.push(bot_move)
            print(f"Bot played: {bot_move}")
            
        if board.is_game_over():
            game_over = True
            print("Game Over:", board.result())

        clock.tick(30)

if __name__ == "__main__":
    main()