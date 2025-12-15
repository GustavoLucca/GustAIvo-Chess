import pygame
import chess
import random
import torch
import numpy as np
from play import GhostBot, MODEL_PATH
import time

# --- CONFIG ---
WIDTH, HEIGHT = 512, 512
SQ_SIZE = WIDTH // 8
WHITE = (240, 217, 181)
import pygame
import chess
import random
import torch
import numpy as np
from play import GhostBot, MODEL_PATH
import time

# --- CONFIG ---
WIDTH, HEIGHT = 512, 512
SQ_SIZE = WIDTH // 8
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT = (186, 202, 68)

# Unicode Pieces
PIECE_UNICODE = {
    'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
    'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚'
}


def draw_board(screen, board, selected_square=None):
    try:
        font = pygame.font.SysFont("DejaVu Sans", 50)
    except:
        font = pygame.font.Font(None, 50)  # fallback
    
    for r in range(8):
        for c in range(8):
            color = WHITE if (r + c) % 2 == 0 else BLACK
            
            # Highlight selected square
            square = chess.square(c, 7-r)
            if selected_square == square:
                color = HIGHLIGHT
                
            rect = pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            pygame.draw.rect(screen, color, rect)
            
            # Draw Piece
            piece = board.piece_at(square)
            if piece:
                text = font.render(PIECE_UNICODE[piece.symbol()], True, (0, 0, 0))
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("My Chess Bot (Ghost Bot)")
    clock = pygame.time.Clock()

    # Choose side
    choice = None
    while choice not in ("w", "b"):
        choice = input("Play as White or Black? (w/b): ").strip().lower()

    human_color = chess.WHITE if choice == "w" else chess.BLACK
    bot_color = chess.BLACK if human_color == chess.WHITE else chess.WHITE

    board = chess.Board()
    selected_square = None

    # Instantiate bot once
    bot = GhostBot(MODEL_PATH)

    running = True
    game_over = False
    result_message = ""
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not game_over:
                # HUMAN MOVE (only when it's the human's turn)
                if board.turn == human_color and event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    col, row = x // SQ_SIZE, y // SQ_SIZE
                    clicked_square = chess.square(col, 7-row)

                    if selected_square is None:
                        # Select piece (only allow selecting human's pieces)
                        if board.piece_at(clicked_square) and board.piece_at(clicked_square).color == human_color:
                            selected_square = clicked_square
                    else:
                        # Try Move
                        move = chess.Move(selected_square, clicked_square)

                        # Handle Promotion (Auto-promote to Queen for simplicity)
                        piece = board.piece_at(selected_square)
                        if piece and piece.piece_type == chess.PAWN:
                            promot_rank = 7 if piece.color == chess.WHITE else 0
                            if chess.square_rank(clicked_square) == promot_rank:
                                move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)

                        if move in board.legal_moves:
                            board.push(move)
                            selected_square = None
                        else:
                            # Deselect or select new piece
                            selected_square = clicked_square if board.piece_at(clicked_square) and board.piece_at(clicked_square).color == human_color else None

        if not game_over:
            # BOT MOVE (when it's bot's turn)
            if board.turn == bot_color and not board.is_game_over():
                draw_board(screen, board, selected_square)
                pygame.display.flip()

                # Bot thinks and plays
                time.sleep(0.5)
                bot_move = bot.predict_move(board)
                board.push(bot_move)

            # Check for game over
            if board.is_game_over():
                game_over = True
                outcome = board.outcome()
                if outcome:
                    if outcome.termination == chess.Termination.CHECKMATE:
                        winner = "White" if outcome.winner else "Black"
                        result_message = f"Checkmate! {winner} wins."
                    elif outcome.termination == chess.Termination.STALEMATE:
                        result_message = "Stalemate! Draw."
                    elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
                        result_message = "Draw by insufficient material."
                    elif outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                        result_message = "Draw by fivefold repetition."
                    elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                        result_message = "Draw by 75-move rule."
                    else:
                        result_message = "Draw."
                else:
                    result_message = "Game over."

        draw_board(screen, board, selected_square)
        if game_over:
            # Display game over message
            font = pygame.font.SysFont("DejaVu Sans", 32)
            text = font.render(result_message, True, (200, 30, 30))
            rect = text.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(text, rect)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()