import chess.pgn


import json
import os

PGN_FILE = "my_games.pgn"

# Try to load username from config file, else prompt user
CONFIG_FILE = "config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
        USERNAME = config.get("username", None)
else:
    USERNAME = None

if not USERNAME:
    USERNAME = input("Enter your Chess.com username: ")
    # Save to config for future use
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"username": USERNAME}, f)

def inspect_pgn():
    print(f"Reading {PGN_FILE}...")
    pgn = open(PGN_FILE, encoding="utf-8")
    
    game_count = 0
    positions_found = 0
    
    while True:
        try:
            game = chess.pgn.read_game(pgn)
        except Exception:
            break # End of file
            
        if game is None:
            break
            
        game_count += 1
        
        # Check if you are White or Black
        headers = game.headers
        white = headers.get("White", "?")
        black = headers.get("Black", "?")
        
        # We only care about positions where YOU moved
        if USERNAME in white:
            user_color = chess.WHITE
        elif USERNAME in black:
            user_color = chess.BLACK
        else:
            continue # Skip games you didn't play (e.g. Daily puzzles)

        # Parse moves
        board = game.board()
        for move in game.mainline_moves():
            # If it is currently YOUR turn, this is a training sample
            if board.turn == user_color:
                positions_found += 1
                
                # Visual Check: Print the first 2 positions found
                if positions_found <= 2:
                    print(f"\n--- Training Position #{positions_found} ---")
                    print(f"Game: {white} vs {black}")
                    print(f"Your Color: {'White' if user_color == chess.WHITE else 'Black'}")
                    print(board) 
                    print(f"Target Move: {move.uci()}")
            
            board.push(move)
            
    print(f"\nDONE. Scanned {game_count} games.")
    print(f"Found {positions_found} positions where '{USERNAME}' had to move.")
    
    if positions_found > 1000:
        print("✅ STATUS: DATA READY FOR TRAINING")
    else:
        print("⚠️ STATUS: LOW DATA (Check username spelling in script)")

if __name__ == "__main__":
    inspect_pgn()