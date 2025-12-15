import requests
import time


# --- CONFIG ---
import json
import os

CONFIG_FILE = "config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
        USERNAME = config.get("username", None)
else:
    USERNAME = None

if not USERNAME:
    USERNAME = input("Enter your Chess.com username: ")
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"username": USERNAME}, f)

OUTPUT_FILE = "my_games.pgn"

def get_games():
    print(f"Fetching games for user: {USERNAME}")
    headers = {
        "User-Agent": "ChessBot/1.0 (contact: your_email@example.com)"
    }
    
    # 1. Get list of monthly archives
    archives_url = f"https://api.chess.com/pub/player/{USERNAME}/games/archives"
    try:
        response = requests.get(archives_url, headers=headers)
        response.raise_for_status()
        archives = response.json().get("archives", [])
    except Exception as e:
        print(f"Error fetching archives: {e}")
        return

    print(f"Found {len(archives)} monthly archives. Downloading...")

    # 2. Download each month's games
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for url in archives:
            print(f"Downloading: {url}")
            try:
                # Get the PGN data for that month
                pgn_url = f"{url}/pgn"
                r = requests.get(pgn_url, headers=headers)
                
                # Write to file
                f_out.write(r.text)
                f_out.write("\n\n") # Ensure separation between months
                
                time.sleep(1) # Be nice to the API
            except Exception as e:
                print(f"Failed to download {url}: {e}")

    print(f"\nSUCCESS! All games saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    get_games()