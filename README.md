# Chess ML Agent Project

This project is the foundation for building a machine learning chess agent trained on your own chess.com games. The repository contains a small GUI and training/playing utilities so you can train a model on PGN data and play against the resulting agent.

What changed recently
- Add a trained Ghost Bot model loader and agent integration
- Ability to choose to play as White or Black (`main.py`)
- `play.py` prompts to choose opponent: Ghost Bot or Random
- `requirements.txt` added with project dependencies

Features
- A chess GUI built with `pygame` and `python-chess`
- Unicode chess piece rendering (no image files needed)
- Play against your trained Ghost Bot or a random-move agent
- Automatic detection and display of game over states (checkmate, stalemate, etc.)

Requirements
- Python 3.10+ (tested with 3.12)
- A virtual environment is recommended
- Dependencies are listed in `requirements.txt` (install with `pip install -r requirements.txt`)

Quick start
1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the main GUI (play against Ghost Bot). The script will prompt whether you want to play as White or Black:

```bash
python main.py
```

4. Alternatively run `play.py` which prompts to choose an opponent (Ghost Bot or Random):

```bash
python play.py
```

Notes about the Ghost Bot
- The Ghost Bot uses the checkpoint file `ghost_bot.pth` (created by `train.py`).
- `main.py` and `play.py` expect `ghost_bot.pth` to exist when you select the Ghost Bot option. If the file is missing, loading will fail — either provide the checkpoint or choose the Random agent.

Workflow: train a Ghost that plays like a user
1. Provide a Chess.com username. The recommended way is to run the fetch script which saves the username to `config.json` and writes a combined PGN to `my_games.pgn`:

```bash
python fetch_data.py
```

2. Inspect the downloaded PGN and confirm the script found games where that username played:

```bash
python loader.py
```

The loader will print how many positions it found where the given username had the move. As a rough rule of thumb, more data is better — `loader.py` prints a small readiness hint (it flags low-data situations).

3. Build the dataset and instantiate the model (quick smoke-check)

Before running a full training run you can (and should) verify the dataset builds correctly and that the model instantiates with the right output size. The repository doesn't commit large data or checkpoints, so these checks run locally against `my_games.pgn`.

```bash
# Quick dataset smoke-test: prints number of positions and vocabulary size
python -c "from dataset import ChessDataset; ds=ChessDataset('my_games.pgn'); print('positions=', len(ds), 'vocab=', len(ds.move_to_int))"

# Quick model smoke-test: instantiate ChessNet with dataset vocabulary
python -c "from dataset import ChessDataset; from model import ChessNet; ds=ChessDataset('my_games.pgn'); m=ChessNet(num_unique_moves=len(ds.move_to_int)); print(m)"
```

4. Train the model on `my_games.pgn`:

```bash
python train.py
```

Training saves the checkpoint to `ghost_bot.pth` in the working directory. Because `ghost_bot.pth` is listed in `.gitignore`, it will not be committed to the repo — it stays local to your machine.

4. Play against your trained Ghost:

```bash
python main.py   # prompts whether you want to play as White or Black
python play.py   # prompts whether to play against Ghost Bot or Random
```

Tips & notes
- If you already have a PGN file you prefer to use, place it as `my_games.pgn` in the repo root before running `train.py`.
- `fetch_data.py` uses the Chess.com public API and will attempt to download all monthly archives for a username. Be mindful of rate limits; the script includes a small delay between requests.
- `loader.py` attempts to detect which games include your username and only uses positions where you had the move (so the model learns *your* moves). If the username is spelled differently in your PGNs, the script may skip games.
- The project currently saves a simple checkpoint with `state_dict`, `move_to_int`, and `int_to_move` which `play.py`/`main.py` expect. If you change the save format, update the loaders accordingly.

Next steps / ideas
- Add an on-screen menu instead of a console prompt
- Add a validation split and monitoring during training to detect overfitting
- Create an CLI interface to run the training from one go, only needing to input username once

Contributing
Feel free to fork, open issues, or submit PRs.

License
MIT
