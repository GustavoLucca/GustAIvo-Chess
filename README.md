# Chess ML Agent Project

This project is the foundation for building a machine learning chess agent trained on your own chess.com games. The current version features:

- A chess GUI built with `pygame` and `python-chess`
- Unicode chess piece rendering (no image files needed)
- Play as White against a random-move Black agent
- Automatic detection and display of game over states (checkmate, stalemate, etc.)

## How to Run

1. **Install dependencies** (in your virtual environment):
   ```bash
   pip install pygame python-chess
   ```

2. **Run the game:**
   ```bash
   python main.py
   ```

3. **How to play:**
   - Click to select and move White pieces.
   - Black moves randomly.
   - When the game ends, a message will display the result.

## Next Steps
- Implement data-loading logic for chess.com games
- Data processing and ML pipeline to train an ML model on said chess games
- Implement logic to play against a copy of your own chess play!

---

Feel free to fork, contribute, and experiment!
