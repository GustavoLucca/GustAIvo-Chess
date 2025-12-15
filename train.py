import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessNet

# --- CONFIG ---
PGN_FILE = "my_games.pgn"
BATCH_SIZE = 32
EPOCHS = 30 # Raised number of epochs since wasn't plateauing
LEARNING_RATE = 0.001

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 1. Prepare Data
    dataset = ChessDataset(PGN_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Setup Model
    model = ChessNet(num_unique_moves=len(dataset.move_to_int)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for boards, labels in dataloader:
            boards, labels = boards.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # 4. Save the Ghost
    print("Saving model...")
    state = {
        'state_dict': model.state_dict(),
        'move_to_int': dataset.move_to_int,
        'int_to_move': dataset.int_to_move
    }
    torch.save(state, "ghost_bot.pth")
    print("DONE! Your Ghost Bot is ready.")

if __name__ == "__main__":
    main()