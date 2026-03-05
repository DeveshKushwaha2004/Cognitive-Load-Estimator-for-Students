"""
LSTM Model Training Script for Cognitive Load Estimation.

This script trains a 2-layer LSTM model on time-series behavioral features
to classify cognitive load into 3 categories: Low, Medium, High.

Features (6 inputs):
    - Typing speed (keys per second)
    - Typing speed variance
    - Backspace frequency
    - Mouse movement distance
    - Mouse jitter
    - Tab-switch frequency

Usage:
    python train_model.py

The script generates synthetic training data if no real dataset exists,
trains the model, and saves:
    - saved_model.pth (model weights)
    - scaler.pkl (StandardScaler for feature normalization)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib


class CognitiveLSTM(nn.Module):
    """LSTM model for cognitive load classification."""

    def __init__(self, input_size=6, hidden_size=64, num_layers=2, num_classes=3):
        super(CognitiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def generate_synthetic_data(n_samples=3000, seq_len=12):
    """
    Generate synthetic labeled data for training.

    Each sample is a sequence of `seq_len` timesteps with 6 features.

    Labels:
        0 = Low cognitive load
        1 = Medium cognitive load
        2 = High cognitive load

    Patterns:
        Low:    steady typing, low backspace, smooth mouse, few tab switches
        Medium: moderate variance in all metrics
        High:   erratic typing, high backspace, jittery mouse, many tab switches
    """
    X, y = [], []

    for _ in range(n_samples):
        label = np.random.choice([0, 1, 2], p=[0.33, 0.34, 0.33])

        if label == 0:  # Low load
            typing_speed = np.random.normal(4.0, 0.5, seq_len)
            speed_variance = np.random.normal(0.2, 0.1, seq_len)
            backspace_rate = np.random.normal(0.05, 0.02, seq_len)
            mouse_distance = np.random.normal(200, 50, seq_len)
            mouse_jitter = np.random.normal(2, 1, seq_len)
            tab_switches = np.random.normal(0.5, 0.3, seq_len)
        elif label == 1:  # Medium load
            typing_speed = np.random.normal(2.5, 0.8, seq_len)
            speed_variance = np.random.normal(0.8, 0.3, seq_len)
            backspace_rate = np.random.normal(0.15, 0.05, seq_len)
            mouse_distance = np.random.normal(400, 100, seq_len)
            mouse_jitter = np.random.normal(8, 3, seq_len)
            tab_switches = np.random.normal(2, 0.8, seq_len)
        else:  # High load
            typing_speed = np.random.normal(1.0, 0.6, seq_len)
            speed_variance = np.random.normal(1.5, 0.5, seq_len)
            backspace_rate = np.random.normal(0.35, 0.1, seq_len)
            mouse_distance = np.random.normal(600, 150, seq_len)
            mouse_jitter = np.random.normal(18, 5, seq_len)
            tab_switches = np.random.normal(5, 1.5, seq_len)

        sequence = np.stack(
            [
                typing_speed,
                speed_variance,
                backspace_rate,
                mouse_distance,
                mouse_jitter,
                tab_switches,
            ],
            axis=1,
        )
        sequence = np.clip(sequence, 0, None)  # No negative values
        X.append(sequence)
        y.append(label)

    return np.array(X), np.array(y)


def train():
    """Train the LSTM model and save artifacts."""
    print("=" * 60)
    print("Cognitive Load LSTM Model Training")
    print("=" * 60)

    # Generate data
    print("\n[1/5] Generating synthetic training data...")
    X, y = generate_synthetic_data(n_samples=3000, seq_len=12)
    print(f"  Dataset shape: X={X.shape}, y={y.shape}")
    print(f"  Label distribution: Low={sum(y==0)}, Medium={sum(y==1)}, High={sum(y==2)}")

    # Fit scaler on flattened features
    print("\n[2/5] Fitting StandardScaler...")
    scaler = StandardScaler()
    X_flat = X.reshape(-1, 6)
    scaler.fit(X_flat)
    X_scaled = scaler.transform(X_flat).reshape(X.shape)
    print(f"  Scaler mean: {np.round(scaler.mean_, 3)}")
    print(f"  Scaler scale: {np.round(scaler.scale_, 3)}")

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    # Initialize model
    print("\n[3/5] Initializing LSTM model...")
    model = CognitiveLSTM(input_size=6, hidden_size=64, num_layers=2, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n[4/5] Training model...")
    epochs = 30
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
        val_acc = val_correct / val_total

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch [{epoch+1:2d}/{epochs}] "
                f"Loss: {total_loss/len(train_loader):.4f}  "
                f"Train Acc: {train_acc:.4f}  "
                f"Val Acc: {val_acc:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # Save
    print(f"\n[5/5] Saving model (best val acc: {best_val_acc:.4f})...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "saved_model.pth")
    scaler_path = os.path.join(script_dir, "scaler.pkl")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Model saved to: {model_path}")
    print(f"  Scaler saved to: {scaler_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    train()
