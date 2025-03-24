import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)

class LSTMModel(nn.Module):
    def __init__(self, n_state_1, n_state_2, n_action, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input dimensions
        self.single_encoder = nn.Linear(n_state_1, hidden_size)
        self.trend_encoder = nn.Linear(n_state_2, hidden_size)
        
        # Previous action embedding
        self.action_embedding = nn.Embedding(n_action, hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size*3,  # Combined features
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer to predict close price
        self.linear = nn.Linear(hidden_size, 1)  # Output one value: predicted close price
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, state, state_trend, previous_action):
        batch_size = state.size(0)
        
        # Encode state features
        single_features = self.single_encoder(state)
        trend_features = self.trend_encoder(state_trend)
        
        # Encode previous action
        action_features = self.action_embedding(previous_action).squeeze(1)
        
        # Combine features
        combined = torch.cat([single_features, trend_features, action_features], dim=-1)
        combined = combined.unsqueeze(1)  # Add sequence dimension [batch, 1, features]
        
        # LSTM layer
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(state.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(state.device)
        
        out, _ = self.lstm(combined, (h0, c0))
        out = self.dropout(out[:, -1, :])
        
        # Predict close price
        close_price = self.linear(out)
        
        return close_price.float()

class LSTMAgent:
    def __init__(self, args):
        # Configuration
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.n_state_1 = args.n_state_1
        self.n_state_2 = args.n_state_2
        self.n_action = 2
        self.args = args
        
        # Model
        self.model = LSTMModel(
            n_state_1=self.n_state_1,
            n_state_2=self.n_state_2,
            n_action=self.n_action,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        ).to(self.device)
        
    def act(self, state, state_trend, info):
        self.model.eval()
        with torch.no_grad():
            x1 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            x2 = torch.FloatTensor(state_trend).unsqueeze(0).to(self.device)
            previous_action = torch.tensor(info["previous_action"], dtype=torch.long).unsqueeze(0).to(self.device)
            
            close_price = self.model(x1, x2, previous_action)  # Get predicted close price
            return close_price.item()  # Return single float value
            
    def act_test(self, state, state_trend, info):
        # Same implementation as act but without randomness
        return self.act(state, state_trend, info)
    
    def _prepare_data_loaders(self):
        # Load data from feather file
        data_path = os.path.join(ROOT, "MacroHFT", "data", "ETHUSDT", "whole", "train.feather")
        try:
            df = pd.read_feather(data_path)
        except FileNotFoundError as e:
            print(f"Error reading feather file: {e}")
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None, None
        
        # Define the state columns based on your provided data
        state_cols = ['volume', 'bid1_size_n', 'bid2_size_n', 'bid3_size_n',
                       'bid4_size_n', 'bid5_size_n', 'ask1_size_n', 'ask2_size_n',
                       'ask3_size_n', 'ask4_size_n']  #First 10 state features

        state_trend_cols = ['ask5_size_n', 'wap_1', 'wap_2',
                       'wap_balance', 'buy_spread', 'sell_spread', 'buy_volume',
                       'sell_volume', 'volume_imbalance', 'price_spread'] #next 10 trend features.
        
        # Check if all required columns exist
        required_columns = state_cols + state_trend_cols + ['previous_action', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: Missing columns in feather file: {missing_columns}")
            return None, None
        
        try:
            states = torch.tensor(df[state_cols].values, dtype=torch.float32)
            state_trends = torch.tensor(df[state_trend_cols].values, dtype=torch.float32)
            prev_actions = torch.tensor(df['previous_action'].values, dtype=torch.long).unsqueeze(1)  # Ensure it's a LongTensor
            targets = torch.tensor(df['close_price'].values, dtype=torch.float32).unsqueeze(1)  # Predicted close price
        except KeyError as e:
            print(f"KeyError: Column '{e}' not found in DataFrame.")
            return None, None
        except Exception as e:
            print(f"An error occurred while creating tensors: {e}")
            return None, None

        # Create dataset
        dataset = TensorDataset(states, state_trends, prev_actions, targets)
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size)
        
        return train_loader, val_loader
    
    def train(self):
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Create target network
        self.target_model = LSTMModel(
            n_state_1=self.n_state_1,
            n_state_2=self.n_state_2,
            n_action=self.n_action,
            hidden_size=self.args.hidden_size,
            num_layers=self.args.num_layers
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Setup data loading
        train_loader, val_loader = self._prepare_data_loaders()
        
        if train_loader is None or val_loader is None:
            print("Data loading failed.  Exiting training.")
            return
            
        # Track best model
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(self.args.epoch_number):
            # Set model to training mode
            self.model.train()
            running_loss = 0.0
            
            # Loop over batches
            for i, (states, state_trends, prev_actions, targets) in enumerate(train_loader):
                # Move data to device
                states = states.to(self.device)
                state_trends = state_trends.to(self.device)
                prev_actions = prev_actions.to(self.device)
                targets = targets.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(states, state_trends, prev_actions)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                
                # Print progress
                if (i+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.args.epoch_number}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            # Validation phase
            val_loss = self._validate(val_loader)
            print(f'Epoch [{epoch+1}/{self.args.epoch_number}], Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}')
            
            # Update target network
            self._update_target_network()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model()
                print(f'Model saved with validation loss: {val_loss:.4f}')
        
        print("Training completed.")

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for states, state_trends, prev_actions, targets in val_loader:
                states = states.to(self.device)
                state_trends = state_trends.to(self.device)
                prev_actions = prev_actions.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(states, state_trends, prev_actions)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)

    def _update_target_network(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def _save_model(self):
        model_dir = os.path.join("result", "low_level", self.args.dataset, "best_model", "lstm", str(int(self.args.alpha)))
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, "model.pth"))
    
    def load_model(self):
        try:
            model_path = os.path.join("result", "low_level", self.args.dataset, "best_model", "lstm", str(int(self.args.alpha)), "model.pth")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETHUSDT")
    parser.add_argument("--clf", default="lstm")
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--RL_window_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch_number", type=int, default=15)
    parser.add_argument("--n_state_1", type=int, default=10, help="Number of state features")
    parser.add_argument("--n_state_2", type=int, default=10, help="Number of trend features")
    parser.add_argument('--tech_indicator_list', type=str, default='open,high,low,close,volume', 
                    help='Comma-separated list of technical indicators')
    parser.add_argument("--tech_indicator_list_trend", type=str, default="price_change,price_change_5",
                    help="Comma-separated list of trend indicators")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--label", type=str, default="label_1", help="Label for the model")
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    agent = LSTMAgent(args)
    
    if args.epoch_number > 0:
        print("Starting training...")
        agent.train()
    else:
        agent.load_model()
        print("Model loaded for inference")
        
        # Test the model
        try:
            state = np.random.rand(args.n_state_1)
            state_trend = np.random.rand(args.n_state_2)
            info = {"previous_action": 0}
            
            close_price = agent.act(state, state_trend, info)
            print(f"Predicted close price: {close_price}")
        except Exception as e:
            print(f"Test prediction failed: {e}")
