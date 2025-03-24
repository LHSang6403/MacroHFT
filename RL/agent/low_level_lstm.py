import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return torch.sigmoid(out)

class LSTMAgent:
    def __init__(self, args):
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.seq_length = args.RL_window_length
        self.batch_size = args.batch_size
        
        # Model configuration
        self.model = LSTMModel(
            input_size=args.n_state_1 + args.n_state_2,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=1
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.BCELoss()

        # Path configuration
        self.data_path = Path(ROOT) / "MacroHFT" / "data" / args.dataset / "whole"
        self.model_path = Path("./result") / "low_level" / args.dataset / args.clf / str(int(args.alpha)) / "label_1"
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Data parameters
        self.tech_indicator_list = args.tech_indicator_list.split(',')
        self.tech_indicator_list_trend = args.tech_indicator_list_trend.split(',')
        
        self._load_data_index()

    def _load_data_index(self):
        labels_path = self.data_path / "slope_labels.pkl"
        if labels_path.exists():
            with open(labels_path, 'rb') as f:
                self.data_index = pickle.load(f)
        else:
            self.data_index = self._create_data_index()

    def _create_data_index(self):
        data_files = []
        for ext in ['*.feather', '*.csv', '*.parquet']:
            data_files.extend(self.data_path.rglob(ext))
        return [{'path': str(f), 'type': f.suffix[1:]} for f in data_files]

    def act(self, state, state_trend, info):
        self.model.eval()
        with torch.no_grad():
            state_tensor = self._preprocess_input(state, state_trend)
            output = self.model(state_tensor)
            return 1 if output.item() > 0.5 else 0

    def act_test(self, state, state_trend, info):
        return self.act(state, state_trend, info)

    def train(self):
        train_loader, val_loader = self._prepare_data_loaders()
        
        best_loss = float('inf')
        for epoch in range(args.epoch_number):
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets.float())
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()

            # Validation
            val_loss = self._evaluate(val_loader)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model()

            print(f"Epoch {epoch+1}/{args.epoch_number} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f}")

    def _prepare_data_loaders(self):
        sequences, targets = self._load_and_process_data()
        dataset = TensorDataset(
            torch.FloatTensor(sequences), 
            torch.FloatTensor(targets)
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        return (
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=self.batch_size)
        )

    def _load_and_process_data(self):
        all_sequences = []
        all_targets = []
        
        for data_chunk in self.data_index:
            sequences, targets = self._process_chunk(data_chunk)
            if sequences is not None:
                all_sequences.append(sequences)
                all_targets.append(targets)
        
        return np.concatenate(all_sequences), np.concatenate(all_targets)

    def _process_chunk(self, data_chunk):
        try:
            df = self._load_data_file(data_chunk)
            df = self._clean_data(df)
            sequences, targets = self._create_sequences(df)
            return sequences, targets
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return None, None

    def _load_data_file(self, data_chunk):
        file_path = data_chunk['path']
        if data_chunk['type'] == 'feather':
            return pd.read_feather(file_path)
        elif data_chunk['type'] == 'csv':
            return pd.read_csv(file_path)
        elif data_chunk['type'] == 'parquet':
            return pd.read_parquet(file_path)

    def _clean_data(self, df):
        # Handle missing values
        df = df.fillna(method='ffill').fillna(0)
        
        # Normalize features
        for col in self.tech_indicator_list + self.tech_indicator_list_trend:
            if col in df.columns:
                df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        return df

    def _create_sequences(self, df):
        sequences = []
        targets = []
        
        features = df[self.tech_indicator_list + self.tech_indicator_list_trend].values
        
        for i in range(len(features) - self.seq_length):
            seq = features[i:i+self.seq_length]
            target = df['target'].iloc[i+self.seq_length]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.model_path / "best_model.pth")

    def load_model(self):
        checkpoint = torch.load(self.model_path / "best_model.pth")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
    args = parser.parse_args()

    agent = LSTMAgent(args)
    
    if args.epoch_number > 0:
        print("Starting training...")
        agent.train()
        print("Training completed.")
    else:
        agent.load_model()
        print("Model loaded for inference")
