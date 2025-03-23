import pathlib
import sys
import random
import argparse
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json

warnings.filterwarnings("ignore")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from MacroHFT.model.net import *
from MacroHFT.env.low_level_env import Testing_Env, Training_Env
from MacroHFT.RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from MacroHFT.RL.util.replay_buffer import ReplayBuffer

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

# Path definitions
DATA_PATH = os.path.join(ROOT, "data")
LOG_PATH = os.path.join(ROOT, "logs")

# Ensure TensorFlow logs are not too verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",type=int,default=1000000,)
parser.add_argument("--dataset",type=str,default="ETHUSDT")
parser.add_argument("--q_value_memorize_freq",type=int, default=10,)
parser.add_argument("--batch_size",type=int,default=512)
parser.add_argument("--eval_update_freq",type=int,default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epsilon_start",type=float,default=0.5)
parser.add_argument("--epsilon_end",type=float,default=0.1)
parser.add_argument("--decay_length",type=int,default=5)
parser.add_argument("--update_times",type=int,default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost",type=float,default=2.0 / 10000)
parser.add_argument("--back_time_length",type=int,default=1)
parser.add_argument("--seed",type=int,default=12345)
parser.add_argument("--n_step",type=int,default=1)
parser.add_argument("--epoch_number",type=int,default=15)
parser.add_argument("--clf",type=str,default="lstm")
parser.add_argument("--alpha",type=float,default="0")
parser.add_argument("--device",type=str,default="cuda:0")
parser.add_argument("--hidden_size",type=int,default=64)
parser.add_argument("--num_layers",type=int,default=2)
parser.add_argument("--sequence_length",type=int,default=10)
parser.add_argument("--data_path", type=str, default="data", help="Path to data directory")
parser.add_argument("--model_name", type=str, default="lstm", help="Model name")
parser.add_argument("--n_state_1", type=int, default=10, help="Number of state features")
parser.add_argument("--n_state_2", type=int, default=10, help="Number of trend features")
parser.add_argument("--RL_window_length", type=int, default=10, help="Sequence length for LSTM")
parser.add_argument("--is_train", type=str, default="true", help="Whether to train the model (true/false)")
parser.add_argument("--tech_indicator_list", type=str, default="open,high,low,close,volume", 
                    help="Comma-separated list of technical indicators")
parser.add_argument("--tech_indicator_list_trend", type=str, default="price_change,price_change_5", 
                    help="Comma-separated list of trend indicators")


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LSTMSubAgent:
    """
    LSTM-based agent that predicts binary actions (0 or 1)
    """
    def __init__(self, model_name="lstm", epoch_number=15, data_path=None):
        """
        Initialize the LSTM agent
        
        Args:
            model_name: Name of the model for saving/loading
            epoch_number: Number of epochs to train
            data_path: Path to data directory (if None, uses default)
        """
        self.model_name = model_name
        self.epoch_number = epoch_number
        
        # Set data path based on input or default
        if data_path is not None:
            if os.path.isabs(data_path):
                self.data_path = data_path
            else:
                self.data_path = os.path.join(ROOT, data_path)
        else:
            self.data_path = DATA_PATH
        
        self.model = None
        self.data_index = None
        self.sequence_length = 10
        self.tech_indicator_list = []
        self.tech_indicator_list_trend = []
        self.model_path = os.path.join(LOG_PATH, self.model_name)
        
        # Ensure directories exist
        os.makedirs(self.model_path, exist_ok=True)
        
        self.is_trained = False
        
    def act(self, state, state_trend, info):
        """Predict action values using the LSTM model"""
        # Prepare input for model
        if len(state.shape) == 1:  # Single state
            state = np.expand_dims(state, axis=0)  # Add batch dimension
        if len(state_trend.shape) == 1:  # Single trend state
            state_trend = np.expand_dims(state_trend, axis=0)  # Add batch dimension
            
        # Combine features
        combined_features = np.concatenate([state, state_trend], axis=1)
        
        # Add sequence dimension if not present
        if len(combined_features.shape) == 2:
            # For single sample, reshape to [1, sequence_length, features]
            combined_features = np.expand_dims(combined_features, axis=0)
        
        # Get prediction from model
        if self.model is not None:
            # Model returns probability of action 1
            action_prob = self.model.predict(combined_features, verbose=0)[0][0]
            # Convert to binary action
            action = 1 if action_prob > 0.5 else 0
            return action
        else:
            # If no model, return random action
            return np.random.randint(0, 2)

    def act_test(self, state, state_trend, info):
        """Same as act but explicitly for testing"""
        # This is identical to act since we're not using exploration
        return self.act(state, state_trend, info)
            
    def train(self, dataset, end_date, start_date, trade_time_list, tech_indicator_list, tech_indicator_list_trend, feature_number, return_label = 'trend_1', vix_file=None, reverse=False, RL_window_length=10, n_state_1=10, n_state_2=10, learn_start=32, is_train_1=True, label_1=None, verbose=True):
        """
        Train the LSTM model using the available data
        """
        # Store parameters
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.n_state_1 = n_state_1
        self.n_state_2 = n_state_2
        self.sequence_length = RL_window_length
        self.is_train_1 = is_train_1
        self.verbose = verbose
        
        # Set data path based on dataset
        self.data_path = os.path.join(ROOT, "data", dataset) if dataset is not None else os.path.join(ROOT, "data")
        
        # Check if data directory exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        print(f"Using data path: {self.data_path}")
        
        # Create directory for model saving
        model_save_path = os.path.join(LOG_PATH, self.model_name)
        best_model_path = os.path.join(model_save_path, "best_model")
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(best_model_path, exist_ok=True)
        
        # Build data index
        self.data_index = []
        
        # First look for .feather files in data path
        feather_files = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.feather'):
                    feather_files.append(os.path.join(root, file))
        
        if not feather_files:
            # If no .feather files are found, try other supported formats
            print(f"No .feather files found in {self.data_path}. Looking for other formats.")
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith('.csv') or file.endswith('.parquet'):
                        self.data_index.append({
                            'path': os.path.join(root, file),
                            'type': 'csv' if file.endswith('.csv') else 'parquet'
                        })
        else:
            # Process the found feather files
            print(f"Found {len(feather_files)} .feather files")
            
            # Look for specific structure (train, test, val folders)
            train_files = [f for f in feather_files if '/train/' in f or 'train' in f.lower()]
            
            if train_files:
                print(f"Found {len(train_files)} training files")
                for file_path in train_files:
                    self.data_index.append({
                        'path': file_path,
                        'type': 'feather'
                    })
            else:
                # If no specific train files, use all feather files
                for file_path in feather_files:
                    self.data_index.append({
                        'path': file_path,
                        'type': 'feather'
                    })
        
        # If we still don't have any data files, create sample data
        if not self.data_index:
            print("No data files found. Creating sample data for training.")
            sample_data_path = os.path.join(self.data_path, "sample_data.csv")
            self._create_sample_data(sample_data_path)
            self.data_index.append({
                'path': sample_data_path,
                'type': 'csv'
            })
        
        # Print data index for debugging
        print(f"Data index contains {len(self.data_index)} files:")
        for i, item in enumerate(self.data_index[:5]):  # Show first 5 items
            print(f"  {i+1}. {item['path']}")
        if len(self.data_index) > 5:
            print(f"  ... and {len(self.data_index)-5} more files")
        
        # Train the model
        if self.verbose:
            print("Starting LSTM model training...")
        
        # Initialize model if not already done
        if self.model is None:
            self._build_model(n_state_1, n_state_2)
        
        # Training parameters
        epochs = self.epoch_number
        batch_size = 64
        best_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            # Process each data chunk
            for chunk_idx, data_chunk in enumerate(self.data_index):
                # Load and preprocess data
                data = self._load_data_chunk(data_chunk)
                
                if data is None:
                    continue  # Skip this chunk if loading failed
                
                X, y = data
                
                # Create mini-batches
                num_samples = len(X)
                num_batches = num_samples // batch_size
                
                if num_batches == 0:
                    # Process the entire chunk as one batch if smaller than batch_size
                    loss = self.model.train_on_batch(X, y)
                    epoch_loss += loss
                    batch_count += 1
                else:
                    # Process mini-batches
                    for i in range(num_batches):
                        start_idx = i * batch_size
                        end_idx = (i + 1) * batch_size
                        
                        X_batch = X[start_idx:end_idx]
                        y_batch = y[start_idx:end_idx]
                        
                        loss = self.model.train_on_batch(X_batch, y_batch)
                        epoch_loss += loss
                        batch_count += 1
            
            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            # Save the best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.model.save(os.path.join(best_model_path, f"lstm_model.h5"))
                if self.verbose:
                    print(f"Saved best model with loss: {best_loss:.4f}")
        
        # Save the final model
        self.model.save(os.path.join(model_save_path, f"lstm_model_final.h5"))
        
        if self.verbose:
            print(f"Training completed. Final loss: {avg_epoch_loss:.4f}")
        
        return True
        
    def _create_sample_data(self, file_path):
        """Create sample data for training when no real data is available"""
        # Create a directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Generate sample data
        num_samples = 5000  # Number of data points
        num_features = max(self.n_state_1, 10)  # Number of features
        num_trends = max(self.n_state_2, 5)  # Number of trend features
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(num_features)]
        trend_names = [f'trend_{i}' for i in range(num_trends)]
        
        # Update the feature lists with the generated names
        self.tech_indicator_list = feature_names
        self.tech_indicator_list_trend = trend_names
        
        # Generate random data
        data = np.random.randn(num_samples, num_features + num_trends)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=feature_names + trend_names)
        
        # Add a binary target column (0 or 1)
        df['target'] = np.random.randint(0, 2, size=num_samples)
        
        # Add timestamp column
        base_timestamp = pd.Timestamp('2022-01-01')
        timestamps = [base_timestamp + pd.Timedelta(minutes=i) for i in range(num_samples)]
        df['timestamp'] = timestamps
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        
        print(f"Created sample data file with {num_samples} rows at {file_path}")
        return file_path

    def _load_data_chunk(self, data_chunk):
        """Load and preprocess a chunk of data
        
        Args:
            data_chunk: A reference to the data chunk to load
            
        Returns:
            tuple: (features, targets) or None if loading failed
        """
        try:
            # Determine file path
            if 'path' in data_chunk:
                file_path = data_chunk['path']
            else:
                file_path = os.path.join(self.data_path, data_chunk['file'])
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
                
            # Load data file based on extension or type
            if 'type' in data_chunk and data_chunk['type'] == 'feather':
                df = pd.read_feather(file_path)
            elif file_path.endswith('.feather'):
                df = pd.read_feather(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                print(f"Unsupported file format: {file_path}")
                return None
                
            print(f"Loaded data file: {file_path}, shape: {df.shape}")
            
            # Check columns to ensure we have what we need
            feature_cols = self.tech_indicator_list.copy()
            trend_cols = self.tech_indicator_list_trend.copy()
            
            # Check if we have all required columns
            missing_features = [col for col in feature_cols if col not in df.columns]
            missing_trends = [col for col in trend_cols if col not in df.columns]
            
            if missing_features or missing_trends:
                # We're missing some columns - print which ones are available
                print(f"Available columns: {df.columns.tolist()}")
                print(f"Missing features: {missing_features}")
                print(f"Missing trends: {missing_trends}")
                
                # Use available columns as fallback if not all expected features are present
                if missing_features:
                    # Use available numeric columns as features
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if 'open' in numeric_cols and 'high' in numeric_cols and 'low' in numeric_cols and 'close' in numeric_cols:
                        # OHLC data is present, use these as primary features
                        feature_cols = ['open', 'high', 'low', 'close']
                        # Add some more numeric features if available
                        for col in numeric_cols:
                            if col not in feature_cols and col != 'target' and col != 'label':
                                feature_cols.append(col)
                                if len(feature_cols) >= self.n_state_1:
                                    break
                    else:
                        # Use first n_state_1 numeric columns as features
                        feature_cols = numeric_cols[:self.n_state_1]
                    
                    print(f"Using these columns as features: {feature_cols}")
                
                if missing_trends or len(trend_cols) == 0:
                    # Create trend features from price data if available
                    if 'close' in df.columns:
                        # Create simple trend features
                        df['price_change'] = df['close'].pct_change()
                        df['price_change_5'] = df['close'].pct_change(5)
                        df['price_change_10'] = df['close'].pct_change(10)
                        df['rolling_mean_5'] = df['close'].rolling(5).mean()
                        df['rolling_std_5'] = df['close'].rolling(5).std()
                        
                        # Fill NaN values with 0
                        df.fillna(0, inplace=True)
                        
                        trend_cols = ['price_change', 'price_change_5', 'price_change_10', 
                                     'rolling_mean_5', 'rolling_std_5']
                        print(f"Created trend features: {trend_cols}")
                    else:
                        # Use some of the numeric columns as trend features
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        trend_cols = numeric_cols[self.n_state_1:self.n_state_1+self.n_state_2]
                        print(f"Using these columns as trend features: {trend_cols}")
            
            # Create or identify target column
            target_column = None
            for col in ['target', 'label', 'action']:
                if col in df.columns:
                    target_column = col
                    break
                    
            if target_column is None:
                # If no explicit target column, create one based on price movement
                if 'close' in df.columns:
                    # Simple strategy: 1 if price increases, 0 otherwise
                    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
                    target_column = 'target'
                else:
                    print(f"Could not determine target column and no 'close' column found")
                    # Create a random target as a last resort
                    df['target'] = np.random.randint(0, 2, size=len(df))
                    target_column = 'target'
            
            # Create sequences for LSTM input
            sequences = []
            targets = []
            seq_length = self.sequence_length
            
            # Make sure we have enough data for at least one sequence
            if len(df) <= seq_length:
                print(f"Not enough data in {file_path} for a sequence of length {seq_length}")
                return None
            
            # Fill NaN values before creating sequences
            df.fillna(0, inplace=True)
            
            # Prepare features and targets
            for i in range(len(df) - seq_length):
                try:
                    # Get sequence of features
                    feature_seq = df[feature_cols].iloc[i:i+seq_length].values
                    trend_seq = df[trend_cols].iloc[i:i+seq_length].values
                    
                    # Features for LSTM should be shape [sequence_length, feature_dim]
                    combined_seq = np.concatenate([feature_seq, trend_seq], axis=1)
                    sequences.append(combined_seq)
                    
                    # Target is the next action after the sequence
                    target = df[target_column].iloc[i+seq_length]
                    targets.append(target)
                except Exception as e:
                    print(f"Error creating sequence at index {i}: {e}")
                    continue
            
            if not sequences:
                print(f"No valid sequences created from {file_path}")
                return None
                
            # Convert to numpy arrays
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            print(f"Created {len(sequences)} sequences from {file_path}")
            
            return sequences, targets
            
        except Exception as e:
            print(f"Error loading data chunk: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _build_model(self, n_state_1, n_state_2):
        """
        Build the LSTM model
        
        Args:
            n_state_1: Number of features in the single state input
            n_state_2: Number of features in the trend state input
        """
        total_features = n_state_1 + n_state_2
        
        # Simple LSTM model for binary classification
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, total_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary output
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

class LSTMAgent(object):
    def __init__(self, args):
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
            
        self.result_path = os.path.join("./result/low_level", 
                                        '{}'.format(args.dataset), '{}'.format(args.clf), str(int(args.alpha)), "label_1")
        self.model_path = os.path.join(self.result_path,
                                       "seed_{}".format(self.seed))
                                       
        # Use the whole dataset instead of split datasets
        self.data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
                                        
        # Load appropriate data from the whole dataset
        if args.clf == 'lstm':
            # For LSTM agent, we'll load the entire dataset
            # If there's a labels file in the whole dataset, load it
            labels_path = os.path.join(self.data_path, 'slope_labels.pkl')
            if os.path.exists(labels_path):
                with open(labels_path, 'rb') as file:
                    self.data_index = pickle.load(file)
            else:
                # If no labels file exists, we'll need to create our own index
                # This is just a placeholder - implement based on your data structure
                self.data_index = None
        else:
            raise ValueError(f"Classifier type {args.clf} not supported for LSTM agent")

        self.dataset = args.dataset
        self.clf = args.clf
        if "BTC" in self.dataset:
            self.max_holding_number = 0.01
        elif "ETH" in self.dataset:
            self.max_holding_number = 0.2
        elif "DOT" in self.dataset:
            self.max_holding_number = 10
        elif "LTC" in self.dataset:
            self.max_holding_number = 10
        else:
            raise Exception("We do not support other dataset yet")
            
        self.epoch_number = args.epoch_number
        self.sequence_length = args.sequence_length
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()

        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)

        # Initialize LSTM model
        self.model = LSTMSubAgent(
            self.n_state_1, self.n_state_2, self.n_action, 
            self.hidden_size, self.num_layers).to(self.device)
            
        # Training parameters
        self.batch_size = args.batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()

    def act(self, state, state_trend, info):
        """Predict action values using the LSTM model"""
        # Prepare input for model
        if len(state.shape) == 1:  # Single state
            state = np.expand_dims(state, axis=0)  # Add batch dimension
        if len(state_trend.shape) == 1:  # Single trend state
            state_trend = np.expand_dims(state_trend, axis=0)  # Add batch dimension
            
        # Combine features
        combined_features = np.concatenate([state, state_trend], axis=1)
        
        # Add sequence dimension if not present
        if len(combined_features.shape) == 2:
            # For single sample, reshape to [1, sequence_length, features]
            combined_features = np.expand_dims(combined_features, axis=0)
        
        # Get prediction from model
        if self.model is not None:
            # Model returns probability of action 1
            action_prob = self.model.predict(combined_features, verbose=0)[0][0]
            # Convert to binary action
            action = 1 if action_prob > 0.5 else 0
            return action
        else:
            # If no model, return random action
            return np.random.randint(0, 2)

    def act_test(self, state, state_trend, info):
        """Same as act but explicitly for testing"""
        # This is identical to act since we're not using exploration
        return self.act(state, state_trend, info)
        
    def train(self, dataset, end_date, start_date, trade_time_list, tech_indicator_list, tech_indicator_list_trend, feature_number, return_label = 'trend_1', vix_file=None, reverse=False, RL_window_length=10, n_state_1=10, n_state_2=10, learn_start=32, is_train_1=True, label_1=None, verbose=True):
        """
        Train the LSTM model using the available data
        """
        # Store parameters
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.n_state_1 = n_state_1
        self.n_state_2 = n_state_2
        self.sequence_length = RL_window_length
        self.is_train_1 = is_train_1
        self.verbose = verbose
        
        # Set data path based on dataset
        self.data_path = os.path.join(ROOT, "MacroHFT", "data", dataset) if dataset is not None else os.path.join(ROOT, "MacroHFT", "data")
        
        # Check if data directory exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        print(f"Using data path: {self.data_path}")
        
        # Create directory for model saving
        model_save_path = os.path.join(self.model_path, "best_model")
        best_model_path = os.path.join(model_save_path, "best_model")
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(best_model_path, exist_ok=True)
        
        # Build data index
        self.data_index = []
        
        # First look for .feather files in data path
        feather_files = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.feather'):
                    feather_files.append(os.path.join(root, file))
        
        if not feather_files:
            # If no .feather files are found, try other supported formats
            print(f"No .feather files found in {self.data_path}. Looking for other formats.")
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith('.csv') or file.endswith('.parquet'):
                        self.data_index.append({
                            'path': os.path.join(root, file),
                            'type': 'csv' if file.endswith('.csv') else 'parquet'
                        })
        else:
            # Process the found feather files
            print(f"Found {len(feather_files)} .feather files")
            
            # Look for specific structure (train, test, val folders)
            train_files = [f for f in feather_files if '/train/' in f or 'train' in f.lower()]
            
            if train_files:
                print(f"Found {len(train_files)} training files")
                for file_path in train_files:
                    self.data_index.append({
                        'path': file_path,
                        'type': 'feather'
                    })
            else:
                # If no specific train files, use all feather files
                for file_path in feather_files:
                    self.data_index.append({
                        'path': file_path,
                        'type': 'feather'
                    })
        
        # If we still don't have any data files, create sample data
        if not self.data_index:
            print("No data files found. Creating sample data for training.")
            sample_data_path = os.path.join(self.data_path, "sample_data.csv")
            self._create_sample_data(sample_data_path)
            self.data_index.append({
                'path': sample_data_path,
                'type': 'csv'
            })
        
        # Print data index for debugging
        print(f"Data index contains {len(self.data_index)} files:")
        for i, item in enumerate(self.data_index[:5]):  # Show first 5 items
            print(f"  {i+1}. {item['path']}")
        if len(self.data_index) > 5:
            print(f"  ... and {len(self.data_index)-5} more files")
        
        # Train the model
        if self.verbose:
            print("Starting LSTM model training...")
        
        # Initialize model if not already done
        if self.model is None:
            self._build_model(n_state_1, n_state_2)
        
        # Training parameters
        epochs = self.epoch_number
        batch_size = 64
        best_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            # Process each data chunk
            for chunk_idx, data_chunk in enumerate(self.data_index):
                # Load and preprocess data
                data = self._load_data_chunk(data_chunk)
                
                if data is None:
                    continue  # Skip this chunk if loading failed
                
                X, y = data
                
                # Create mini-batches
                num_samples = len(X)
                num_batches = num_samples // batch_size
                
                if num_batches == 0:
                    # Process the entire chunk as one batch if smaller than batch_size
                    loss = self.model.train_on_batch(X, y)
                    epoch_loss += loss
                    batch_count += 1
                else:
                    # Process mini-batches
                    for i in range(num_batches):
                        start_idx = i * batch_size
                        end_idx = (i + 1) * batch_size
                        
                        X_batch = X[start_idx:end_idx]
                        y_batch = y[start_idx:end_idx]
                        
                        loss = self.model.train_on_batch(X_batch, y_batch)
                        epoch_loss += loss
                        batch_count += 1
            
            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            # Save the best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.model.save(os.path.join(best_model_path, f"lstm_model.h5"))
                if self.verbose:
                    print(f"Saved best model with loss: {best_loss:.4f}")
        
        # Save the final model
        self.model.save(os.path.join(model_save_path, f"lstm_model_final.h5"))
        
        if self.verbose:
            print(f"Training completed. Final loss: {avg_epoch_loss:.4f}")
        
        return True
        
    def _create_sample_data(self, file_path):
        """Create sample data for training when no real data is available"""
        # Create a directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Generate sample data
        num_samples = 5000  # Number of data points
        num_features = max(self.n_state_1, 10)  # Number of features
        num_trends = max(self.n_state_2, 5)  # Number of trend features
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(num_features)]
        trend_names = [f'trend_{i}' for i in range(num_trends)]
        
        # Update the feature lists with the generated names
        self.tech_indicator_list = feature_names
        self.tech_indicator_list_trend = trend_names
        
        # Generate random data
        data = np.random.randn(num_samples, num_features + num_trends)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=feature_names + trend_names)
        
        # Add a binary target column (0 or 1)
        df['target'] = np.random.randint(0, 2, size=num_samples)
        
        # Add timestamp column
        base_timestamp = pd.Timestamp('2022-01-01')
        timestamps = [base_timestamp + pd.Timedelta(minutes=i) for i in range(num_samples)]
        df['timestamp'] = timestamps
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        
        print(f"Created sample data file with {num_samples} rows at {file_path}")
        return file_path

    def _load_data_chunk(self, data_chunk):
        """Load and preprocess a chunk of data
        
        Args:
            data_chunk: A reference to the data chunk to load
            
        Returns:
            tuple: (features, targets) or None if loading failed
        """
        try:
            # Determine file path
            if 'path' in data_chunk:
                file_path = data_chunk['path']
            else:
                file_path = os.path.join(self.data_path, data_chunk['file'])
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
                
            # Load data file based on extension or type
            if 'type' in data_chunk and data_chunk['type'] == 'feather':
                df = pd.read_feather(file_path)
            elif file_path.endswith('.feather'):
                df = pd.read_feather(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                print(f"Unsupported file format: {file_path}")
                return None
                
            print(f"Loaded data file: {file_path}, shape: {df.shape}")
            
            # Check columns to ensure we have what we need
            feature_cols = self.tech_indicator_list.copy()
            trend_cols = self.tech_indicator_list_trend.copy()
            
            # Check if we have all required columns
            missing_features = [col for col in feature_cols if col not in df.columns]
            missing_trends = [col for col in trend_cols if col not in df.columns]
            
            if missing_features or missing_trends:
                # We're missing some columns - print which ones are available
                print(f"Available columns: {df.columns.tolist()}")
                print(f"Missing features: {missing_features}")
                print(f"Missing trends: {missing_trends}")
                
                # Use available columns as fallback if not all expected features are present
                if missing_features:
                    # Use available numeric columns as features
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if 'open' in numeric_cols and 'high' in numeric_cols and 'low' in numeric_cols and 'close' in numeric_cols:
                        # OHLC data is present, use these as primary features
                        feature_cols = ['open', 'high', 'low', 'close']
                        # Add some more numeric features if available
                        for col in numeric_cols:
                            if col not in feature_cols and col != 'target' and col != 'label':
                                feature_cols.append(col)
                                if len(feature_cols) >= self.n_state_1:
                                    break
                    else:
                        # Use first n_state_1 numeric columns as features
                        feature_cols = numeric_cols[:self.n_state_1]
                    
                    print(f"Using these columns as features: {feature_cols}")
                
                if missing_trends or len(trend_cols) == 0:
                    # Create trend features from price data if available
                    if 'close' in df.columns:
                        # Create simple trend features
                        df['price_change'] = df['close'].pct_change()
                        df['price_change_5'] = df['close'].pct_change(5)
                        df['price_change_10'] = df['close'].pct_change(10)
                        df['rolling_mean_5'] = df['close'].rolling(5).mean()
                        df['rolling_std_5'] = df['close'].rolling(5).std()
                        
                        # Fill NaN values with 0
                        df.fillna(0, inplace=True)
                        
                        trend_cols = ['price_change', 'price_change_5', 'price_change_10', 
                                     'rolling_mean_5', 'rolling_std_5']
                        print(f"Created trend features: {trend_cols}")
                    else:
                        # Use some of the numeric columns as trend features
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        trend_cols = numeric_cols[self.n_state_1:self.n_state_1+self.n_state_2]
                        print(f"Using these columns as trend features: {trend_cols}")
            
            # Create or identify target column
            target_column = None
            for col in ['target', 'label', 'action']:
                if col in df.columns:
                    target_column = col
                    break
                    
            if target_column is None:
                # If no explicit target column, create one based on price movement
                if 'close' in df.columns:
                    # Simple strategy: 1 if price increases, 0 otherwise
                    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
                    target_column = 'target'
                else:
                    print(f"Could not determine target column and no 'close' column found")
                    # Create a random target as a last resort
                    df['target'] = np.random.randint(0, 2, size=len(df))
                    target_column = 'target'
            
            # Create sequences for LSTM input
            sequences = []
            targets = []
            seq_length = self.sequence_length
            
            # Make sure we have enough data for at least one sequence
            if len(df) <= seq_length:
                print(f"Not enough data in {file_path} for a sequence of length {seq_length}")
                return None
            
            # Fill NaN values before creating sequences
            df.fillna(0, inplace=True)
            
            # Prepare features and targets
            for i in range(len(df) - seq_length):
                try:
                    # Get sequence of features
                    feature_seq = df[feature_cols].iloc[i:i+seq_length].values
                    trend_seq = df[trend_cols].iloc[i:i+seq_length].values
                    
                    # Features for LSTM should be shape [sequence_length, feature_dim]
                    combined_seq = np.concatenate([feature_seq, trend_seq], axis=1)
                    sequences.append(combined_seq)
                    
                    # Target is the next action after the sequence
                    target = df[target_column].iloc[i+seq_length]
                    targets.append(target)
                except Exception as e:
                    print(f"Error creating sequence at index {i}: {e}")
                    continue
            
            if not sequences:
                print(f"No valid sequences created from {file_path}")
                return None
                
            # Convert to numpy arrays
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            print(f"Created {len(sequences)} sequences from {file_path}")
            
            return sequences, targets
            
        except Exception as e:
            print(f"Error loading data chunk: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def save_model(self, path=None):
        """Save the LSTM model"""
        if path is None:
            path = os.path.join(self.model_path, "best_model.pkl")
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path=None):
        """Load the LSTM model"""
        if path is None:
            path = os.path.join(self.model_path, "best_model.pkl")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            return True
        return False

    def _build_model(self, n_state_1, n_state_2):
        """
        Build the LSTM model
        
        Args:
            n_state_1: Number of features in the single state input
            n_state_2: Number of features in the trend state input
        """
        total_features = n_state_1 + n_state_2
        
        # Simple LSTM model for binary classification
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, total_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary output
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to data directory")
    parser.add_argument("--dataset", type=str, default="ETHUSDT", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="lstm", help="Model name")
    parser.add_argument("--n_state_1", type=int, default=10, help="Number of state features")
    parser.add_argument("--n_state_2", type=int, default=10, help="Number of trend features")
    parser.add_argument("--RL_window_length", type=int, default=10, help="Sequence length for LSTM")
    parser.add_argument("--epoch_number", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--is_train", type=str, default="true", help="Whether to train the model (true/false)")
    parser.add_argument("--tech_indicator_list", type=str, default="open,high,low,close,volume", 
                        help="Comma-separated list of technical indicators")
    parser.add_argument("--tech_indicator_list_trend", type=str, default="price_change,price_change_5", 
                        help="Comma-separated list of trend indicators")
    
    args = parser.parse_args()
    
    # Parse technical indicators
    tech_indicator_list = args.tech_indicator_list.split(',')
    tech_indicator_list_trend = args.tech_indicator_list_trend.split(',')
    
    # Convert is_train string to boolean
    is_train = args.is_train.lower() == "true"
    
    print("LSTM Agent Starting...")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print(f"Training mode: {is_train}")
    
    # Create and configure the agent
    agent = LSTMSubAgent(
        model_name=args.model_name,
        epoch_number=args.epoch_number,
        data_path=args.data_path
    )
    
    if is_train:
        print("Starting training...")
        agent.train(
            dataset=args.dataset,
            end_date=None,  # Not used but part of interface
            start_date=None,  # Not used but part of interface
            trade_time_list=None,  # Not used but part of interface
            tech_indicator_list=tech_indicator_list,
            tech_indicator_list_trend=tech_indicator_list_trend,
            feature_number=len(tech_indicator_list) + len(tech_indicator_list_trend),
            RL_window_length=args.RL_window_length,
            n_state_1=args.n_state_1,
            n_state_2=args.n_state_2,
            is_train_1=True,
            verbose=True
        )
        print("Training completed.")
    else:
        print("Skipping training (is_train=false).")
