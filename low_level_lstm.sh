#!/bin/bash

# Create necessary directories
# mkdir -p logs
# mkdir -p result/low_level/ETHUSDT/best_model/lstm
# mkdir -p result/low_level/ETHUSDT/best_model/lstm/1

# Run LSTM agent training
nohup python3 -u RL/agent/low_level_lstm.py --dataset ETHUSDT --epoch_number 15 --n_state_1 10 --n_state_2 10 --batch_size 512 --lr 1e-4 --tech_indicator_list "open,high,low,close,volume" --tech_indicator_list_trend "price_change,price_change_5" 2>&1 &