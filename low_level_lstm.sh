#!/bin/bash

# Create necessary directories
mkdir -p logs
mkdir -p result/low_level/ETHUSDT/best_model/lstm
mkdir -p result/low_level/ETHUSDT/best_model/lstm/1

# Run LSTM agent training
nohup python3 -m RL.agent.low_level_lstm \
  --data_path "MacroHFT/data/ETHUSDT/df_train.feather" \
  --dataset "ETHUSDT" \
  --model_name "lstm" \
  --n_state_1 10 \
  --n_state_2 10 \
  --RL_window_length 10 \
  --epoch_number 15 \
  --is_train "true" \
  --tech_indicator_list "open,high,low,close,volume" \
  --tech_indicator_list_trend "price_change,price_change_5,price_change_10,rolling_mean_5,rolling_std_5" >./lstm.log 2>&1 &