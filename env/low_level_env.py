from logging import raiseExceptions
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import argparse
import os
import torch
import sys
import pathlib
import pdb

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from MacroHFT.tools.demonstration import make_q_table_reward

tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()



transcation_cost = 0.0002
back_time_length = 1
max_holding_number = 0.01
alpha = 0


class Testing_Env(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        initial_action=0,
    ):
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.df = df
        self.initial_action = initial_action
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(back_time_length * len(self.tech_indicator_list), ))
        self.terminal = False
        self.stack_length = back_time_length
        self.m = back_time_length
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = initial_action
        self.comission_fee = transcation_cost
        self.max_holding_number = max_holding_number
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.previous_position = initial_action * self.max_holding_number
        self.position = initial_action * self.max_holding_number
        self.initial_action = initial_action



    def calculate_value(self, price_information, position):
        print("low_level_agent->calculate_value(): ", price_information["close"] * position)

        return price_information["close"] * position

    def reset(self):
        self.terminal = False
        self.m = back_time_length
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = self.initial_action
        price_information = self.data.iloc[-1]
        self.previous_position = self.initial_action * self.max_holding_number
        self.position = self.initial_action * self.max_holding_number
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.previous_position = self.initial_action * self.max_holding_number
        self.position = self.initial_action * self.max_holding_number
        self.needed_money_memory.append(self.position *
                                        self.data.iloc[-1]["close"])
        self.sell_money_memory.append(0)

        print("low_level_agent->reset()")

        return self.single_state, self.trend_state, {
            "previous_action": self.initial_action,
        }

    def step(self, action):
        normlized_action = action
        position = self.max_holding_number * normlized_action # Tính số coin agent nắm giữ
        self.terminal = (self.m >= len(self.df.index.unique()) - 1) # Kiểm tra xem đã đến cuối dữ liệu chưa (done=True)
        previous_position = self.previous_position
        previous_price_information = self.data.iloc[-1]
        self.m += 1
        self.data = self.df.iloc[self.m - self.stack_length:self.m] # ???
        current_price_information = self.data.iloc[-1] # Lấy giá (và thông tin khác) tại dòng cuối trong self.data (chính là giá cũ)
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values

        self.previous_position = previous_position
        self.position = position # Vị thế mới = position (sau action)
        self.changing = (self.position != self.previous_position) # Kiểm tra agent có thay đổi vị thế hay không

        if previous_position >= position: # nghĩa là agent bán bớt (hoặc giữ nguyên) số coin
            self.sell_size = previous_position - position # tính số coin bán, case sell_size=0 là hold

            print("low_level_agent->step(): position: ", self.position)
            print("low_level_agent->step(): sell_size: ", self.sell_size)

            cash = self.sell_size * previous_price_information['close'] * (1 - self.comission_fee)
            self.comission_fee_history.append(self.comission_fee * self.sell_size * previous_price_information['close']) # ghi lại phí giao dịch

            print("low_level_agent->step(): cash: ", cash)

            self.sell_money_memory.append(cash) # lưu tiền thu về từ việc bán coin
            self.needed_money_memory.append(0) # lưu tiền cần thiết để mua coin
            self.position = position
            previous_value = self.calculate_value(previous_price_information,
                                                  self.previous_position) # giá trị tài sản trước khi thay đổi vị thế
            current_value = self.calculate_value(current_price_information,
                                                 self.position) # giá trị tài sản sau khi thay đổi vị thế
            self.reward = current_value + cash - previous_value # lãi/lỗ = giá trị mới + tiền bán - giá trị cũ

            print("low_level_agent->step(): reward: sell ", self.reward)
            print("low_level_agent->step(): current_value: sell ", current_value)
            print("low_level_agent->step(): previous_value: sell ", previous_value)

            if previous_value == 0:
                return_rate = 0
            else:
                return_rate = (current_value + cash -
                               previous_value) / previous_value # Nếu return_rate > 0 return_rate>0 ⇒ lãi (tăng so với giá trị cũ), ngược lại thì lỗ
            self.return_rate = return_rate
            self.reward_history.append(self.reward)

            print("low_level_agent->step(): return_rate: ", self.return_rate)

        if previous_position < position: # nghĩa là agent mua thêm coin
            self.buy_size = position - previous_position # tính số coin mua
            needed_cash = self.buy_size * previous_price_information['close'] * (1 + self.comission_fee) # tính tiền cần thiết để mua coin
            self.comission_fee_history.append(self.comission_fee * self.buy_size * previous_price_information['close']) # ghi lại phí giao dịch

            print("low_level_agent->step(): position: ", self.position)
            print("low_level_agent->step(): buy_size: ", self.buy_size)
            print("low_level_agent->step(): needed_cash: ", needed_cash)

            self.needed_money_memory.append(needed_cash) # lưu tiền cần thiết để mua coin
            self.sell_money_memory.append(0) # lưu tiền thu về từ việc bán coin

            self.position = position
            previous_value = self.calculate_value(previous_price_information,
                                                  self.previous_position) # giá trị tài sản trước khi thay đổi vị thế
            current_value = self.calculate_value(current_price_information,
                                                 self.position) # giá trị tài sản sau khi thay đổi vị thế
            self.reward = current_value - needed_cash - previous_value # lãi/lỗ = giá trị mới - tiền cần thiết - giá trị cũ
            return_rate = (current_value - needed_cash -
                           previous_value) / (previous_value + needed_cash) # Nếu return_rate > 0 return_rate>0 ⇒ lãi (tăng so với giá trị cũ), ngược lại thì lỗ

            print("low_level_agent->step(): reward: buy ", self.reward)
            print("low_level_agent->step(): current_value: buy ", current_value)
            print("low_level_agent->step(): previous_value: buy ", previous_value)

            self.reward_history.append(self.reward)
            self.return_rate = return_rate

        self.previous_position = self.position # cập nhật vị thế trước đó

        if self.terminal:
            return_margin, pure_balance, required_money, commission_fee = self.get_final_return_rate(
            )
            self.pured_balance = pure_balance # số tiền ròng (tiền thu - tiền mua)
            self.final_balance = self.pured_balance + self.calculate_value(
                current_price_information, self.position) # pured_balance + giá trị coin hiện tại
            self.required_money = required_money
            
            print("low_level_agent->step(): the portfit margine is ", self.final_balance / self.required_money)
            print("low_level_agent->step(): final_balance: ", self.final_balance)
            print("low_level_agent->step(): required_money: ", self.required_money)
            print("low_level_agent->step(): commission_fee: ", commission_fee)

        return self.single_state, self.trend_state, self.reward, self.terminal, {
            "previous_action": action,
        }

    def get_final_return_rate(self, slient=False):
        sell_money_memory = np.array(self.sell_money_memory)
        needed_money_memory = np.array(self.needed_money_memory)
        true_money = sell_money_memory - needed_money_memory
        final_balance = np.sum(true_money)
        balance_list = []
        for i in range(len(true_money)):
            balance_list.append(np.sum(true_money[:i + 1]))
        required_money = -np.min(balance_list)
        commission_fee = np.sum(self.comission_fee_history)

        print("low_level_agent->get_final_return_rate(): final_balance: ", final_balance)
        print("low_level_agent->get_final_return_rate(): required_money: ", required_money)
        print("low_level_agent->get_final_return_rate(): commission_fee: ", commission_fee)

        return final_balance / required_money, final_balance, required_money, commission_fee


class Training_Env(Testing_Env):
    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        initial_action = 0,
        alpha=alpha,
    ):
        super(Training_Env,
              self).__init__(df, tech_indicator_list, tech_indicator_list_trend, transcation_cost,
                             back_time_length, max_holding_number)
        self.q_table = make_q_table_reward(df,
                                           num_action=2,
                                           max_holding=max_holding_number,
                                           commission_fee=0.001,
                                           reward_scale=1,
                                           gamma=0.99,
                                           max_punish=1e12)
        self.initial_action = initial_action


    def reset(self):
        single_state, trend_state, info = super(Training_Env, self).reset()
        self.previous_action = self.initial_action
        self.previous_position = self.initial_action * self.max_holding_number
        self.position = self.initial_action * self.max_holding_number
        info['q_value'] = self.q_table[self.m - 1][self.previous_action][:]
        return single_state, trend_state, info

    def step(self, action):
        single_state, trend_state, reward, done, info = super(Training_Env, self).step(action)
        info['q_value'] = self.q_table[self.m - 1][action][:]
        return single_state, trend_state, reward, done, info


