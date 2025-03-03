import numpy as np
import pandas as pd


def make_q_table_reward(df: pd.DataFrame,
                        num_action,
                        max_holding,
                        reward_scale=1000,
                        gamma=0.999,
                        commission_fee=0.001,
                        max_punish=1e12):
    q_table = np.zeros((len(df), num_action, num_action))

    def calculate_value(price_information, position):
        return price_information["close"] * position

    scale_factor = num_action - 1

    for t in range(2, len(df) + 1):
        current_price_information = df.iloc[-t]
        future_price_information = df.iloc[-t + 1] # Tính Q-value ngược từ cuối DF về đầu

        for previous_action in range(num_action):
            for current_action in range(num_action):
                if current_action > previous_action: # trường hợp tăng vị thế (mua thêm)
                    previous_position = previous_action / (
                        scale_factor) * max_holding # số coin trước đó

                    current_position = current_action / (
                        scale_factor) * max_holding # số coin hiện tại
            
                    position_change = (current_action-previous_action) / \
                        scale_factor*max_holding # số coin mới mua = current_position - previous_position

                    buy_money = position_change * current_price_information['close'] * (1 + commission_fee) # tiền bỏ ra để mua (có tính phí)

                    current_value = calculate_value(current_price_information,
                                                    previous_position) # giá trị coin cũ tại giá hiện tại

                    future_value = calculate_value(future_price_information,
                                                   current_position) # giá trị coin cũ tại giá tương lai

                    reward = future_value - (current_value + buy_money) # lãi hoặc lỗ

                    reward = reward_scale * reward # nhân thêm hệ số để phóng đại reward

                    q_table[len(df) - t][previous_action][
                        current_action] = reward + gamma * np.max(
                            q_table[len(df) - t + 1][current_action][:]) # công thức q-learning

                else: # giảm vị thế hoặc giữ nguyên (bán bớt hoặc giữ nguyên)
                    previous_position = previous_action / (
                        scale_factor) * max_holding
                    current_position = current_action / (
                        scale_factor) * max_holding
                    position_change = (previous_action-current_action) / \
                        scale_factor*max_holding
                    sell_money = position_change * current_price_information['close'] * (1 - commission_fee)
                    current_value = calculate_value(current_price_information,
                                                    previous_position)
                    future_value = calculate_value(future_price_information,
                                                   current_position)
                    reward = future_value + sell_money - current_value
                    reward = reward_scale * reward
                    q_table[len(df) - t][previous_action][
                        current_action] = reward + gamma * np.max(
                            q_table[len(df) - t + 1][current_action][:])
    return q_table