import pathlib
import sys
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import joblib
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from MacroHFT.model.net import *
from MacroHFT.env.high_level_env import Testing_Env, Training_Env
from MacroHFT.RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from MacroHFT.RL.util.replay_buffer import ReplayBuffer_High
from MacroHFT.RL.util.memory import episodicmemory

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",type=int,default=1000000,)
parser.add_argument("--dataset",type=str,default="ETHUSDT")
parser.add_argument("--q_value_memorize_freq",type=int, default=10,)
parser.add_argument("--batch_size",type=int,default=512)
parser.add_argument("--eval_update_freq",type=int,default=512)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epsilon_start",type=float,default=0.7)
parser.add_argument("--epsilon_end",type=float,default=0.3)
parser.add_argument("--decay_length",type=int,default=5)
parser.add_argument("--update_times",type=int,default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost",type=float,default=0.2 / 1000)
parser.add_argument("--back_time_length",type=int,default=1)
parser.add_argument("--seed",type=int,default=12345)
parser.add_argument("--n_step",type=int,default=1)
parser.add_argument("--epoch_number",type=int,default=15)
parser.add_argument("--device",type=str,default="cuda:0")
parser.add_argument("--alpha",type=float,default=0.5)
parser.add_argument("--beta",type=int,default=5)
parser.add_argument("--exp",type=str,default="exp1")
parser.add_argument("--num_step",type=int,default=10)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
        self.result_path = os.path.join("./result/high_level", '{}'.format(args.dataset), args.exp)
        self.model_path = os.path.join(self.result_path,
                                       "seed_{}".format(self.seed))
        self.train_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
        self.val_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
        self.test_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
        self.dataset=args.dataset
        self.num_step = args.num_step

        if "BTC" in self.dataset:
            self.max_holding_number=0.01
        elif "ETH" in self.dataset:
            self.max_holding_number=0.2
        elif "DOT" in self.dataset:
            self.max_holding_number=10
        elif "LTC" in self.dataset:
            self.max_holding_number=10
        else:
            raise Exception ("we do not support other dataset yet")

        self.epoch_number = args.epoch_number # số epoch huấn luyện
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.writer = SummaryWriter(self.log_path) # khởi tạo công cụ ghi log (loss, reward,...) lên TensorBoard
        self.update_counter = 0 # đếm số lần update mô hình
        self.q_value_memorize_freq = args.q_value_memorize_freq # ần suất ghi nhớ Q-value (nếu dùng demonstration)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()
        self.clf_list = ['slope_360', 'vol_360']

        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length # số bước lookback
        self.n_action = 2 # 2 hành động (0 hoặc 1) – nắm giữ hay không

        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)

        self.slope_1 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)

        self.slope_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)

        self.slope_3 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)

        self.vol_1 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)

        self.vol_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)

        self.vol_3 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)

        model_list_slope = [
            "./result/low_level/ETHUSDT/best_model/slope/1/best_model.pkl",
            "./result/low_level/ETHUSDT/best_model/slope/2/best_model.pkl",
            "./result/low_level/ETHUSDT/best_model/slope/3/best_model.pkl"
        ]
        model_list_vol = [
            "./result/low_level/ETHUSDT/best_model/vol/1/best_model.pkl",
            "./result/low_level/ETHUSDT/best_model/vol/2/best_model.pkl",
            "./result/low_level/ETHUSDT/best_model/vol/3/best_model.pkl"
        ]

        # Load trọng số (state_dict) từ file best_model.pkl (đã train ở "low_level")
        self.slope_1.load_state_dict(
            torch.load(model_list_slope[0], map_location=self.device))
        self.slope_2.load_state_dict(
            torch.load(model_list_slope[1], map_location=self.device))
        self.slope_3.load_state_dict(
            torch.load(model_list_slope[2], map_location=self.device))

        self.vol_1.load_state_dict(
            torch.load(model_list_vol[0], map_location=self.device))
        self.vol_2.load_state_dict(
            torch.load(model_list_vol[1], map_location=self.device))
        self.vol_3.load_state_dict(
            torch.load(model_list_vol[2], map_location=self.device))

        self.slope_1.eval()
        self.slope_2.eval()
        self.slope_3.eval()

        self.vol_1.eval()
        self.vol_2.eval()
        self.vol_3.eval()

        self.slope_agents = {
            0: self.slope_1,
            1: self.slope_2,
            2: self.slope_3
        }
        self.vol_agents = {
            0: self.vol_1,
            1: self.vol_2,
            2: self.vol_3
        }

        # Add LSTM agent (7th sub-agent)
        self.lstm_agent = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        
        # Path for the LSTM model
        lstm_model_path = "./result/low_level/ETHUSDT/best_model/lstm/1/best_model.pkl"
        
        # Try to load the model if it exists, otherwise it will be initialized with random weights
        if os.path.exists(lstm_model_path):
            self.lstm_agent.load_state_dict(
                torch.load(lstm_model_path, map_location=self.device))
        
        self.lstm_agent.eval()
        
        # Update hyperagent to handle 7 sub-agents instead of 6
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32, num_subagents=7).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32, num_subagents=7).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())
        self.update_times = args.update_times # số lần update liên tiếp mỗi khi đủ điều kiện
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(), 
                                          lr=args.lr) # tối ưu tham số hyperagent
        self.loss_func = nn.MSELoss() # hàm mất mát (MSE) cho Q-learning
        self.batch_size = args.batch_size
        self.gamma = args.gamma # discount factor
        self.tau = args.tau # soft update factor
        self.n_step = args.n_step # n-step RL
        self.eval_update_freq = args.eval_update_freq # tần suất update
        self.buffer_size = args.buffer_size # kích thước ReplayBuffer
        self.epsilon_start = args.epsilon_start # tham số epsilon-greedy
        self.epsilon_end = args.epsilon_end
        self.decay_length = args.decay_length
        self.epsilon_scheduler = LinearDecaySchedule(start_epsilon=self.epsilon_start, end_epsilon=self.epsilon_end, decay_length=self.decay_length) # hàm tuyến tính giảm dần epsilon
        self.epsilon = args.epsilon_start
        self.memory = episodicmemory(4320, 5, self.n_state_1, self.n_state_2, 64, self.device)

    # vector trọng số (được "hyperagent" sinh ra), có shape [batch, 6], đại diện cho mức độ "tin tưởng" 6 sub-agent (3 slope + 3 vol)
    # danh sách Q-value từ 6 sub-agent (mỗi sub-agent ra [batch, action_dim]). Ở đây action_dim = 2, nên mỗi sub-agent cho [batch, 2]
    # hàm này là bước "trộn" Q-value của nhiều sub-agent để ra một Q-value cuối, giúp mô hình "hyperagent" khai thác những gì 6 sub-agent đã học
    def calculate_q(self, w, qs):
        q_tensor = torch.stack(qs)           # [7, batch, action_dim]
        q_tensor = q_tensor.permute(1, 0, 2) # [batch, 7, action_dim]
        weights_reshaped = w.view(-1, 1, 7)  # [batch, 1, 7]
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)

        return combined_q  # [batch, action_dim]

    def update(self, replay_buffer):
        self.hyperagent.train()
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            next_qs = [
                self.slope_agents[0](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                self.slope_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                self.slope_agents[2](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                self.vol_agents[0](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                self.vol_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                self.vol_agents[2](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                self.lstm_agent(batch['next_state'], batch['next_state_trend'], batch['next_previous_action'])
            ]

            next_w = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_class_state'],
                                    batch['next_previous_action'])

            next_actions_value = self.calculate_q(next_w, next_qs)

            a_argmax = next_actions_value.argmax(dim=-1, keepdim=True)
            next_target_w = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_class_state'],
                                    batch['next_previous_action'])
            next_target_actions_value = self.calculate_q(next_target_w, next_qs)
            q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * next_target_actions_value.gather(-1, a_argmax).squeeze(-1)

        qs = [
            self.slope_agents[0](batch['state'], batch['state_trend'], batch['previous_action']),
            self.slope_agents[1](batch['state'], batch['state_trend'], batch['previous_action']),
            self.slope_agents[2](batch['state'], batch['state_trend'], batch['previous_action']),
            self.vol_agents[0](batch['state'], batch['state_trend'], batch['previous_action']),
            self.vol_agents[1](batch['state'], batch['state_trend'], batch['previous_action']),
            self.vol_agents[2](batch['state'], batch['state_trend'], batch['previous_action']),
            self.lstm_agent(batch['state'], batch['state_trend'], batch['previous_action'])
        ]

        w = self.hyperagent(batch['state'], batch['state_trend'], batch['class_state'],
                                batch['previous_action'])
        actions_value = self.calculate_q(w, qs)
        q_current = actions_value.gather(-1, batch['action']).squeeze(-1)

        td_error = self.loss_func(q_current, q_target)
        memory_error = self.loss_func(q_current, batch['q_memory'])
        
        self.optimizer.zero_grad()
        loss = td_error + args.beta * memory_error
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hyperagent.parameters(), 1)
        self.optimizer.step()

        for param, target_param in zip(self.hyperagent.parameters(), self.hyperagent_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.update_counter += 1
        self.hyperagent.eval()
        return w.detach().cpu().numpy(), td_error.detach().cpu(), memory_error.detach().cpu()

    def act(self, state, state_trend, state_clf, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long(), 0).to(self.device)

        if np.random.uniform() < (1-self.epsilon):
            with torch.no_grad():
                qs = [
                    self.slope_agents[0](x1, x2, previous_action),
                    self.slope_agents[1](x1, x2, previous_action),
                    self.slope_agents[2](x1, x2, previous_action),
                    self.vol_agents[0](x1, x2, previous_action),
                    self.vol_agents[1](x1, x2, previous_action),
                    self.vol_agents[2](x1, x2, previous_action),
                    self.lstm_agent(x1, x2, previous_action)
                ]
                w = self.hyperagent(x1, x2, x3, previous_action)
                actions_value = self.calculate_q(w, qs)
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()
                action = action[0]
        else:
            action_choice = [0,1]
            action = random.choice(action_choice)

        self.epsilon = self.epsilon_scheduler.get_epsilon()
        return action

    def act_test(self, state, state_trend, state_clf, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long(), 0).to(self.device)

        with torch.no_grad():
            qs = [
                self.slope_agents[0](x1, x2, previous_action),
                self.slope_agents[1](x1, x2, previous_action),
                self.slope_agents[2](x1, x2, previous_action),
                self.vol_agents[0](x1, x2, previous_action),
                self.vol_agents[1](x1, x2, previous_action),
                self.vol_agents[2](x1, x2, previous_action),
                self.lstm_agent(x1, x2, previous_action)
            ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]

        return action

    def q_estimate(self, state, state_trend, state_clf, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long(), 0).to(self.device)

        with torch.no_grad():
            qs = [
                self.slope_agents[0](x1, x2, previous_action),
                self.slope_agents[1](x1, x2, previous_action),
                self.slope_agents[2](x1, x2, previous_action),
                self.vol_agents[0](x1, x2, previous_action),
                self.vol_agents[1](x1, x2, previous_action),
                self.vol_agents[2](x1, x2, previous_action),
                self.lstm_agent(x1, x2, previous_action)
            ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)

        return actions_value.cpu().numpy()

    def calculate_hidden(self, state, state_trend, info): # embedding của state (kết hợp x1, x2, previous_action) bên trong hyperagent, trước khi ra trọng số w
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)

        print("high_level_agent->calculate_hidden(): x1 ", x1)
        print("high_level_agent->calculate_hidden(): x2 ", x2)
        print("high_level_agent->calculate_hidden(): previous_action ", previous_action)

        with torch.no_grad(): # tắt gradient (không cập nhật)
            hs = self.hyperagent.encode(x1, x2, previous_action).cpu().numpy() # hidden state (vector) của hyperagent, shape [batch, hidden_dim * 2] hoặc tương tự        
        return hs

    def train(self):
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        step_counter = 0 # đếm tổng số bước (step) đã thực hiện (giữa các epoch)
        episode_counter = 0 # đếm số episode đã xong
        epoch_counter = 0 # đếm số epoch đã xong
        best_return_rate = -float('inf') # lưu trữ giá trị tốt nhất (để so sánh và cập nhật best model)
        best_model = None
        self.replay_buffer = ReplayBuffer_High(args, self.n_state_1, self.n_state_2, self.n_action) # ReplayBuffer dành cho high-level agent (có thêm field "q_memory", "demo_action",… so với ReplayBuffer thường)

        for sample in range(self.epoch_number):
            print('high_level_agent->train() epoch ', epoch_counter + 1)

            self.df = pd.read_feather(
                os.path.join(self.train_data_path, "train.feather"))
            
            train_env = Training_Env(
                    df=self.df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    clf_list=self.clf_list,
                    transcation_cost=self.transcation_cost,
                    back_time_length=self.back_time_length,
                    max_holding_number=self.max_holding_number,
                    initial_action=random.choices(range(self.n_action), k=1)[0],
                    alpha = 0)
            s, s2, s3, info = train_env.reset()
            episode_reward_sum = 0

            print("high_level_agent->train(): s ", s)
            print("high_level_agent->train(): s2 ", s2)
            print("high_level_agent->train(): s3 ", s3)
            
            while True:
                a = self.act(s, s2, s3, info) # agent chọn action 0/1, dùng "hyperagent" + sub-agent Q-value gộp
                s_, s2_, s3_, r, done, info_ = train_env.step(a) # environment thực hiện action → trả về next_state, reward, done, info_
                hs = self.calculate_hidden(s, s2, info) # tính vector ẩn (embedding) của hyperagent (chưa qua net) cho state hiện tại
                q = r + self.gamma * (1 - done) * self.q_estimate(s_, s2_, s3_, info_)
                q_memory = self.memory.query(hs, a) # tra cứu Q cũ đã lưu cho (hs, action)

                print("high_level_agent->train()->while: a, s, s2, s3, r, done, q, q_memory", a, s_, s2_, s3_, r, done, q, q_memory)

                if np.isnan(q_memory):
                    q_memory = q

                # lưu transition (state, action, reward, next_state, done, q_memory, etc.) vào ReplayBuffer
                self.replay_buffer.store_transition(s, s2, s3, info['previous_action'], info['q_value'], a, r, s_, s2_, s3_, info_['previous_action'],
                                info_['q_value'], done, q_memory)
                self.memory.add(hs, a, q, s, s2, info['previous_action']) # cập nhật "episodicmemory"
                episode_reward_sum += r

                # print("high_level_agent->train()->while: replay_buffer ", self.replay_buffer)
                print("high_level_agent->train()->while: episode_reward_sum ", episode_reward_sum)

                s, s2, s3, info = s_, s2_, s3_, info_
                step_counter += 1
                if step_counter % self.eval_update_freq == 0 and step_counter > (
                        self.batch_size + self.n_step):
                    for i in range(self.update_times):
                        w, td_error, memory_error = self.update(self.replay_buffer) # để huấn luyện hyperagent
                        if self.update_counter % self.q_value_memorize_freq == 1:
                            self.writer.add_scalar(
                                tag="td_error",
                                scalar_value=td_error,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="memory_error",
                                scalar_value=memory_error,
                                global_step=self.update_counter,
                                walltime=None)
                    if step_counter > 4320:
                        self.memory.re_encode(self.hyperagent)

                if done:
                    print("high_level_agent->train()->while: done ", done)
                    print("high_level_agent->train()->while: episode_reward_sum ", episode_reward_sum)
                    break

            episode_counter += 1
            final_balance, required_money = train_env.final_balance, train_env.required_money
            self.writer.add_scalar(tag="return_rate_train",
                                scalar_value=final_balance / (required_money),
                                global_step=episode_counter,
                                walltime=None)
            self.writer.add_scalar(tag="final_balance_train",
                                scalar_value=final_balance,
                                global_step=episode_counter,
                                walltime=None)
            self.writer.add_scalar(tag="required_money_train",
                                scalar_value=required_money,
                                global_step=episode_counter,
                                walltime=None)
            self.writer.add_scalar(tag="reward_sum_train",
                                scalar_value=episode_reward_sum,
                                global_step=episode_counter,
                                walltime=None)
            # Lưu 4 chỉ số vào list (để tính trung bình cuối epoch)
            epoch_return_rate_train_list.append(final_balance / (required_money))
            epoch_final_balance_train_list.append(final_balance)
            epoch_required_money_train_list.append(required_money)
            epoch_reward_sum_train_list.append(episode_reward_sum)

            # Tăng epoch_counter, cập nhật epsilon, tính trung bình
            epoch_counter += 1
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch_counter)
            
            mean_return_rate_train = np.mean(epoch_return_rate_train_list)
            mean_final_balance_train = np.mean(epoch_final_balance_train_list)
            mean_required_money_train = np.mean(epoch_required_money_train_list)
            mean_reward_sum_train = np.mean(epoch_reward_sum_train_list)

            print("high_level_agent->train(): mean_return_rate_train ", mean_return_rate_train)
            print("high_level_agent->train(): mean_final_balance_train ", mean_final_balance_train)
            print("high_level_agent->train(): mean_required_money_train ", mean_required_money_train)
            print("high_level_agent->train(): mean_reward_sum_train ", mean_reward_sum_train)

            self.writer.add_scalar(
                    tag="epoch_return_rate_train",
                    scalar_value=mean_return_rate_train,
                    global_step=epoch_counter,
                    walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_final_balance_train",
                scalar_value=mean_final_balance_train,
                global_step=epoch_counter,
                walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_required_money_train",
                scalar_value=mean_required_money_train,
                global_step=epoch_counter,
                walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_reward_sum_train",
                scalar_value=mean_reward_sum_train,
                global_step=epoch_counter,
                walltime=None,
                )

            epoch_path = os.path.join(self.model_path,
                                        "epoch_{}".format(epoch_counter))
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            torch.save(self.hyperagent.state_dict(),
                        os.path.join(epoch_path, "trained_model.pkl"))
            val_path = os.path.join(epoch_path, "val")
            if not os.path.exists(val_path):
                    os.makedirs(val_path)

            return_rate_eval = self.val_cluster(epoch_path, val_path) # Gọi hàm val_cluster để đánh giá mô hình trên tập val

            if return_rate_eval > best_return_rate: # Nếu tốt hơn best_return_rate, cập nhật best_return_rate và best_model
                best_return_rate = return_rate_eval
                best_model = self.hyperagent.state_dict()
                print("high_level_agent->train(): best_return_rate ", best_return_rate)

            epoch_return_rate_train_list = []
            epoch_final_balance_train_list = []
            epoch_required_money_train_list = []
            epoch_reward_sum_train_list = []

        best_model_path = os.path.join("./result/high_level", 
                                        '{}'.format(self.dataset), 'best_model.pkl')
        torch.save(best_model, best_model_path)
        final_result_path = os.path.join("./result/high_level", '{}'.format(self.dataset))
        final_result_path = os.path.join(final_result_path, "val")
        self.test_cluster(epoch_path, final_result_path)

    def val_cluster(self, epoch_path, save_path): # hàm này giúp đánh giá mô hình (val hoặc test), luôn dùng act_test (argmax Q), không update DQN, chỉ ghi lại kết quả
        self.hyperagent.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        self.hyperagent.eval() # chuyển mạng sang chế độ đánh giá (tắt dropout, batchnorm ở eval mode)
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = pd.read_feather(
            os.path.join(self.val_data_path, "val.feather")) # tập validation

        val_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0)
        s, s2, s3, info = val_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []

        print("high_level_agent->val_cluster(): s ", s)
        print("high_level_agent->val_cluster(): s2 ", s2)
        print("high_level_agent->val_cluster(): s3 ", s3)

        while not done:
            a = self.act_test(s, s2, s3, info) # luôn chọn argmax Q, không epsilon-greedy
            s_, s2_, s3_, r, done, info_ = val_env.step(a) # environment chạy step => next_state, reward, done, etc
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_ # Cập nhật s, s2, s3, info
            action_list_episode.append(a)

            print("high_level_agent->val_cluster()->while: a ", a, s_, s2_, s3_, r)


        portfit_magine, final_balance, required_money, commission_fee = val_env.get_final_return_rate(
            slient=True) # lấy lãi/lỗ cuối

        final_balance = val_env.final_balance

        print("high_level_agent->val_cluster(): portfit_magine ", portfit_magine)
        print("high_level_agent->val_cluster(): final_balance ", final_balance)
        print("high_level_agent->val_cluster(): required_money ", required_money)
        print("high_level_agent->val_cluster(): commission_fee ", commission_fee)

        action_list.append(action_list_episode) # lưu thành NumPy array
        reward_list.append(reward_list_episode)
        final_balance_list.append(final_balance)
        required_money_list.append(required_money)
        commission_fee_list.append(commission_fee)

        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)

        np.save(os.path.join(save_path, "action_val.npy"), action_list)
        np.save(os.path.join(save_path, "reward_val.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance_val.npy"),
            final_balance_list)
        np.save(os.path.join(save_path, "require_money_val.npy"),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val.npy"),
                commission_fee_list)

        return_rate = final_balance / required_money

        print("high_level_agent->val_cluster(): return_rate ", return_rate)

        return return_rate # dùng so sánh với best_return_rate

    def test_cluster(self, epoch_path, save_path): # hàm này giúp đánh giá mô hình (val hoặc test), luôn dùng act_test (argmax Q), không update DQN, chỉ ghi lại kết quả
        self.hyperagent.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        self.hyperagent.eval()
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = pd.read_feather(
            os.path.join(self.test_data_path, "test.feather")) # tập test
        
        test_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0)
        s, s2, s3, info = test_env.reset()

        done = False
        action_list_episode = []
        reward_list_episode = []

        print("high_level_agent->test_cluster(): s ", s)
        print("high_level_agent->test_cluster(): s2 ", s2)
        print("high_level_agent->test_cluster(): s3 ", s3)

        while not done:
            a = self.act_test(s, s2, s3, info)
            s_, s2_, s3_, r, done, info_ = test_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            action_list_episode.append(a)

            print("high_level_agent->test_cluster()->while: a ", a, s_, s2_, s3_, r, done)


        portfit_magine, final_balance, required_money, commission_fee = test_env.get_final_return_rate(
            slient=True)

        final_balance = test_env.final_balance

        action_list.append(action_list_episode)
        reward_list.append(reward_list_episode)
        final_balance_list.append(final_balance)
        required_money_list.append(required_money)
        commission_fee_list.append(commission_fee)

        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)

        print("high_level_agent->test_cluster(): action_list ", action_list)
        print("high_level_agent->test_cluster(): reward_list ", reward_list)
        print("high_level_agent->test_cluster(): final_balance_list ", final_balance_list)
        print("high_level_agent->test_cluster(): required_money_list ", required_money_list)
        print("high_level_agent->test_cluster(): commission_fee_list ", commission_fee_list)

        np.save(os.path.join(save_path, "action.npy"), action_list)
        np.save(os.path.join(save_path, "reward.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance.npy"),
            final_balance_list)
        np.save(os.path.join(save_path, "require_money.npy"),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history.npy"),
                commission_fee_list)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    agent = DQN(args)
    agent.train()
