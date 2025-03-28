import pathlib
import sys
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
import warnings


warnings.filterwarnings("ignore")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from MacroHFT.env.low_level_env import Testing_Env, Training_Env
from MacroHFT.model.net import *
from MacroHFT.RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from MacroHFT.RL.util.replay_buffer import ReplayBuffer

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",type=int,default=1000000,) # change
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
parser.add_argument("--label",type=str,default="label_1")
parser.add_argument("--clf",type=str,default="slope")
parser.add_argument("--alpha",type=float,default="0")
parser.add_argument("--device",type=str,default="cuda:0")
parser.add_argument("--use_lstm_predictions", type=bool, default=True, 
                   help="Whether to use LSTM predictions as observations")
parser.add_argument("--hidden_size", type=int, default=64, 
                   help="Hidden size for LSTM model")
parser.add_argument("--num_layers", type=int, default=2, 
                   help="Number of layers for LSTM model")


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def calculate_alpha(diff, k):
    alpha = 16 * (1 - torch.exp(-k * diff))
    return torch.clip(alpha, 0, 16)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Only use the last output
        return out

class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
        self.result_path = os.path.join("./result/low_level", 
                                        '{}'.format(args.dataset), '{}'.format(args.clf), str(int(args.alpha)), args.label)
        self.label = int(args.label.split('_')[1])
        self.model_path = os.path.join(self.result_path,
                                       "seed_{}".format(self.seed))
        self.train_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "train")
        self.val_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "val")
        self.test_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "test")
        if args.clf == 'slope':
            with open(os.path.join(self.train_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.train_index = pickle.load(file)
            with open(os.path.join(self.val_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.val_index = pickle.load(file)
            with open(os.path.join(self.test_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.test_index = pickle.load(file)
        elif args.clf == 'vol':
            with open(os.path.join(self.train_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.train_index = pickle.load(file)
            with open(os.path.join(self.val_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.val_index = pickle.load(file)
            with open(os.path.join(self.test_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.test_index = pickle.load(file)


        self.dataset=args.dataset
        self.clf = args.clf
        if "BTC" in self.dataset:
            self.max_holding_number=0.01 # BTC có giá cao nên 0.01 BTC cũng có giá trị tương đối lớn
        elif "ETH" in self.dataset:
            self.max_holding_number=0.2
        elif "DOT" in self.dataset:
            self.max_holding_number=10
        elif "LTC" in self.dataset:
            self.max_holding_number=10
        else:
            raise Exception ("we do not support other dataset yet")
        self.epoch_number = args.epoch_number
        
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.update_counter = 0
        self.q_value_memorize_freq = args.q_value_memorize_freq

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()

        self.transcation_cost = args.transcation_cost # phí giao dịch (hoặc tỷ lệ chi phí) mà agent phải trả mỗi lần mua/bán
        self.back_time_length = args.back_time_length # độ dài window mà agent nhìn lại để lấy state
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)

        self.eval_net, self.target_net = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device), subagent(
                self.n_state_1, self.n_state_2, self.n_action,
                64).to(self.device) ## hàm to ở đâu ra ???

        self.hardupdate() # để đồng bộ target_net = eval_net

        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), # các tham số huấn luyện
                                          lr=args.lr)
        self.loss_func = nn.MSELoss() # hàm mất mát (loss function) dùng để so sánh Q-value dự đoán (từ eval_net) với Q-target (giá trị mong đợi)
        self.batch_size = args.batch_size
        self.gamma = args.gamma # Discount factor trong học tăng cường, xác định mức độ ưu tiên reward tương lai so với reward hiện tại
        self.tau = args.tau # Hệ số dùng cho soft update giữa eval_net và target_net
        self.n_step = args.n_step # Số bước (n-step) để tính return trong Q-learning
        self.eval_update_freq = args.eval_update_freq
        self.buffer_size = args.buffer_size
        self.epsilon_start = args.epsilon_start # Giá trị ϵ ban đầu cho epsilon-greedy (tỉ lệ "random")
        self.epsilon_end = args.epsilon_end # Giá trị ϵ cuối cùng khi quá trình huấn luyện kết thúc
        self.decay_length = args.decay_length # Số epoch (hoặc số bước) để ϵ giảm dần từ epsilon_start về epsilon_end
        self.epsilon_scheduler = LinearDecaySchedule(start_epsilon=self.epsilon_start, end_epsilon=self.epsilon_end, decay_length=self.decay_length)
        self.epsilon = args.epsilon_start

        # Load LSTM model if specified
        self.lstm_model = None
        if args.use_lstm_predictions:
            print("Loading LSTM model...")
            try:
                # Load the pre-trained LSTM model directly from LSTM4.ipynb
                lstm_model_path = os.path.join("result", "low_level", "ETHUSDT", "best_model", "lstm", 
                                             str(int(0)), "model.pth")
                # Create a new LSTM model instance
                self.lstm_model = LSTM(
                    input_size=5,  # open, close, high, low, volume
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers,
                    output_size=1
                ).to(self.device)
                # Load the state dict into the model
                self.lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=self.device))
                self.lstm_model.eval()
                print(f"Successfully loaded LSTM model from {lstm_model_path}")
            except Exception as e:
                print(f"Error loading LSTM model: {e}")
                print("Training without LSTM predictions")
                self.lstm_model = None

    def update(self, replay_buffer):
        self.eval_net.train()
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        a_argmax = self.eval_net(batch['next_state'], batch['next_state_trend'], batch['next_previous_action']).argmax(dim=-1, keepdim=True)
        q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state'], batch['next_state_trend'], 
                                                    batch['next_previous_action']).gather(-1, a_argmax).squeeze(-1) # Double DQN: chọn action argmax từ eval_net, lấy Q-value từ target_net

        # print("low_level_agent->update(): a_argmax ", a_argmax)
        # print("low_level_agent->update(): q_target ", q_target)

        q_distribution = self.eval_net(batch['state'], batch['state_trend'], batch['previous_action'])
        q_current = q_distribution.gather(-1, batch['action']).squeeze(-1) # Lấy Q-value của action đã thực hiện

        # print("low_level_agent->update(): q_distribution ", q_distribution)
        # print("low_level_agent->update(): q_current ", q_current)

        td_error = self.loss_func(q_current, q_target)

        demonstration = batch['demo_action']
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )

        alpha = args.alpha
        loss = td_error + alpha * KL_loss
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1)
        self.optimizer.step()
        for param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()): # Soft update target_net
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.update_counter += 1
        self.eval_net.eval()
        return td_error.cpu(), KL_loss.cpu(), torch.mean(
            q_current.cpu()), torch.mean(q_target.cpu())

    def hardupdate(self):
        self.target_net.load_state_dict(self.eval_net.state_dict()) # Đồng bộ hoàn toàn target_net = eval_net

    def act(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        
        # Get LSTM prediction if available in info
        lstm_prediction = info.get("lstm_prediction")
        if lstm_prediction is not None:
            lstm_prediction = torch.tensor(lstm_prediction, dtype=torch.float32).to(self.device)

        print("low_level_agent->act(): previous_action ", previous_action)
        print("low_level_agent->act(): lstm_prediction ", lstm_prediction)

        if np.random.uniform() < (1-self.epsilon): # Epsilon-greedy: với xác suất (1 - epsilon), chọn argmax(Q), ngược lại random {0,1}
            actions_value = self.eval_net(x1, x2, previous_action, lstm_prediction)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            print("low_level_agent->act(): action ", action)
        else:
            action_choice = [0,1]
            action = random.choice(action_choice)
            print("low_level_agent->act(): random action ", action)

        return action

    def act_test(self, state, state_trend, info): # Luôn chọn argmax(Q), không random. Dùng khi validation/test
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long(), 0).to(self.device)
        
        # Get LSTM prediction if available in info
        lstm_prediction = info.get("lstm_prediction")
        if lstm_prediction is not None:
            lstm_prediction = torch.tensor(lstm_prediction, dtype=torch.float32).to(self.device)

        print("low_level_agent->act_test(): lstm_prediction ", lstm_prediction)
        
        actions_value = self.eval_net(x1, x2, previous_action, lstm_prediction)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]

        print("low_level_agent->act_test(): action, action_value ", action, actions_value)

        return action

    def train(self, lstm_model=None):
        # Use provided LSTM model or the one loaded during initialization
        if lstm_model is not None:
            self.lstm_model = lstm_model
            
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        df_list = self.train_index[self.label]
        df_number=int(len(df_list)) # số chunk
        step_counter = 0 # đếm tổng số bước (step) để quyết định khi nào update DQN
        episode_counter = 0 # đếm số episode đã chạy
        epoch_counter = 0 # đếm số epoch đã chạy
        self.replay_buffer = ReplayBuffer(args, self.n_state_1, self.n_state_2, self.n_action) # ReplayBuffer để lưu transition (state, action, reward, next_state,...)
        best_return_rate = -float('inf') # để lưu "tỷ suất lợi nhuận" (return_rate) tốt nhất
        best_model = None # lưu state_dict của mô hình tốt nhất

        for sample in range(self.epoch_number): # duyệt qua số epoch
            print('low_level_agent->train(): epoch ', epoch_counter + 1)

            random_list = self.train_index[self.label] # lấy danh sách chunk index. Shuffle chúng
            random.shuffle(random_list)
            random_position_list = random.choices(range(self.n_action), k=df_number) # random initial action (0 hoặc 1) cho từng chunk
            
            print('low_level_agent->train(): random_list ', random_list)

            for i in range(df_number):
                df_index = random_list[i]
    
                print("low_level_agent->train(): training with df", df_index)

                self.df = pd.read_feather(
                    os.path.join(self.train_data_path, "df_{}.feather".format(df_index)))
                self.eval_net.eval() # đặt mạng ở chế độ evaluation (tắt dropout,...)

                train_env = Training_Env(
                        df=self.df,
                        tech_indicator_list=self.tech_indicator_list,
                        tech_indicator_list_trend=self.tech_indicator_list_trend,
                        transcation_cost=self.transcation_cost,
                        back_time_length=self.back_time_length,
                        max_holding_number=self.max_holding_number,
                        initial_action=random_position_list[i],
                        alpha = 0,
                        lstm_model=self.lstm_model,  # Pass LSTM model to environment
                        device=self.device      # Pass device to environment)
                )
                s, s2, info = train_env.reset()
                episode_reward_sum = 0 # để cộng dồn reward của mỗi episode

                print("low_level_agent->train(): train_env.s ", s)
                print("low_level_agent->train(): train_env.s2 ", s2)

                while True:
                    a = self.act(s, s2, info)
                    s_, s2_, r, done, info_ = train_env.step(a) # Môi trường trả về (next_state, reward, done, info_)

                    self.replay_buffer.store_transition(s, s2, info['previous_action'], info['q_value'], a, r, s_, s2_, info_['previous_action'],
                                    info_['q_value'], done) # Lưu transition vào ReplayBuffer
                    episode_reward_sum += r

                    s, s2, info = s_, s2_, info_
                    step_counter += 1

                    print("low_level_agent->train()->while: train_env.step.s_ ", s_)
                    print("low_level_agent->train()->while: train_env.step.s2 ", s2)

                    if step_counter % self.eval_update_freq == 0 and step_counter > (
                            self.batch_size + self.n_step): # Mỗi khi step_counter % eval_update_freq == 0, gọi update(...) để train DQN
                        for i in range(self.update_times):
                            td_error, KL_loss, q_eval, q_target = self.update(self.replay_buffer)
                            if self.update_counter % self.q_value_memorize_freq == 1:
                                self.writer.add_scalar(
                                    tag="td_error",
                                    scalar_value=td_error,
                                    global_step=self.update_counter,
                                    walltime=None)
                                self.writer.add_scalar(
                                    tag="KL_loss",
                                    scalar_value=KL_loss,
                                    global_step=self.update_counter,
                                    walltime=None)
                                self.writer.add_scalar(
                                    tag="q_eval",
                                    scalar_value=q_eval,
                                    global_step=self.update_counter,
                                    walltime=None)
                                self.writer.add_scalar(
                                    tag="q_target",
                                    scalar_value=q_target,
                                    global_step=self.update_counter,
                                    walltime=None)
                    if done:
                        print("low_level_agent->train(): done ", done)
                        print("low_level_agent->train(): episode_reward_sum ", episode_reward_sum)
                        print("low_level_agent->train(): step_counter ", step_counter)

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

                epoch_return_rate_train_list.append(final_balance / (required_money)) # Lưu vào list để tính trung bình
                epoch_final_balance_train_list.append(final_balance)
                epoch_required_money_train_list.append(required_money)
                epoch_reward_sum_train_list.append(episode_reward_sum)

                print ("low_level_agent->train(): final_balance ", final_balance)
                print ("low_level_agent->train(): required_money ", required_money)
                print ("low_level_agent->train(): episode_reward_sum ", episode_reward_sum)

            epoch_counter += 1 # Mỗi epoch xong, cập nhật epsilon, tính trung bình, lưu model
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch_counter)
            mean_return_rate_train = np.mean(epoch_return_rate_train_list)
            mean_final_balance_train = np.mean(epoch_final_balance_train_list)
            mean_required_money_train = np.mean(epoch_required_money_train_list)
            mean_reward_sum_train = np.mean(epoch_reward_sum_train_list)

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

            print("low_level_agent->train(): epsilon ", self.epsilon)
            print("low_level_agent->train(): mean_return_rate_train ", mean_return_rate_train)
            print("low_level_agent->train(): mean_final_balance_train ", mean_final_balance_train)
            print("low_level_agent->train(): mean_required_money_train ", mean_required_money_train)
            print("low_level_agent->train(): mean_reward_sum_train ", mean_reward_sum_train)

            epoch_path = os.path.join(self.model_path,
                                        "epoch_{}".format(epoch_counter))
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)

            torch.save(self.eval_net.state_dict(),
                        os.path.join(epoch_path, "trained_model.pkl"))
            val_path = os.path.join(epoch_path, "val")
            if not os.path.exists(val_path):
                    os.makedirs(val_path)

            return_rate_0 = self.val_cluster(epoch_path, val_path, 0, self.lstm_model)
            return_rate_1 = self.val_cluster(epoch_path, val_path, 1, self.lstm_model)
            return_rate_eval = (return_rate_0 + return_rate_1) / 2

            print("low_level_agent->train(): return_rate_eval ", return_rate_eval)
            print("low_level_agent->train(): return_rate_0 ", return_rate_0)
            print("low_level_agent->train(): return_rate_1 ", return_rate_1)

            if return_rate_eval > best_return_rate:
                best_return_rate = return_rate_eval
                best_model = self.eval_net.state_dict()

                print("low_level_agent->train(): best_return_rate ", best_return_rate)
                print("best model updated to epoch ", epoch_counter)

            epoch_return_rate_train_list = []
            epoch_final_balance_train_list = []
            epoch_required_money_train_list = []
            epoch_reward_sum_train_list = []

        best_model_path = os.path.join("./result/low_level", 
                                        '{}'.format(self.dataset), '{}'.format(self.clf), str(self.label), 'best_model.pkl') # Cuối cùng, lưu best_model thành file best_model.pkl
        torch.save(best_model, best_model_path)

    def val_cluster(self, epoch_path, save_path, initial_action, lstm_model=None):
        # Use provided LSTM model or the one loaded during initialization
        if lstm_model is not None:
            self.lstm_model = lstm_model
            
        self.eval_net.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        self.eval_net.eval() # chế độ đánh giá (tắt dropout,...)

        df_list = self.val_index[self.label]
        df_number=int(len(df_list)) 
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []

        for i in range(df_number): # Duyệt qua từng chunk (df_i) trong val_index
            print("low_level_agent->val_cluster(): validating on df", df_list[i])
            self.df = pd.read_feather(
                os.path.join(self.val_data_path, "df_{}.feather".format(df_list[i])))

            val_env = Testing_Env(
                    df=self.df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    transcation_cost=self.transcation_cost,
                    back_time_length=self.back_time_length,
                    max_holding_number=self.max_holding_number,
                    initial_action=initial_action,
                    lstm_model=self.lstm_model,  # Pass LSTM model to environment
                    device=self.device      # Pass device to environment)
            )
            s, s2, info = val_env.reset()
            done = False
            action_list_episode = []
            reward_list_episode = []

            print("low_level_agent->val_cluster(): val_env.s ", s)
            print("low_level_agent->val_cluster(): val_env.s2 ", s2)

            while not done: # Chạy act_test(...) (argmax Q) đến khi done
                a = self.act_test(s, s2, info)

                s_, s2_, r, done, info_ = val_env.step(a)
                reward_list_episode.append(r)

                s, s2, info = s_, s2_, info_
                action_list_episode.append(a)
                
                print("low_level_agent->val_cluster()->while: val_env.step.a ", a)
                print("low_level_agent->val_cluster()->while: val_env.step.r ", r)

            portfit_magine, final_balance, required_money, commission_fee = val_env.get_final_return_rate(
                slient=True)
            final_balance = val_env.final_balance
            required_money = val_env.required_money
            action_list.append(action_list_episode)
            reward_list.append(reward_list_episode)
            final_balance_list.append(final_balance)
            required_money_list.append(required_money)
            commission_fee_list.append(commission_fee)

            print("low_level_agent->val_cluster(): final_balance ", final_balance)
            print("low_level_agent->val_cluster(): required_money ", required_money)
            print("low_level_agent->val_cluster(): commission_fee ", commission_fee)

        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)

        np.save(os.path.join(save_path, "action_val_{}.npy".format(initial_action)), action_list)
        np.save(os.path.join(save_path, "reward_val_{}.npy".format(initial_action)), reward_list)
        np.save(os.path.join(save_path, "final_balance_val_{}.npy".format(initial_action)),
            final_balance_list)
        np.save(os.path.join(save_path, "require_money_val_{}.npy".format(initial_action)),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val_{}.npy".format(initial_action)),
                commission_fee_list)

        return_rate_mean = np.nan_to_num(final_balance_list / required_money_list).mean() # Tính trung bình return_rate_mean (final_balance / required_money)
        np.save(os.path.join(save_path, "return_rate_mean_val_{}.npy".format(initial_action)),
                return_rate_mean)

        return return_rate_mean


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    
    agent = DQN(args)
    agent.train()
