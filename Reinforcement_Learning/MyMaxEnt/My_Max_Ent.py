import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class S_Net(nn.Module):
    def __init__(self, obs_shape, act_size):
        super().__init__()
        if len(obs_shape) == 1:
            obs_size = obs_shape[0]
            self.flatten = False
        else:
            obs_size = np.prod(obs_shape)
            self.flatten = True

        self.obs_shape = obs_shape
        self.obs_size = obs_size
        self.act_size = act_size

        self.fc1 = nn.Linear(obs_size + act_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, obs_size)
        self.logVar = nn.Linear(128, obs_size)
        self.device = torch.device('cpu')
    
    def to(self, device):
        self.device = device
        return super().to(device)
    
    def reparameterise(self, mu, logVar):
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, s, a, return_logVar=False):
        if self.flatten:
            s = s.view(-1, self.input_size)
        a = F.one_hot(a, self.act_size).float()
        x = torch.cat([s, a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu, logVar = self.mu(x), self.logVar(x)
        s = self.reparameterise(mu, logVar)

        if return_logVar:
            return s, logVar
        else:
            return s
        
    def copy(self):
        target_model = S_Net(self.obs_shape, self.act_size).to(self.device)
        target_model.load_state_dict(self.state_dict())
        return target_model


class Q_Net(nn.Module):
    def __init__(self, input_shape, output_size):
        super().__init__()
        self.input_shape = input_shape
        if len(input_shape) == 1:
            input_size = input_shape[0]
            self.flatten = False
        else:
            input_size = np.prod(input_shape)
            self.flatten = True
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.device = torch.device('cpu')
    
    def to(self, device):
        self.device = device
        return super().to(device)
    
    def forward(self, x):
        if self.flatten:
            x = x.view(-1, self.input_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def copy(self):
        target_model = Q_Net(self.input_shape, self.fc3.out_features).to(self.device)
        target_model.load_state_dict(self.state_dict())
        return target_model


class ReplayBuffer():
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.states = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.position = 0
        self.full = False
        self.device = device
    
    def push(self, state, action, reward, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.position = self.position + 1
        if self.position == self.capacity:
            self.position = 0
            self.full = True
    
    def sample(self, batch_size):
        if not self.full and batch_size > self.position:
            raise ValueError("Batch size is greater than current buffer size")

        max_idx = self.position-1 if not self.full else self.capacity
        idx = np.random.choice(max_idx, batch_size, replace=False)
        return self.states[idx], self.actions[idx], self.states[(idx+1)%self.capacity], self.rewards[idx], self.dones[idx]

class MaxEnt():
    def __init__(self, env, lr, gamma, buffer_size=10000, epsilon=None, device='cpu'):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.obs_space = env.observation_space.shape
        self.act_space = env.action_space.n

        self.Q = Q_Net(env.observation_space.shape, env.action_space.n).to(device)
        self.S = S_Net(env.observation_space.shape, env.action_space.n).to(device)
        self.replay_buffer = ReplayBuffer(buffer_size, env.observation_space.shape, device)
    
    #  Select action following epsilon-greedy policy
    def _select_action(self, state, k):
        if self.epsilon is None:
            if k == 0:
                k = 1
            epsilon =  1/k
        else:
            epsilon = self.epsilon
        rand = np.random.uniform(size=1)

        if rand <= epsilon:
            return np.random.randint(0, self.env.action_space.n)
        
        return self.Q(state).argmax().item()

    #  S - state, A - action, R - reward, S - state_1, A - action_1
    def train(self, num_episodes, batch_size=32, ent_scale=1.0, start_training=1000, train_every=1, done_reward=None):
        target_Q = self.Q.copy()
        target_S = self.S.copy()
        tau = 0.996
        # optimiser = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        optimiser = torch.optim.AdamW(list(self.Q.parameters()) + list(self.S.parameters()), lr=self.lr)
        rewards = []
        Q_losses = []
        S_losses = []
        loop = tqdm(range(num_episodes))

        for k in loop:

            loop.set_description(f"Episode {k+1}/{num_episodes}")
            if len(rewards) > 100:
                mean_reward = np.mean(rewards[-100:])
                mean_Q_losses = np.mean(Q_losses[-100:])
                mean_S_losses = np.mean(S_losses[-100:])
                loop.set_postfix({"Est. Reward": mean_reward, "Est. Q Loss": mean_Q_losses, "Est. S Loss": mean_S_losses})
            state = self.env.reset()[0]
            done = False
            ep_reward = 0
            Q_loss = 0
            S_loss = 0

            while not done:
                # collect experience
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
                action = self._select_action(state.unsqueeze(0), k)
                (next_state, reward, done, truncated, _) = self.env.step(action)
                if done and done_reward is not None:
                    reward += done_reward
                done = done or truncated
                self.replay_buffer.push(state, action, reward, done)

                state = next_state
                ep_reward += reward

                if k > start_training and k % train_every == 0:
                    # Sample batch
                    state_b, action_b, next_state_b, reward_b, done_b = self.replay_buffer.sample(batch_size)

                    # Determine S loss, minimise prediction error and maximise entropy
                    state_pred = self.S(state_b, action_b)
                    S_loss = F.mse_loss(state_pred, next_state_b) * 0.001

                    # Determine Q loss
                    with torch.no_grad():
                        target_rewards = target_Q(next_state_b).max(1)[0]
                        entropy = target_S(state_b, action_b, return_logVar=True)[1].sum(dim=1)
                        target = (reward_b + ent_scale * entropy) + self.gamma * target_rewards * (1-done_b.float())
                    preds = self.Q(state_b)[range(batch_size), action_b]
                    Q_loss = F.mse_loss(preds, target)

                    # Update Q and S Networks
                    loss = Q_loss + S_loss
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    Q_loss = Q_loss.item()
                    S_loss = S_loss.item()

                    with torch.no_grad():
                        # Update target Q Network
                        for target_param, param in zip(target_Q.parameters(), self.Q.parameters()):
                            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
                        # Update target S Network
                        for target_param, param in zip(target_S.parameters(), self.S.parameters()):
                            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
            
            rewards.append(ep_reward)
            Q_losses.append(Q_loss)
            S_losses.append(S_loss)
        
        return rewards

    def policy(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.Q(state).argmax().item()