import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

#  Following Aleksandar Haber from: https://www.youtube.com/watch?v=ZXEVznd5kaM&t=881s

#  alpha - learning rate / step size
#  gamma - discount factor
#  epsilon - probability to take random action (following epsilon-greedy approach)
class Q_Net(nn.Module):
    def __init__(self, input_shape, output_size):
        super(Q_Net, self).__init__()
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

        self.device = 'cpu'
    
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

class QL():
    def __init__(self, env, lr, gamma, buffer_size=10000, epsilon=None, device='cpu'):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.obs_space = env.observation_space.shape
        self.act_space = env.action_space.n

        self.Q = Q_Net(env.observation_space.shape, env.action_space.n).to(device)
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
    def train(self, num_episodes, batch_size=32, start_training=1000, train_every=1):
        target_model = self.Q.copy()
        tau = 0.996
        optimiser = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        rewards = []
        losses = []
        loop = tqdm(range(num_episodes))
        for k in loop:

            loop.set_description(f"Episode {k+1}/{num_episodes}")
            if len(rewards) > 100:
                mean_reward = np.mean(rewards[-100:])
                mean_losses = np.mean(losses[-100:])
                loop.set_postfix({"Est. Reward": mean_reward, "Est. Loss": mean_losses})
            state = self.env.reset()[0]
            done = False
            ep_reward = 0
            loss = 0

            while not done:
                # collect experience
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
                action = self._select_action(state.unsqueeze(0), k)
                (next_state, reward, done, _, _) = self.env.step(action)
                if reward > 200:
                    done = True
                self.replay_buffer.push(state, action, reward, done)

                state = next_state
                ep_reward += reward

                if k > start_training and k % train_every == 0:
                    # Sample batch
                    state_b, action_b, next_state_b, reward_b, done_b = self.replay_buffer.sample(batch_size)

                    # Determine Q loss
                    with torch.no_grad():
                        target_rewards = reward_b + self.gamma * target_model(next_state_b).max(1)[0] * (1-done_b.float())

                    preds = self.Q(state_b)[range(batch_size), action_b]
                    loss = F.mse_loss(preds, target_rewards)

                    # Update Q Network
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    loss = loss.item()

                    with torch.no_grad():
                        # Update target Network
                        for target_param, param in zip(target_model.parameters(), self.Q.parameters()):
                            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

            
            rewards.append(ep_reward)
            losses.append(loss)
        
        return rewards

    def policy(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.Q(state).argmax().item()