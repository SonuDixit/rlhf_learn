import gymnasium as gym
import numpy as np
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.distributions import Normal
import logging

# env = gym.make("BipedalWalker-v3", render_mode="human", hardcore=True)
# env = gym.make("BipedalWalker-v3", hardcore=True)
# observation, info = env.reset(seed=42)

max_time_step = 2000 # https://gymnasium.farama.org/environments/box2d/bipedal_walker/#description
n_trial = 100

logging.basicConfig(level=logging.INFO,  
                    filename='train.log',  # Logs will be written to this file. If not specified, logs are printed to stderr.  
                    filemode='a',  # 'a' for append, 'w' for overwrite  
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  

def test_random_policy():
    env = gym.make("BipedalWalker-v3", render_mode="human", hardcore=True)
    # env = gym.make("BipedalWalker-v3", hardcore=True)
    observation, info = env.reset(seed=42)
    rewards = []
    for trial in range(n_trial):
        rew = 0
        for _ in range(max_time_step):
            action = env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)
            rew += reward
            if terminated or truncated:
                observation, info = env.reset()
                rewards.append(rew)
                break

    env.close()
    return sum(rewards)/len(rewards)

# Define the policy network  
class PolicyNet(nn.Module):  
    def __init__(self):  
        super(PolicyNet, self).__init__()  
        self.fc = nn.Sequential(  
            nn.Linear(24, 64),  # 24 is the size of the observation space  
            nn.ReLU(),
            nn.Linear(64, 64),  # 24 is the size of the observation space  
            nn.ReLU(),  
            nn.Linear(64, 4),  # 4 is the size of the action space  
            nn.Tanh()  # BipedalWalker actions are in the range [-1, 1]  
        )  
  
    def forward(self, x):  
        return self.fc(x)  
  
# Function to select an action and calculate its log probability  
def select_action(state, sample=True):  
    state = torch.Tensor(state).unsqueeze(0) 
    action_mean = policy_net(state)
    # print(f'action mean: {action_mean}')
    # Assuming a fixed variance of 1.0   
    cov = torch.diag(torch.ones(action_mean.size()))  
    dist = Normal(action_mean, cov)
    
    if sample:  
        action = dist.sample()
        # print(f'sampled action:{action}')
        action = torch.clamp(action, min=-1.0+1e-5, max = 1.0-1e-5)
    else:
        action = action_mean
    
    log_prob = dist.log_prob(action)
    # print(f'log_prob:{log_prob}')  
    return action.detach().numpy(), log_prob   
  
# Training hyperparameters  
learning_rate = 3e-3  
gamma = 0.99  # Discount factor for future rewards  
  
# Setup environment and policy network  
env = gym.make("BipedalWalker-v3")  
policy_net = PolicyNet()  
optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate)  

# Function to compute the returns by summing rewards  
def compute_returns(rewards, gamma=0.99):  
    R = 0  
    returns = []  
    for r in rewards[::-1]:  
        R = r + gamma * R  
        returns.insert(0, R)  
    return returns  

@torch.no_grad()
def val(n_episode = 20):
      
    # saved_log_probs = []  
    avg_rew = []
    for eps in range(n_episode):
        rewards = []  
        state, _ = env.reset()
        for _ in range(max_time_step):  # limit the number of steps per episode 
            with torch.no_grad(): 
                action, log_prob = select_action(state, sample=False)  
            state, reward, terminated, truncated, info = env.step(action[0])  
            # saved_log_probs.append(log_prob)  
            rewards.append(reward)  
            if terminated or truncated:  
                break
        avg_rew.append(np.sum(rewards))
    return np.mean(avg_rew)


def print_policy_net_params():
    print("Policy Network Parameters:\n")
    
    # Iterate through the named parameters of the network
    for name, param in policy_net.named_parameters():
        print(f"Parameter: {name}")
        print(f" - Size: {param.size()}")
        print(f" - Values: \n{param.data}\n")


# Train the policy gradient  
def train_pg(episodes=1000):
    # print_policy_net_params()  
    for episode in range(episodes):  
        state, _ = env.reset()  
        saved_log_probs = []  
        rewards = []  
  
        for _ in range(max_time_step):  # limit the number of steps per episode  
            action, log_prob = select_action(state)
            # print(f'log_prob:{log_prob}')  
            state, reward, terminated, truncated, info = env.step(action[0])  
            saved_log_probs.append(log_prob)  
            rewards.append(reward)  
            if terminated or truncated:  
                break  
  
        # returns = compute_returns(rewards, gamma)  
        # returns = torch.tensor(returns)  
        # returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # Normalize  
  
        policy_loss = []  
        for log_prob, R in zip(saved_log_probs, rewards):  
            policy_loss.append(-log_prob * R)
        # print(f'policy loss:{policy_loss}')  
        policy_loss = torch.stack(policy_loss).sum()
        # print(f'episode:{episode}, loss:{policy_loss}')  
        logging.info(f'episode:{episode}, train_loss:{policy_loss:.2f}')
        optimizer.zero_grad()  
        policy_loss.backward()  
        optimizer.step()
        # print_policy_net_params()
        # exit() 
  
        if episode % 100 == 0: 
            avg_val_reward = val() 
            # print('episode:{}\t avg_val_reward: {:.2f}'.format(episode, avg_val_reward))
            logging.info('episode:{}, avg_val_reward:{:.2f}'.format(episode, avg_val_reward))  

if __name__ == "__main__":  
    train_pg(10000)  
