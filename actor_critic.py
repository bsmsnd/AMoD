import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        ##### TODO ######
        ### Complete definition
        self.affine = nn.Linear(4, 128)

        self.action_layer = nn.Linear(128, 2)
        self.value_layer = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []


    def forward(self, x):
        ##### TODO ######
        ### Complete definition
        x = F.relu(self.affine(x))
        action_scores = self.action_layer(x)
        state_values = self.value_layer(x)
        return F.softmax(action_scores, dim=-1), state_values

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append((SavedAction(m.log_prob(action), state_value)))
    return action

def sample_episode():
    state, ep_reward = env.reset(), 0
    episode = []

    for t in range(1, 10000):  # Run for a max of 10k steps

        action = select_action(state)

        # Perform action
        next_state, reward, done, _ = env.step(action.item())
        model.rewards.append((reward))

        episode.append((state, action, reward))
        state = next_state

        ep_reward += reward

        if args.render:
            env.render()

        if done:
            break

    return episode, ep_reward

def compute_losses(episode):
    ####### TODO #######
    #### Compute the actor and critic losses
    actor_losses, critic_losses = [], []
    R = 0
    saved_actions = model.saved_actions
    returns = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        actor_losses.append((-log_prob * advantage))
        critic_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    actor_loss = torch.stack(actor_losses).sum()
    critic_loss = torch.stack(critic_losses).sum()
    del model.rewards[:]
    del model.saved_actions[:]
    return actor_loss, critic_loss

def main():
    running_reward = 10
    average_rewards = []
    for i_episode in count(1):

        episode, episode_reward = sample_episode()

        optimizer.zero_grad()

        actor_loss, critic_loss = compute_losses(episode)

        loss = actor_loss + critic_loss

        loss.backward()

        optimizer.step()

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        average_rewards.append(running_reward)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, episode_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, len(episode)))
            print(len(average_rewards))
            break

    time = [i for i in range(len(average_rewards))]
    plt.plot(time, average_rewards)
    plt.xlabel('training processes')
    plt.ylabel('average reward')
    plt.show()

if __name__ == '__main__':
    main()
