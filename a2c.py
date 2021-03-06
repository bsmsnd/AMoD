import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class A2C(nn.Module):

    def __init__(self, n_features, n_actions):
        super(A2C, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(16*3*3, 9)
        self.affine = nn.Linear(n_features, 128)
        self.relu = nn.ReLU()
        self.action_layer = nn.Linear(128, n_actions)
        self.value_layer = nn.Linear(128, 1)
        
        self.saved_actions = []
        self.rewards = []


    def forward(self, x, y, z):
       
        out1 = self.conv(z)
        out2 = self.fc(out1.view(x.size(0), -1))
        out3 = torch.cat((x, y, out2), 1)

        out4 = self.relu(self.affine(out3))
        action_scores = self.action_layer(out4)
        state_values = self.value_layer(out4)

        return F.softmax(action_scores, dim=-1), state_values

# model = A2C()
# optimizer = optim.Adam(model.parameters(), lr=3e-2)
# eps = np.finfo(np.float32).eps.item()


def a2c_select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append((SavedAction(m.log_prob(action), state_value)))
    return action

def a2c_sample_episode():
    
    # DON'T KNOW HOW TO DEAL WITH "env"
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

def a2c_compute_losses(episode):
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

def saveweight(model, path):
    # model: the network model
    # path: the path where the weights save
    torch.save(model.state_dict(), path)


def loadweight(model, path):
    # model: the new defined network model
    # path: the path where the weights save
    model.load_state_dict(torch.load(path))
    return model

'''
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
'''