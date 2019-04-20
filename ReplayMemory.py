from collections import namedtuple
import random
import torch

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))
Transition = namedtuple('Transition',
                        ('open_req_last', 'num_veh_last', 'his_req_last',
                         'open_req_global_last', 'num_veh_global_last', 'his_req_global_last',
                         'action',
                         'open_req_new', 'num_veh_new', 'his_req_new',
                         'open_req_global_new', 'num_veh_global_new', 'his_req_global_new',
                         'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def Memory_dataProcess(s, s_global, a, s_new, s_new_global, r, memory):
    # s: state list
    # a: action scalar
    # s_new: next state list
    # r: reward scalar
    # memory: class ReplayMemory
    open_req_last = torch.tensor(s[0], dtype=torch.float)
    num_veh_last = torch.tensor(s[2], dtype=torch.float)
    his_req_last = torch.tensor(s[1], dtype=torch.float).transpose(0, 1).view(-1, 3, 3)
    open_req_global_last = torch.tensor(s_global[0], dtype=torch.float)
    num_veh_global_last = torch.tensor(s_global[2], dtype=torch.float)
    his_req_global_last = torch.tensor(s_global[1], dtype=torch.float).transpose(0, 1)

    open_req_new = torch.tensor(s_new[0], dtype=torch.float)
    num_veh_new = torch.tensor(s_new[2], dtype=torch.float)
    his_req_new = torch.tensor(s_new[1], dtype=torch.float).transpose(0, 1).view(-1, 3, 3)
    open_req_global_new = torch.tensor(s_new_global[0], dtype=torch.float)
    num_veh_global_new = torch.tensor(s_new_global[2], dtype=torch.float)
    his_req_global_new = torch.tensor(s_new_global[1], dtype=torch.float).transpose(0, 1)

    action = a.view(1, 1)
    reward = torch.tensor(r, dtype=torch.float).view(1, 1)
    memory.push(open_req_last, num_veh_last, his_req_last,
                open_req_global_last, num_veh_global_last, his_req_global_last,
                action,
                open_req_new, num_veh_new, his_req_new,
                open_req_global_new, num_veh_global_new, his_req_global_new,
                reward)