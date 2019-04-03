from collections import namedtuple
import random

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))
Transition = namedtuple('Transition',
                        ('open_req_last', 'num_veh_last', 'his_req_last', 'action', 'open_req_new', 'num_veh_new',
                         'his_req_new', 'reward'))


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