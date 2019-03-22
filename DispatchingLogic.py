import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from utils.RoboTaxiStatus import RoboTaxiStatus
from ReplayMemory import ReplayMemory
from Dqn import DQN
from collections import namedtuple

BATCH_SIZE = 128  # Some hyper-parameters to adjust!
memory = ReplayMemory(10000)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DispatchingLogic:
    """
    dispatching logic in the AidoGuest demo to compute dispatching instructions that are forwarded to the AidoHost
    """

    def __init__(self, bottomLeft, topRight):
        """
        :param bottomLeft: {lngMin, latMin}
        :param topRight: {lngMax, latMax}
        """
        self.lngMin = bottomLeft[0]
        self.lngMax = topRight[0]
        self.latMin = bottomLeft[1]
        self.latMax = topRight[1]

        print("minimum longitude in network: ", self.lngMin)
        print("maximum longitude in network: ", self.lngMax)
        print("minimum latitude  in network: ", self.latMin)
        print("maximum latitude  in network: ", self.latMax)

        # Example:
        # minimum longitude in network: -71.38020297181387
        # maximum longitude in network: -70.44406349551404
        # minimum latitude in network: -33.869660953686626
        # maximum latitude in network: -33.0303523690584

        self.matchedReq = set()
        self.matchedTax = set()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.last_state = None
        self.policy_net = DQN(bottomLeft, topRight).to(self.device)
        self.target_net = DQN(bottomLeft, topRight).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())


    def of(self, status):
        # This function returns the commands to vehicles given status
        # Return format: [pickup, rebalance]
        # pickup: a list of pickup commands [ [# vehicle, # request],...]
        # Rebalance: a list of rebalance commands: [ [# vehicle, rebalance_to], ...]

        pickup = None
        rebalance = None

        # Pre-process the data
        pre_processed = None
        state, action, next_state = self.last_state, None, None

        # Calculate the reward --> KEY!
        reward = self.reward_compute(state, next_state)

        # Push this new status
        memory.push(state, action, next_state, reward)

        # Optimize the network
        self.optimize_model()

        # get best action for this step
        dqn_output = self.policy_net(state)

        # Analyze dqn_output then construct commands

        # Prepare for next cycle
        self.last_state = next_state

        return [pickup, rebalance]

    def data_processing(self, status):
        # This function processes the data so that we can use it for further
        # Expected to do: Grid, # of requests within the grid, Poisson Distribution with parameter lambda, ...
        pass

    def optimize_model(self, GAMMA=0.999):
        # this function trains the model with decay factor GAMMA
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def reward_compute(self, last_state, state):
        # this function computes the reward given the last state and current state
        reward = None
        return reward