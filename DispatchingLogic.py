import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from utils.RoboTaxiStatus import RoboTaxiStatus
from ReplayMemory import ReplayMemory
from Dqn import DQN
from collections import namedtuple
from Vehicle import Vehicle
from constant import *
from generic import *


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

        self.history_requests = []
        self.numRequest = 0

        self.keep_history_time = 1800
        self.time = 0



        # Assume that coordination will be converted to distances in miles
        self.unitLongitude = (self.lngMax - self.lngMin) / GRAPHMAXCOORDINATE
        self.unitLatitude = (self.latMax - self.latMin) / GRAPHMAXCOORDINATE

        self.fleet = [Vehicle() for _ in range(NUMBER_OF_VEHICLES)]

        # Requests
        self.responded_requests = []



    def of(self, status):
        # This function returns the commands to vehicles given status
        # Return format: [pickup, rebalance]
        # pickup: a list of pickup commands [ [# vehicle, # request],...]
        # Rebalance: a list of rebalance commands: [ [# vehicle, rebalance_to], ...]

        pickup = []
        rebalance = []

        # Pre-process the data
        num_vehicles_in_area, distance_to_each_area, request_distribution, open_requests = self.data_processing(status)
        state, action, next_state = self.last_state, None, None

        for request in open_requests:
            vehicles_to_request_distance = [1e8 for i in range(NUMBER_OF_VEHICLES)]
            for i in range(NUMBER_OF_VEHICLES):
                if self.fleet[i].status == STAY or self.fleet[i].status == REBALANCE:
                    vehicles_to_request_distance[i] = self.fleet[i].get_distance_to(request[1][0], request[1][1])
            choose_car = arg_min(vehicles_to_request_distance)
            if vehicles_to_request_distance[choose_car] > 1e7:  # if no car available
                pass
            else:
                pickup.append([choose_car, request[0]])
                self.responded_requests.append(request[0])

        # Calculate the reward --> KEY!
        reward = self.reward_compute(state, next_state)

        # Push this new status
        memory.push(state, action, next_state, reward)

        # Optimize the network
        self.optimize_model()

        # get best action for this step
        # input: all available vehicles including STAY
        available_vehicles = []
        dqn_output = self.policy_net(state)

        # Analyze dqn_output then construct commands
        best_actions = torch.argmax(dqn_output,dim=1)
        for i in best_actions.shape[0]:
            if self.fleet[available_vehicles[i]].area == best_actions[i]: # should stay
                pass
            else:
                self.fleet[available_vehicles[i]].rebalanceStartTime = self.time
                self.fleet[available_vehicles[i]].rebalanceTo = best_actions[i]
                self.fleet[available_vehicles[i]].data[0] = state[available_vehicles[i]]
                self.fleet[available_vehicles[i]].data[1] = MID_POINTS[best_actions[i]]
                rebalance.append([available_vehicles[i], self.coordinate_change('TO_COMMAND', MID_POINTS[best_actions[i]])])

        # Prepare for next cycle
        self.last_state = next_state

        # Commmand Finalize: coordination change,


        return [pickup, rebalance]

    def data_processing(self, status):
        # This function processes the data so that we can use it for further
        # Expected to do: Grid, # of requests within the grid, Poisson Distribution with parameter lambda, ...

        self.time = status[0]

        # coordination change and update vehicle information
        num_vehicles_in_area = [0 for _ in range(MAP_DIVIDE ** 2)]
        distance_to_each_area = [[0. for _ in range(MAP_DIVIDE ** 2)] for __ in range(NUMBER_OF_VEHICLES)]

        for i in range(NUMBER_OF_VEHICLES):
            loc = self.coordinate_change('TO_MODEL', status[1][i][1])
            status = status[1][i][2]
            self.fleet[i].update(loc, status, self.time)
            if self.fleet[i].status is STAY or self.fleet[i].status is REBALANCE:
                num_vehicles_in_area[self.fleet[i].area] += 1
            for j in range(MAP_DIVIDE ** 2):
                distance_to_each_area[i][j] = self.fleet[i].get_distance_to(MID_POINTS[j])

        # Process Requests
        # Process Request Distribution & open requests
        open_requests = []
        # add
        for request in status[2]:
            if request[0] < self.numRequest:
                flag = False
                for responded in self.responded_requests:
                    if request[0] == responded:
                        flag = True
                        break
                if flag == False:
                    open_requests.append([request[0], request[2]])
                pass
            else:
                self.history_requests.append([request[1], which_area(request[2][0], request[2][1])])
                self.numRequest = request[0]
                open_requests.append([request[0], request[2]])

        while self.history_requests[0][0] < self.time - self.keep_history_time:
            self.history_requests.pop(0)

        request_distribution = [0 for _ in range(MAP_DIVIDE ** 2)]
        for his_request in self.history_requests:
            request_distribution[his_request[1]] += 1
        # update s', r
        for i in range(NUMBER_OF_VEHICLES):
            if self.fleet[i] == 1:
                self.fleet[i].data[2] = [] # match the variable state
                self.fleet[i].data[3] = reward
                memory.push(*tuple(self.fleet[i].data))

        # remove
        return num_vehicles_in_area, distance_to_each_area, request_distribution, open_requests

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

    def coordinate_change(self, direction, loc):
        if direction == 'TO_MODEL':
            return [ (loc[0] - self.lngMin) / self.unitLatitude, (loc[1] - self.latMin) / self.unitLatitude]
        elif direction == 'TO_COMMAND':
            return [loc[0] * self.unitLongitude + self.lngMin, loc[1] * self.unitLatitude + self.latMin]
        else:
            raise ValueError
