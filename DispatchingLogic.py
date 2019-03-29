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
from distance_on_unit_sphere import *


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
        self.numRequestSeen = 0

        self.keep_history_time = 1800
        self.time = 0

        # Assume that coordination will be converted to distances in miles
        self.unitLongitude = (self.lngMax - self.lngMin) / GRAPHMAXCOORDINATE
        self.map_width = distance_on_unit_sphere(self.latMin, self.lngMin, self.latMin, self.lngMax)
        self.map_length = distance_on_unit_sphere(self.latMax, self.lngMax, self.latMin, self.lngMax)
        LNG_SCALE = GRAPHMAXCOORDINATE * self.map_length / self.map_width
        self.unitLatitude = (self.lngMax - self.lngMin) / LNG_SCALE

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
        # num_vehicles_in_area, distance_to_each_area, request_distribution, open_requests = self.data_preprocess(status)
        states, open_requests_info_in_area = self.data_preprocess(status)
        # States: [0] open requests in surroundings, [1] history req in surroundings
        #         [2] number vehicles, [3] label which car, [4] which area
        #TODO: CNN and DQN

        actions = None
        # The list output of DQNs, length of which is the same as that of states
        # Some possible values are defined as follows:
        # 0: STAY, 1-9: PICKUP at the relative region from topleft to bottomright
        # 10-18: REBALANCE to 1-9 regions from topleft to bottomright

        pickup_list = [[]] * (MAP_DIVIDE ** 2)
        for i, individual_state in enumerate(states):
            cmd = actions[i]
            if cmd >= 1 and cmd <= 9:
                to_area2D = [individual_state[4] // MAP_DIVIDE, individual_state[4] % MAP_DIVIDE] + NINE_REGIONS[cmd - 1]
                goto = to_area2D[0] * MAP_DIVIDE + to_area2D[1]
                pickup_list[goto].append(individual_state[3])
            if cmd > 9:
                to_which_area = [individual_state[4] // MAP_DIVIDE, individual_state[4] % MAP_DIVIDE] + NINE_REGIONS[cmd - 9 - 1]
                if self.fleet[i].rebalanceTo != to_which_area:  # State will change!


                pass

        # choose pickups
        for region_code in range(MAP_DIVIDE ** 2):
            if not pickup_list[region_code] or not open_requests_info_in_area[region_code]:
                continue
            dist_table = [[0 for _ in range(len(pickup_list[region_code]))] for __ in range(len(open_requests_info_in_area[region_code]))]  # Req x Vehicle
            for vehicle_label in range(len(pickup_list[region_code])):
                for request_label in range(len(open_requests_info_in_area[region_code])):
                    dist_table[request_label][vehicle_label] = self.fleet[vehicle_label].get_distance_to(
                        open_requests_info_in_area[region_code][request_label][1][0],
                        open_requests_info_in_area[region_code][request_label][1][1])



        # for request in open_requests:
        #     vehicles_to_request_distance = [1e8 for i in range(NUMBER_OF_VEHICLES)]
        #     for i in range(NUMBER_OF_VEHICLES):
        #         if self.fleet[i].status == STAY or self.fleet[i].status == REBALANCE:
        #             vehicles_to_request_distance[i] = self.fleet[i].get_distance_to(request[1][0], request[1][1])
        #     choose_car = arg_min(vehicles_to_request_distance)
        #     if vehicles_to_request_distance[choose_car] > 1e7:  # if no car available
        #         pass
        #     else:
        #         pickup.append([choose_car, request[0]])
        #         self.responded_requests.append(request[0])

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

    def data_preprocess(self, status):
        # This function processes the data so that we can use it for further learning
        # Expected to do: Grid, # of requests within the grid, Poisson Distribution with parameter lambda, ...

        self.time = status[0]

        # coordination change and update vehicle information
        num_vehicles_in_area = [0 for _ in range(MAP_DIVIDE ** 2)]
        vehicles_in_each_area = [[]] * (NUMBER_OF_VEHICLES**2)
        distance_to_each_area = [[0. for _ in range(MAP_DIVIDE ** 2)] for __ in range(NUMBER_OF_VEHICLES)]
        vehicles_should_get_rewards = []
        vehicles_should_update = [False] * NUMBER_OF_VEHICLES

        vehicle_last_state = []  # should save in this order:
        # status, location, rebalance to, rebalance start time, pickup start time, get pickup at rebalance, last stay time

        for i in range(NUMBER_OF_VEHICLES):
            loc = self.coordinate_change('TO_MODEL', status[1][i][1])
            status = status[1][i][2]  # this status has the type RoboTaxiStatus.XXX
            vehicle_last_state.append(
                [self.fleet[i].status, self.fleet[i].loc, self.fleet[i].rebalanceTo, self.fleet[i].rebalanceStartTime,
                 self.fleet[i].pickupStartTime, self.fleet[i].getPickupAtRebalance, self.fleet[i].lastStayTime])
            self.fleet[i].update(loc, status, self.time)
            if self.fleet[i].status is STAY or self.fleet[i].status is REBALANCE:
                num_vehicles_in_area[self.fleet[i].area] += 1
                vehicles_in_each_area[self.fleet[i].area].append(i)
            for j in range(MAP_DIVIDE ** 2):
                distance_to_each_area[i][j] = self.fleet[i].get_distance_to(MID_POINTS[j])

        # Process Requests
        # Process Request Distribution & open requests
        open_requests = []  # this saves open requests' labels & ori. position
        # add
        for request in status[2]:
            this_location = self.coordinate_change('TO_MODEL', request[2])
            if request[0] < self.numRequestSeen:
                flag = False
                for responded in self.responded_requests:
                    if request[0] == responded:
                        flag = True
                        break
                if not flag:
                    open_requests.append([request[0], this_location])
                pass
            else:
                self.history_requests.append([request[1], which_area(this_location), this_location])  # time, area, location
                self.numRequestSeen = request[0]
                open_requests.append([request[0], this_location])

        # Here put requests into areas: open_requests_in_area
        open_requests_in_area = [0 for _ in range(MAP_DIVIDE ** 2)]
        open_requests_info_in_area = [[]] * (MAP_DIVIDE ** 2)
        for req in open_requests:
            my_area = which_area(req[1][0], req[1][1])
            open_requests_in_area[my_area] += 1
            open_requests_info_in_area[my_area].append(req)

        while self.history_requests[0][0] < self.time - self.keep_history_time:
            self.history_requests.pop(0)

        # Update history request on the map
        request_distribution = [0 for _ in range(MAP_DIVIDE ** 2)]
        for his_request in self.history_requests:
            request_distribution[his_request[1]] += 1

        # TODO: further process history request for CNN

        # get all vehicles that should update action
        update_areas = self.areas_to_handle_requests(open_requests_in_area)
        for i in range(MAP_DIVIDE ** 2):
            if update_areas[i] == True:
                for j in range(vehicles_in_each_area[i]):
                    vehicles_should_update[j] = True
        for i in range(NUMBER_OF_VEHICLES):
            if vehicles_should_update[i] == False and self.should_update_individual(self.fleet[i], vehicle_last_state[i]):
                vehicles_should_update[i] = True


        states = []
        for i in range(NUMBER_OF_VEHICLES):
            individual_state = [[]] * 5
            if vehicles_should_update[i]:
                individual_state[3] = i
                curr_area = self.fleet[i].area
                individual_state[4] = curr_area
                update_area2D = [curr_area / MAP_DIVIDE, curr_area % MAP_DIVIDE] + NINE_REGIONS
                for area in update_area2D:
                    if area[0] >= 0 and area[1] >= 0 and area[0] < MAP_DIVIDE and area[1] < MAP_DIVIDE:
                        individual_state[0].append(open_requests_in_area[area])
                        individual_state[1].append(request_distribution[area])
                        individual_state[2].append(num_vehicles_in_area[area])
                    else:
                        # -1: Illegal Region
                        individual_state[0].append(-1)
                        individual_state[1].append(-1)
                        individual_state[2].append(-1)
                states.append(individual_state)







        # # update s', r
        # for i in range(NUMBER_OF_VEHICLES):
        #     if self.fleet[i].flagStateChange == 1:
        #         self.fleet[i].data[2] = [] # match the variable state
        #         self.fleet[i].data[3] = reward
        #         memory.push(*tuple(self.fleet[i].data))
        #
        # # remove ?
        # return num_vehicles_in_area, distance_to_each_area, request_distribution, open_requests
        return states, open_requests_info_in_area

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

    def reward_compute(self, vehicle, new_state):
        # this function computes the reward given the last state and current state

        assert isinstance(vehicle, Vehicle)
        reward = None
        if vehicle.status == REBALANCE and new_state == STAY:  # end of rebalance: give deduction
            reward = (self.time - vehicle.rebalanceStartTime) * DISTANCE_COST
        if new_state == DRIVEWITHCUSTOMER and vehicle.status != DRIVEWITHCUSTOMER:
            reward = PICKUP_REWARD
            if vehicle.getPickupAtRebalance == True:
                reward += (self.time - vehicle.rebalanceStartTime) * DISTANCE_COST
            else:
                reward += (self.time - vehicle.pickupStartTime) * DISTANCE_COST
        return reward

    def coordinate_change(self, direction, loc):
        if direction == 'TO_MODEL':
            return [ (loc[0] - self.lngMin) / self.unitLongitude, (loc[1] - self.latMin) / self.unitLatitude]
        elif direction == 'TO_COMMAND':
            return [loc[0] * self.unitLongitude + self.lngMin, loc[1] * self.unitLatitude + self.latMin]
        else:
            raise ValueError

    def areas_to_handle_requests(self, open_requests_in_area):
        areas = [False for _ in range(NUMBER_OF_VEHICLES ** 2)]
        for area1D in range(MAP_DIVIDE ** 2):
            if open_requests_in_area[area1D] > 0:
                area_code2D = [area1D // MAP_DIVIDE, area1D % MAP_DIVIDE]
                nine_regions = area_code2D + NINE_REGIONS
                for area in nine_regions:
                    if area[0] >= 0 and area[1] >= 0 and area[0] < MAP_DIVIDE and area[1] < MAP_DIVIDE:
                        areas[area[0] * MAP_DIVIDE + area[1]] = True
        return areas

    def should_update_individual(self,vehicle, last_state):
        assert isinstance(vehicle, Vehicle)
        if last_state == REBALANCE and vehicle.status == STAY:
            return True
        if last_state == DRIVEWITHCUSTOMER and vehicle.status == STAY:
            return True
        if last_state == STAY and vehicle.status == STAY and vehicle.lastStayTime - self.time >= STAY_TIMEOUT:
            return True
        return False

    def should_get_reward(self, vehicle, last_state):
        assert isinstance(vehicle, Vehicle)
        if last_state == STAY and vehicle.status == STAY and vehicle.lastStayTime - self.time >= STAY_TIMEOUT:
            return True
        if last_state == REBALANCE and vehicle.status == STAY:
            return True
        if last_state == DRIVETOCUSTOMER and vehicle.status == DRIVEWITHCUSTOMER:
            return True
        else:
            return False





