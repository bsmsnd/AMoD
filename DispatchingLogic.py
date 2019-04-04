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
import warnings

memory = ReplayMemory(10000)
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))


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
        self.policy_net = DQN(N_FEATURE, N_ACTION).to(self.device)
        self.target_net = DQN(N_FEATURE, N_ACTION).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.steps_done = 0

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
        states, open_requests_info_in_area, vehicles_should_get_rewards, vehicle_last_state = self.data_preprocess(status)
        # States: [0] open requests in surroundings, [1] history req in surroundings
        #         [2] number vehicles, [3] label which car, [4] which area

        states_as_records = []
        for i in range(len(states[0])):
            states_as_records.append([states[0][i], states[1][i], states[2][i], states[3][i], states[4][i]])

        # Save the state for later learning (maybe?) and update states
        old_states = [self.fleet[i].last_state for i in range(NUMBER_OF_VEHICLES)]
        for individual_state in states_as_records:
            self.fleet[individual_state[3]] = individual_state

        #TODO: CNN and DQN
        open_req = torch.tensor(states[0])  # size of batch_size x 9
        num_veh = torch.tensor(states[2])  # size of batch_size x 9
        his_req = torch.tensor(states[1]).view(len(states), -1, 3, 3) # size of batch_size x 4 x 3 x 3
        actions = self.select_action(open_req, num_veh, his_req)
        # 'actions' is the list output of DQNs, length of which is the same as that of states
        # Some possible values are defined as follows:
        # 0: STAY, 1-9: PICKUP at the relative region from topleft to bottomright
        # 10-18: REBALANCE to 1-9 regions from topleft to bottomright

        final_command_for_each_vehicle = [-1] * NUMBER_OF_VEHICLES
        pickup_list = [[]] * (MAP_DIVIDE ** 2)
        for i, individual_state in enumerate(states_as_records):
            cmd = actions[i]
            vehicle_label = individual_state[3]
            if cmd == 0:
                vehicles_should_get_rewards[vehicle_label] = True
                final_command_for_each_vehicle[vehicle_label] = cmd
            elif 1 <= cmd <= 9:  # pick up 1-9
                goto = convert_area(individual_state[4], cmd-1, '1D', '1D')
                pickup_list[goto].append(individual_state[3])
            elif cmd > 9:  # rebalance 1-9
                goto = convert_area(individual_state[4], cmd - 9 - 1, '1D', '1D')
                if self.fleet[vehicle_label].rebalanceTo != goto:  # State will change!
                    new_location = None  # TODO: get the new rebalance location (Hui)
                    rebalance.append([individual_state[3], self.coordinate_change('TO_COMMAND', new_location)])
                    vehicles_should_get_rewards[vehicle_label] = True
                    final_command_for_each_vehicle[vehicle_label] = cmd
            else:
                raise ValueError('Illegal Action')

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

        # TODO: Use dist_table to choose pickups (Hui)
        # pickups is expected to be in the following form: [ [# vehicle, # req], ... ]
        # left_vehicles obtains the labels of vehicles that are not assigned requests
        # and yet choose pickup for next action.
        pickup, left_vehicles, left_requests = None, None, None

        for single_pickup in pickup:
            get_action = -1
            vehicle_label = single_pickup[0]
            for i in range(len(states_as_records)):
                if states_as_records[i][3] == single_pickup[0]:
                    get_action = actions[i]
                    break
            if get_action == -1:
                raise ValueError("Internal Error: did not find action. Ref to Dong.")
            self.fleet[single_pickup[0]].last_action = get_action
            self.fleet[vehicle_label].pickupStartTime = self.time
            self.fleet[vehicle_label].getPickupAtRebalance = (old_states[vehicle_label] == REBALANCE)

            self.responded_requests.append(single_pickup[0])
            
        # Handle all leftover vehicles
        leftover_states = []
        for vehicle_label in range(left_vehicles):
            get_state = None
            for i in range(len(states_as_records)):
                if states_as_records[i][3] == vehicle_label:
                    get_state = states_as_records[i].copy()
                    break
            if get_state is None:
                raise ValueError("Internal Error: did not find the state. Ref to Dong.")
            get_state[0] = [0] * 9  # Close all requests around
            leftover_states.append(get_state)

        state_for_dqn_leftover =  [[]] * 5
        for individual_state in range(leftover_states):
            state_for_dqn_leftover[0].append(individual_state[0])
            state_for_dqn_leftover[1].append(individual_state[1])
            state_for_dqn_leftover[2].append(individual_state[2])
            state_for_dqn_leftover[3].append(individual_state[3])
            state_for_dqn_leftover[4].append(individual_state[4])

        # TODO: Run CNN & DQN once again for left_over vehicles
        open_req_left = torch.tensor(leftover_states[0])  # size of batch_size x 9
        num_veh_left = torch.tensor(leftover_states[2])  # size of batch_size x 9
        his_req_left = torch.tensor(leftover_states[1]).view(len(leftover_states), -1, 3, 3)  # size of batch_size x 4 x 3 x 3
        remaining_actions = self.select_action(open_req_left, num_veh_left, his_req_left)

        for i, individual_state in enumerate(leftover_states):
            vehicle_label = individual_state[3]
            cmd = actions[i]
            if cmd == 0:
                vehicles_should_get_rewards[vehicle_label] = True
                final_command_for_each_vehicle[vehicle_label] = cmd
            if 1 <= cmd <= 9:  # pick up 1-9
                # NO requests any more!
                # TODO: QUESTION ON HOW TO HANDLE NO REQ BUT CHOOSE PICKUP; Current way: no change on status
                pass
            if cmd > 9:  # rebalance 1-9
                goto = convert_area(individual_state[4], cmd - 9 - 1, '1D', '1D')
                if self.fleet[i].rebalanceTo != goto:  # State will change!
                    new_location = None  # TODO: get the new rebalance location
                    rebalance.append([individual_state[3], self.coordinate_change('TO_COMMAND', new_location)])
                    vehicles_should_get_rewards[vehicle_label] = True
                    final_command_for_each_vehicle[vehicle_label] = cmd
            else:
                raise ValueError('Illegal Action')

        # handle rewards & ensemble a piece of record for Replay memory
        # all_replay = []
        for i in range(NUMBER_OF_VEHICLES):
            if vehicles_should_get_rewards[i]:
                r = self.reward_compute(self.fleet[i], vehicle_last_state[i])
                get_state = None
                for j in range(len(state_for_dqn_leftover)):
                    if state_for_dqn_leftover[j][3] == i:
                        get_state = state_for_dqn_leftover[j].copy()
                if not get_state:
                    for j in range(len(states_as_records)):
                        if states_as_records[j][3] == i:
                            get_state = states_as_records[j].copy()
                # should get a get state
                if not get_state:
                    warnings.warn('State not found for vehicle %d' % i)
                    continue
                # push the data into the memory
                open_req_last = torch.tensor(self.fleet[i].last_state[0])
                num_veh_last = torch.tensor(self.fleet[i].last_state[2])
                his_req_last = torch.tensor(self.fleet[i].last_state[1]).view(3, 3)
                open_req_new = torch.tensor(get_state[0])
                num_veh_new = torch.tensor(get_state[2])
                his_req_new = torch.tensor(get_state[1])
                memory.push(open_req_last, num_veh_last, his_req_last, self.fleet[i].last_action, open_req_new,
                            num_veh_new, his_req_new, r)

                # record = [self.fleet[i].last_state, self.fleet[i].last_action, get_state, r]
                # all_replay.append(record)
                # Set Status after getting reward
                if final_command_for_each_vehicle[i] == 0:  # 0: Action = 0 is STAY
                    self.fleet[i].update_stay(self.time)
                elif 10 <= final_command_for_each_vehicle[i] < 19:  # Action = 10 ~ 18 is REBALANCE
                    goto_relative = final_command_for_each_vehicle[i] - 9 - 1
                    to_area = convert_area(self.fleet[i].area,goto_relative,'2D', '1D')
                    self.fleet[i].update_rebalance(self.time, to_area)

        # Push this new status to Replay memory
        # TODO: Adjust the format for Replay memory
        # memory.push(all_replay)

        # Optimize the network
        self.optimize_model()

        return [pickup, rebalance]

    def data_preprocess(self, status):
        # This function processes the data so that we can use it for further learning
        # Expected to do: Grid, # of requests within the grid, Poisson Distribution with parameter lambda, ...

        self.time = status[0]

        # coordination change and update vehicle information
        num_vehicles_in_area = [0 for _ in range(MAP_DIVIDE ** 2)]
        vehicles_in_each_area = [[]] * (NUMBER_OF_VEHICLES**2)
        distance_to_each_area = [[0. for _ in range(MAP_DIVIDE ** 2)] for __ in range(NUMBER_OF_VEHICLES)]

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
            if self.fleet[i].status == STAY or self.fleet[i].status == REBALANCE:
                num_vehicles_in_area[self.fleet[i].area] += 1
                vehicles_in_each_area[self.fleet[i].area].append(i)
            for j in range(MAP_DIVIDE ** 2):
                distance_to_each_area[i][j] = self.fleet[i].get_distance_to(MID_POINTS[j][0], MID_POINTS[j][1])

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
                self.history_requests.append([request[1], which_area(this_location[0], this_location[1]), this_location])  # time, area, location
                self.numRequestSeen = request[0]
                open_requests.append([request[0], this_location])

        # Here put requests into areas: open_requests_in_area
        open_requests_in_area = [0 for _ in range(MAP_DIVIDE ** 2)]
        open_requests_info_in_area = [[]] * (MAP_DIVIDE ** 2)
        for req in open_requests:
            my_area = which_area(req[1][0], req[1][1])
            open_requests_in_area[my_area] += 1
            open_requests_info_in_area[my_area].append(req)

        while self.history_requests and self.history_requests[0][0] < self.time - self.keep_history_time:
            self.history_requests.pop(0)

        # Update history request on the map
        request_distribution = [0 for _ in range(MAP_DIVIDE ** 2)]
        for his_request in self.history_requests:
            request_distribution[his_request[1]] += 1

        # get all vehicles that should update action
        update_areas = self.areas_to_handle_requests(open_requests_in_area)
        for i in range(MAP_DIVIDE ** 2):
            if update_areas[i] == True:
                for j in range(vehicles_in_each_area[i]):
                    vehicles_should_update[j] = True
        for i in range(NUMBER_OF_VEHICLES):
            if not vehicles_should_update[i] and self.should_update_individual(self.fleet[i], vehicle_last_state[i]):
                vehicles_should_update[i] = True

        states = []
        for i in range(NUMBER_OF_VEHICLES):
            individual_state = [[]] * 5
            if vehicles_should_update[i]:
                individual_state[3] = i
                curr_area = self.fleet[i].area
                individual_state[4] = curr_area
                update_area2D = convert_area(curr_area, None, '1D', '2D') + NINE_REGIONS
                for area in update_area2D:
                    if 0 <= area[0] < MAP_DIVIDE and 0 <= area[1] < MAP_DIVIDE:
                        area1D = convert_area(area, None,'2D', '1D')
                        individual_state[0].append(open_requests_in_area[area1D])
                        individual_state[1].append(request_distribution[area1D])
                        individual_state[2].append(num_vehicles_in_area[area1D])
                    else:
                        # -1: Illegal Region
                        individual_state[0].append(-1)
                        individual_state[1].append(-1)
                        individual_state[2].append(-1)
                states.append(individual_state)

        vehicles_should_get_rewards = [False] * NUMBER_OF_VEHICLES
        for i in range(NUMBER_OF_VEHICLES):
            vehicles_should_get_rewards[i] = self.should_get_reward(self.fleet[i], vehicle_last_state[i])

        # # update s', r
        # for i in range(NUMBER_OF_VEHICLES):
        #     if self.fleet[i].flagStateChange == 1:
        #         self.fleet[i].data[2] = [] # match the variable state
        #         self.fleet[i].data[3] = reward
        #         memory.push(*tuple(self.fleet[i].data))
        #
        # # remove ?
        # return num_vehicles_in_area, distance_to_each_area, request_distribution, open_requests
        return states, open_requests_info_in_area, vehicles_should_get_rewards, vehicle_last_state

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
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
        #                               dtype=torch.uint8)
        # non_final_next_states = torch.cat([s for s in batch.next_state
        #                                    if s is not None])
        open_req_last_batch = torch.cat(batch.open_req_last)
        num_veh_last_batch = torch.cat(batch.num_veh_last)
        his_req_last_batch = torch.cat(batch.his_req_last)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        open_req_new_batch = torch.cat(batch.open_req_new)
        num_veh_new_batch = torch.cat(batch.num_veh_new)
        his_req_new_batch = torch.cat(batch.his_req_new)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(open_req_last_batch, num_veh_last_batch, his_req_last_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = self.target_net(open_req_new_batch, num_veh_new_batch, his_req_new_batch).max(1)[0].detach()
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

    def reward_compute(self, vehicle, old_state):
        """
        This function computes the reward given the last state and current state
        :param vehicle: the vehicle to get reward. Type: Vehicle
        :param old_state: a constant representing the old state.
        :return: a score in double.
        """

        assert isinstance(vehicle, Vehicle)
        reward = None
        if old_state == REBALANCE and vehicle.status == STAY:  # end of rebalance: give deduction
            reward = (self.time - vehicle.rebalanceStartTime) * DISTANCE_COST
        if vehicle.status == DRIVEWITHCUSTOMER and old_state != DRIVEWITHCUSTOMER:
            reward = PICKUP_REWARD
            if vehicle.getPickupAtRebalance:
                reward += (vehicle.pickupEndTime - vehicle.rebalanceStartTime) * DISTANCE_COST
            else:
                reward += (vehicle.pickupEndTime - vehicle.pickupStartTime) * DISTANCE_COST
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

    def select_action(self, open_req, num_veh, his_req):
        sample = torch.randn(open_req.size(0))
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1  # need to change to global variable
        mask = sample > eps_threshold
        with torch.no_grad():
            actions = self.policy_net(open_req, num_veh, his_req).max(1)[1]
        actions_greedy = torch.randint_like(actions)
        actions = (mask * actions + (torch.ones_like(mask) - mask) * actions_greedy).to(self.device)
        return actions
        # torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)




