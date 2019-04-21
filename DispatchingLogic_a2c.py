import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from utils.RoboTaxiStatus import RoboTaxiStatus
from ReplayMemory import *
from Dqn import *
from Dqn_dueling import *
from collections import namedtuple
from Vehicle import Vehicle
from constant import *
from generic import *
from distance_on_unit_sphere import *
import warnings
import numpy as np
import scipy.optimize as op

# FOR A2C

from itertools import count
from collections import namedtuple
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from a2c import *



# memory = ReplayMemory(MEMORY_SIZE)
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

#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')

        # self.last_state = None

        # Change the policy net below
 #       self.policy_net = DQN(N_FEATURE, N_ACTION).to(self.device)
        
#        self.policy_net = DuelingDQN(N_FEATURE, N_ACTION).to(self.device)
        self.a2c_model = A2C(N_FEATURE, N_ACTION).to(self.device)
        if LOAD_FLAG == True:
            self.a2c_model = loadweight(self.a2c_model, LOAD_PATH)

        
#        self.target_net = DQN(N_FEATURE, N_ACTION).to(self.device)
#        self.target_net = DuelingDQN(N_FEATURE, N_ACTION).to(self.device)       
#        self.target_net.load_state_dict(self.policy_net.state_dict())
#        self.target_net.eval()
#        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
#        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=1e-2, momentum=0.95)

        # FOR A2C
        
        self.optimizer = optim.Adam(self.a2c_model.parameters(), lr=1e-4)
        self.eps = np.finfo(np.float32).eps.item()

        
        self.steps_done = 0

        self.history_requests = []
        self.numRequestSeen = 0

        self.keep_history_time = 3600
        self.part_history = 900
        self.time = 0

        # Assume that coordination will be converted to distances in miles
        self.unitLongitude = (self.lngMax - self.lngMin) / GRAPHMAXCOORDINATE
        self.map_width = distance_on_unit_sphere(self.latMin, self.lngMin, self.latMin, self.lngMax)
        self.map_length = distance_on_unit_sphere(self.latMax, self.lngMax, self.latMin, self.lngMax)
        global LAT_SCALE
        self.lat_scale = GRAPHMAXCOORDINATE * self.map_length / self.map_width
        LAT_SCALE = self.lat_scale
        self.unitLatitude = (self.latMax - self.latMin) / self.lat_scale

        self.fleet = [Vehicle() for _ in range(NUMBER_OF_VEHICLES)]

        # Requests
        self.responded_requests = []

        # rewards
        self.running_reward = 0
        self.n_rewards = 0
        self.slid_reward = []

        self.smallRegionToGlobalRegion = [0 for _ in range(MAP_DIVIDE ** 2)]
        for region_code in range(MAP_DIVIDE ** 2):
            area2D = convert_area(region_code, None, '1D', '2D')
            global_area_code = 0
            if area2D[0] // GLOBAL_DIVIDE > 0:
                global_area_code += 2
            if area2D[1] // GLOBAL_DIVIDE > 0:
                global_area_code += 1
            self.smallRegionToGlobalRegion[region_code] = global_area_code

    def of(self, status):
        ####################################################################################
        # This function returns the commands to vehicles given status
        # Return format: [pickup, rebalance]
        # pickup: a list of pickup commands [ [# vehicle, # request],...]
        # Rebalance: a list of rebalance commands: [ [# vehicle, rebalance_to], ...]
        ####################################################################################

        # Output these values:
        pickup = []
        rebalance = []

        # Pre-process the data
        # num_vehicles_in_area, distance_to_each_area, request_distribution, open_requests = self.data_preprocess(status)
        states, open_requests_info_in_area, vehicles_should_get_rewards, vehicle_last_state, statistics = self.data_preprocess(status)
        # states: [0] open requests in surroundings, [1] history req in surroundings
        #         [2] number vehicles, [3] label which car, [4] which area
        open_requests_global, request_distribution_global, num_vehicles_in_area_global = statistics

        # states_as_records stores all data in states in pieces of record, each one corresponds to a vehicle
        states_as_records = []  # in the same order of states

        vehicles_should_update = [False for _ in range(NUMBER_OF_VEHICLES)]
        for vehicle_label in states[3]:
            vehicles_should_update[vehicle_label] = True

        states_copy = states.copy()
        vehicle_label_to_index_in_states = [-1 for _ in range(NUMBER_OF_VEHICLES)]  # saves the index for each vehicle

        for i in range(len(states[3])):
            vehicle_label_to_index_in_states[states[3][i]] = i

        if not states[0]:
            return [pickup, rebalance]  # empty

        for i in range(len(states[0])):
            states_as_records.append([states[0][i], states[1][i], states[2][i], states[3][i], states[4][i]])

        old_states = [self.fleet[i].last_state for i in range(NUMBER_OF_VEHICLES)]
        all_last_actions = [self.fleet[i].last_action for i in range(NUMBER_OF_VEHICLES)]

        while states_as_records:
            pickup_one_step = []

            # Save the state for later learning (maybe?) and update states


            for individual_state in states_as_records:
                self.fleet[individual_state[3]].last_state = individual_state

            # A2C
            open_req = torch.tensor(states[0], dtype=torch.float)  # size of batch_size x 9
            num_veh = torch.tensor(states[2], dtype=torch.float)  # size of batch_size x 9
            his_req = torch.tensor(states[1], dtype=torch.float).transpose(1, 2).view(len(states[0]), -1, 3, 3) # size of batch_size x 4 x 3 x 3
            # Add new global state
            batch_size = open_req.size()[0]

            open_req_global = torch.tensor(open_requests_global, dtype=torch.float).to(self.device) # 4
            num_veh_global = torch.tensor(num_vehicles_in_area_global, dtype=torch.float).to(self.device)  # size of 4
            his_req_global = torch.tensor(request_distribution_global, dtype=torch.float).transpose(0, 1).to(self.device)  # size of 4 x 4

            open_req_global = open_req_global.view(1, open_req_global.size()[0])
            num_veh_global = num_veh_global.view(1, num_veh_global.size()[0])
            his_req_global = his_req_global.contiguous().view(1, his_req_global.size()[0]*his_req_global.size()[1])
            open_req_global = torch.ones([batch_size, open_req_global.size()[1]]) * open_req_global
            num_veh_global = torch.ones([batch_size, num_veh_global.size()[1]]) * num_veh_global
            his_req_global = torch.ones([batch_size, his_req_global.size()[1]]) * his_req_global
            actions, saved_actions = self.a2c_select_action(open_req, num_veh, his_req, open_req_global, num_veh_global, his_req_global)

            # gather vehicle labels in each region
            vehicles_in_each_region = [[] for _ in range(MAP_DIVIDE ** 2)]
            for single_state in states_as_records:
                area = single_state[4]
                label = single_state[3]
                vehicles_in_each_region[area].append(label)

            bad_pickup_vehicles = []
            bad_rebalance_vehicles = []
            # 'actions' is the list output of DQNs, length of which is the same as that of states
            # Some possible values are defined as follows:
            # 0: STAY, 1-9: PICKUP at the relative region from topleft to bottomright
            # 10-18: REBALANCE to 1-9 regions from topleft to bottomright

            final_command_for_each_vehicle = [-1 for _ in range(NUMBER_OF_VEHICLES)]
            pickup_list = [[] for _ in range(MAP_DIVIDE**2)]
            vehicles_decided_new_action = [False for _ in range(len(states_as_records))] # saves the indices to remove in the list
            request_handled = [[False for _ in range(len(open_requests_info_in_area[region_code]))] for region_code in
                               range(MAP_DIVIDE ** 2)]  # saves the indices to remove in the list
            for i, individual_state in enumerate(states_as_records):
                cmd = actions[i]
                vehicle_label = individual_state[3]
                if cmd == 0:
                    vehicles_decided_new_action[i] = True
                    final_command_for_each_vehicle[vehicle_label] = cmd
                    if self.fleet[vehicle_label].last_action is None:
                        pass
                    else:
                        vehicles_should_get_rewards[vehicle_label] = True
                elif 1 <= cmd <= 9:  # pick up 1-9
                    goto = convert_area(individual_state[4], cmd-1, '1D', '1D')
                    if goto != ILLEGAL_AREA and open_requests_info_in_area[goto]:  # not empty
                        pickup_list[goto].append(individual_state[3])
                    else:
                        vehicles_decided_new_action[i] = True  # TODO
                        #Memory_dataProcess(individual_state, cmd, individual_state, R_ILLEGAL, memory)
                        self.running_reward = self.running_reward + R_ILLEGAL
                        self.slid_reward.append(R_ILLEGAL)
                        if len(self.slid_reward) > SLIDE_WIN_SIZE:
                            self.running_reward -= self.slid_reward[0]
                            del self.slid_reward[0]
                        self.n_rewards = len(self.slid_reward)
                        a2c_data_process(self.fleet[vehicle_label], R_ILLEGAL, saved_actions[i])
                        bad_pickup_vehicles.append(vehicle_label)
                elif 9 < cmd <= 18:  # rebalance 1-9
                    goto = convert_area(individual_state[4], cmd - 9 - 1, '1D', '1D')

                    vehicles_decided_new_action[i] = True
                    if goto == ILLEGAL_AREA:
                        #Memory_dataProcess(individual_state, cmd, individual_state, R_ILLEGAL, memory)
                        self.running_reward = self.running_reward + R_ILLEGAL
                        self.slid_reward.append(R_ILLEGAL)
                        if len(self.slid_reward) > SLIDE_WIN_SIZE:
                            self.running_reward -= self.slid_reward[0]
                            del self.slid_reward[0]
                        self.n_rewards = len(self.slid_reward)
                        a2c_data_process(self.fleet[vehicle_label], R_ILLEGAL, saved_actions[i])
                        bad_rebalance_vehicles.append(vehicle_label)
                    else:
                        # Will sample a rebalance location, even if the rebalance is not changed
                        # random location
                        long_min = GRAPHMAXCOORDINATE / MAP_DIVIDE * (goto % MAP_DIVIDE)
                        lati_min = self.lat_scale / MAP_DIVIDE * (goto // MAP_DIVIDE)
                        new_location = (np.random.uniform(long_min, long_min + GRAPHMAXCOORDINATE / MAP_DIVIDE),
                                        np.random.uniform(lati_min, lati_min + self.lat_scale / MAP_DIVIDE))  # TODO: get the new rebalance location
                        rebalance.append([individual_state[3], self.coordinate_change('TO_COMMAND', new_location)])
                        if self.fleet[vehicle_label].last_action is None:
                            pass
                        else:
                            vehicles_should_get_rewards[vehicle_label] = True
                        final_command_for_each_vehicle[vehicle_label] = cmd
                elif 18 < cmd <= 22:
                    index = cmd - 9 - 9 - 1
                    mid_num = MAP_DIVIDE // GLOBAL_DIVIDE
                    mid_lon = mid_num * GRAPHMAXCOORDINATE / MAP_DIVIDE
                    mid_lat = mid_num * self.lat_scale / MAP_DIVIDE
                    if index // GLOBAL_DIVIDE == 0:
                        sample_lat_min = 0
                        sample_lat_max = mid_lat
                    else:
                        sample_lat_min = mid_lat
                        sample_lat_max = self.lat_scale
                    if index % GLOBAL_DIVIDE == 0:
                        sample_lon_min = 0
                        sample_lon_max = mid_lon
                    else:
                        sample_lon_min = mid_lon
                        sample_lon_max = GRAPHMAXCOORDINATE

                    vehicles_decided_new_action[i] = True
                    if False:
                    #if goto == ILLEGAL_AREA:
                        pass
                        #Memory_dataProcess(individual_state, cmd, individual_state, R_ILLEGAL, memory)
                        #bad_rebalance_vehicles.append(vehicle_label)
                    else:
                        # Will sample a rebalance location, even if the rebalance is not changed
                        # random location
                        new_location = (np.random.uniform(sample_lon_min, sample_lon_max),
                                        np.random.uniform(sample_lat_min,
                                                          sample_lat_max))  # TODO: get the new rebalance location
                        rebalance.append([individual_state[3], self.coordinate_change('TO_COMMAND', new_location)])
                        if self.fleet[vehicle_label].last_action is None:
                            pass
                        else:
                            vehicles_should_get_rewards[vehicle_label] = True
                        final_command_for_each_vehicle[vehicle_label] = cmd
                else:
                    raise ValueError('Illegal Action')

            left_vehicles, left_requests = [], []
            # pickups is expected to be in the following form: [ [# vehicle, # req], ... ]
            # left_vehicles obtains the labels of vehicles that are not assigned requests
            # and yet choose pickup for next action.

            # choose pickups
            n_left_requests = [0 for _ in range(MAP_DIVIDE ** 2)]
            for region_code in range(MAP_DIVIDE ** 2):
                if not pickup_list[region_code] or not open_requests_info_in_area[region_code]:
                    n_left_requests[region_code] = len(open_requests_info_in_area[region_code])
                dist_table = [[0 for _ in range(len(pickup_list[region_code]))] for __ in range(len(open_requests_info_in_area[region_code]))]  # Req x Vehicle
                for i in range(len(pickup_list[region_code])):
                    vehicle_label = pickup_list[region_code][i]
                    for request_label in range(len(open_requests_info_in_area[region_code])):
                        dist_table[request_label][i] = self.fleet[vehicle_label].get_distance_to(
                            open_requests_info_in_area[region_code][request_label][1][0],
                            open_requests_info_in_area[region_code][request_label][1][1])

                # Use dist_table to choose pickups
                if dist_table:
                    # row is request label   col is vehicle label
                    row, col = op.linear_sum_assignment(dist_table)
                    for i in range(len(row)):
                        which_car = pickup_list[region_code][col[i]]
                        which_request = open_requests_info_in_area[region_code][row[i]][0]
                        pickup_one_step.append([which_car, which_request])
                        final_command_for_each_vehicle[which_car] = actions[vehicle_label_to_index_in_states[which_car]]
                        # update global stats
                        global_area_code_for_req = self.smallRegionToGlobalRegion[region_code]
                        open_requests_global[global_area_code_for_req] -= 1
                        global_area_code_for_vehicle = self.smallRegionToGlobalRegion[self.fleet[which_car].area]
                        num_vehicles_in_area_global[global_area_code_for_vehicle] -= 1
                        for j in range(len(states[3])):
                            if states[3][j] == which_car:
                                vehicles_decided_new_action[j] = True
                                break
                        for j in range(len(open_requests_info_in_area[region_code])):
                            if open_requests_info_in_area[region_code][j][0] == which_request:
                                request_handled[region_code][j] = True
                    m, n = len(dist_table), len(dist_table[0])
                    if m < n:
                        # #request is less than #vehicle label
                        for i in range(n):
                            if i not in col:
                                left_vehicles.append(pickup_list[region_code][i])
                    elif m > n:
                        n_left_requests[region_code] = m - n
                        for i in range(m):
                            if i not in row:
                                left_requests.append(open_requests_info_in_area[region_code][i])
            

            

            for single_pickup in pickup_one_step:
                get_action = -1
                vehicle_label = single_pickup[0]
                for i in range(len(states_as_records)):
                    if states_as_records[i][3] == single_pickup[0]:
                        get_action = actions[i]
                        break
                if get_action == -1:
                    raise ValueError("Internal Error: did not find action.")
                self.fleet[vehicle_label].getPickupAtRebalance = (self.fleet[vehicle_label].status == REBALANCE)
                self.fleet[vehicle_label].last_action = get_action
                self.fleet[vehicle_label].pickupStartTime = self.time
                self.responded_requests.append(single_pickup[1])

            for region_code in range(MAP_DIVIDE ** 2):
                n_removed = 0
                for i in range(len(open_requests_info_in_area[region_code])):
                    if request_handled[region_code][i]:
                        open_requests_info_in_area[region_code].pop(i - n_removed)
                        n_removed += 1
                    else:
                        this_request_location = open_requests_info_in_area[region_code][i - n_removed][1]
                        area2D = [convert_area(region_code, i, '1D', '2D') for i in range(9)]
                        for this_region2D in area2D:
                            if this_region2D[0] != ILLEGAL_AREA:
                                area1D = convert_area(this_region2D, None, '2D', '1D')
                                for vehicle in vehicles_in_each_region[area1D]:  # type: int
                                    if final_command_for_each_vehicle[vehicle] == 0 or 10 <= final_command_for_each_vehicle[vehicle] <= 18:
                                            self.fleet[vehicle].penalty_for_not_pickup_for_next_time += NO_PICKUP_PENALTY / self.fleet[vehicle].get_distance_to(this_request_location[0], this_request_location[1])

            n_removed = 0
            for i in range(len(states_as_records)):
                if vehicles_decided_new_action[i]:
                    self.fleet[states_copy[3][i]].a2c_saved_actions.append(saved_actions[i])
                    states_as_records.pop(i - n_removed)
                    n_removed += 1

            n_available_vehicles = [0 for _ in range(MAP_DIVIDE ** 2)]
            # Will not count the vehicles that already decided to STAY or REBALANCE
            # because these actions are decided and hence will not pick up in this step
            for single_state in states_as_records:
                n_available_vehicles[single_state[4]] += 1

            n_open_request = [len(open_requests_info_in_area[region_code]) for region_code in range(MAP_DIVIDE ** 2)]

            new_state = [[] for _ in range(5)]
            for single_state in states_as_records:
                curr_area = single_state[4]
                update_area2D = [convert_area(curr_area, i, '1D', '2D') for i in range(9)]
                open_req_for_this_vehicle = [0 for _ in range(9)]
                n_vehicles_for_this_vehicle = [0 for _ in range(9)]

                for i in range(9):
                    area = update_area2D[i]
                    if 0 <= area[0] < MAP_DIVIDE and 0 <= area[1] < MAP_DIVIDE:
                        area1D = convert_area(area, None, '2D', '1D')
                        open_req_for_this_vehicle[i] = n_open_request[area1D]
                        n_vehicles_for_this_vehicle[i] = n_available_vehicles[area1D]
                    else:
                        # -1: Illegal Region
                        open_req_for_this_vehicle[i] = -1
                        n_vehicles_for_this_vehicle[i] = -1
                single_state[0] = open_req_for_this_vehicle
                single_state[2] = n_vehicles_for_this_vehicle

                for i in range(5):
                    new_state[i].append(single_state[i])

            states = new_state
            for i in range(len(new_state[3])):
                idx = vehicle_label_to_index_in_states[new_state[3][i]]
                assert states_copy[3][idx] == new_state[3][i]
                states_copy[0][idx] = new_state[0][i].copy()
                states_copy[1][idx] = new_state[1][i].copy()
                states_copy[2][idx] = new_state[2][i].copy()

            pickup = pickup + pickup_one_step
            # END OF WHILE LOOP
        # print('bad pickup:')
        # print(bad_pickup_vehicles)
        # print('bad rebalance')
        # print(bad_rebalance_vehicles)

        for vehicle_label in range(NUMBER_OF_VEHICLES):
            if vehicles_should_update[vehicle_label]:
                if final_command_for_each_vehicle[vehicle_label] == 0:  # 0: Action = 0 is STAY
                    self.fleet[vehicle_label].update_stay(self.time)
                elif 1 <= final_command_for_each_vehicle[vehicle_label] < 10:
                    self.fleet[vehicle_label].last_action = final_command_for_each_vehicle[vehicle_label]
                elif 10 <= final_command_for_each_vehicle[vehicle_label] < 19:  # Action = 10 ~ 18 is REBALANCE
                    goto_relative = final_command_for_each_vehicle[vehicle_label] - 9 - 1
                    to_area = convert_area(self.fleet[vehicle_label].area, goto_relative,'1D', '1D')
                    self.fleet[vehicle_label].update_rebalance(self.time, to_area)



        # handle rewards & ensemble a piece of record for Replay memory
        # all_replay = []
        for i in range(NUMBER_OF_VEHICLES):  # i: vehicle label
            if vehicles_should_get_rewards[i]:
                r = self.reward_compute(self.fleet[i], vehicle_last_state[i][0])
                
                # Save rewards here
                self.running_reward = self.running_reward + r
                self.slid_reward.append(r)
                if len(self.slid_reward) > SLIDE_WIN_SIZE:
                    self.running_reward -= self.slid_reward[0]
                    del self.slid_reward[0]
                self.n_rewards = len(self.slid_reward)
                
                """
                self.n_rewards = self.n_rewards + 1
                self.running_reward = self.running_reward + r
                """
                idx = vehicle_label_to_index_in_states[i]
                assert idx != -1
                get_state = [states_copy[0][idx], states_copy[1][idx], states_copy[2][idx],
                             states_copy[3][idx], states_copy[4][idx]]
                # should get a get state
                if not get_state:
                    warnings.warn('State not found for vehicle %d' % i)
                    continue
                # push the data into the memory
                # Memory_dataProcess(old_states[i], all_last_actions[i], get_state, r, memory)
                self.fleet[i].a2c_reward.append(r)
                # optimize the network
                if len(self.fleet[i].a2c_reward) >= TRAIN_THRESHOLD:
                    self.optimize_model(self.fleet[i].a2c_saved_actions[:len(self.fleet[i].a2c_reward)], self.fleet[i].a2c_reward)
                    del self.fleet[i].a2c_saved_actions[:len(self.fleet[i].a2c_reward)]
                    del self.fleet[i].a2c_reward[:]
                # record = [self.fleet[i].last_state, self.fleet[i].last_action, get_state, r]
                # all_replay.append(record)
                # Set Status after getting reward


        # Optimize the network
        # self.optimize_model()
        if SAVE_FLAG==True:
            if self.time % SAVE_PERIOD == 0:
                saveweight(self.a2c_model, SAVE_PATH)


        if self.time % PRINT_REWARD_PERIOD == 0:
            if self.n_rewards > 0:
                print("time %d:  %d rewards with average reward = %.4f" % (self.time, self.n_rewards, self.running_reward / self.n_rewards))
            else:
                print("time %d: no rewards so far" % self.time)
        return [pickup, rebalance]

    def data_preprocess(self, status):
        # This function processes the data so that we can use it for further learning
        # Expected to do: Grid, # of requests within the grid, Poisson Distribution with parameter lambda, ...

        self.time = status[0]

        # coordination change and update vehicle information
        num_vehicles_in_area = [0 for _ in range(MAP_DIVIDE ** 2)]
        vehicles_in_each_area = [[] for _ in range(MAP_DIVIDE**2)]
        distance_to_each_area = [[0. for _ in range(MAP_DIVIDE ** 2)] for __ in range(NUMBER_OF_VEHICLES)]

        vehicles_should_update = [False for _ in range(NUMBER_OF_VEHICLES)]

        vehicle_last_state = []
        # should save in this order: status, location, rebalance to, rebalance start time,
        # pickup start time, get pickup at rebalance, last stay time

        for i in range(NUMBER_OF_VEHICLES):
            loc = self.coordinate_change('TO_MODEL', status[1][i][1])
            this_status = status[1][i][2]  # this status has the type RoboTaxiStatus.XXX
            if self.fleet[i].status == DRIVEWITHCUSTOMER and this_status is RoboTaxiStatus.STAY:
                a = 1
            vehicle_last_state.append(
                [self.fleet[i].status, self.fleet[i].loc, self.fleet[i].rebalanceTo, self.fleet[i].rebalanceStartTime,
                 self.fleet[i].pickupStartTime, self.fleet[i].getPickupAtRebalance, self.fleet[i].lastStayTime])
            self.fleet[i].update(loc, this_status, self.time)
            if self.fleet[i].status == STAY or self.fleet[i].status == REBALANCE:
                num_vehicles_in_area[self.fleet[i].area] += 1
                vehicles_in_each_area[self.fleet[i].area].append(i)
            # for j in range(MAP_DIVIDE ** 2):
            #     distance_to_each_area[i][j] = self.fleet[i].get_distance_to(MID_POINTS[j][0], MID_POINTS[j][1])

        # Process Requests
        # Process Request Distribution & open requests
        open_requests = []  # this saves open requests' labels & ori. position
        # add
        for request in status[2]:
            this_location = self.coordinate_change('TO_MODEL', request[2])
            if request[0] <= self.numRequestSeen:
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
        open_requests_info_in_area = [[] for _ in range(MAP_DIVIDE**2)]
        for req in open_requests:
            my_area = which_area(req[1][0], req[1][1])
            open_requests_in_area[my_area] += 1
            open_requests_info_in_area[my_area].append(req)

        while self.history_requests and self.history_requests[0][0] < self.time - self.keep_history_time:
            self.history_requests.pop(0)

        # Update history request on the map
        request_distribution = [[0 for _ in range(4)] for __ in range(MAP_DIVIDE ** 2)]
        time_slot = [max(0, self.time - self.keep_history_time / 4), max(0, self.time - 3 * self.keep_history_time / 4),
                     max(self.time - 2 * self.keep_history_time / 4, 0), max(self.time - self.keep_history_time / 4, 0)]
        time_flag = 0
        for his_request in self.history_requests:
            while his_request[1] < time_slot[time_flag] and time_flag < 3:
                time_flag += 1
            request_distribution[his_request[1]][time_flag]+= 1

        # get all vehicles that should update action
        update_areas = self.areas_to_handle_requests(open_requests_in_area)
        for i in range(MAP_DIVIDE ** 2):
            if update_areas[i]:
                for j in vehicles_in_each_area[i]:
                    vehicles_should_update[j] = True
        for i in range(NUMBER_OF_VEHICLES):
            if not vehicles_should_update[i] and self.should_update_individual(self.fleet[i], vehicle_last_state[i][0]):
                vehicles_should_update[i] = True

        states = [[] for _ in range(5)]
        for i in range(NUMBER_OF_VEHICLES):
            if vehicles_should_update[i]:
                states[3].append(i)
                curr_area = self.fleet[i].area
                states[4].append(curr_area)
                update_area2D = [convert_area(curr_area, i, '1D', '2D') for i in range(9)]
                open_req_for_this_vehicle = [0 for _ in range(9)]
                his_req_for_this_vehicle = [[0 for _ in range(4)] for _ in range(9)]
                n_vehicles_for_this_vehicle = [0 for _ in range(9)]

                for i in range(9):
                    area = update_area2D[i]
                    if 0 <= area[0] < MAP_DIVIDE and 0 <= area[1] < MAP_DIVIDE:
                        area1D = convert_area(area, None, '2D', '1D')
                        open_req_for_this_vehicle[i] = open_requests_in_area[area1D]
                        his_req_for_this_vehicle[i] = request_distribution[area1D]
                        n_vehicles_for_this_vehicle[i] = num_vehicles_in_area[area1D]
                    else:
                        # -1: Illegal Region
                        open_req_for_this_vehicle[i] = -1
                        his_req_for_this_vehicle[i] = [-1, -1, -1, -1]
                        n_vehicles_for_this_vehicle[i] = -1
                states[0].append(open_req_for_this_vehicle)
                states[1].append(his_req_for_this_vehicle)
                states[2].append(n_vehicles_for_this_vehicle)

        vehicles_should_get_rewards = [False] * NUMBER_OF_VEHICLES
        for i in range(NUMBER_OF_VEHICLES):
            vehicles_should_get_rewards[i] = self.should_get_reward(self.fleet[i], vehicle_last_state[i][0])

        # # update s', r
        # for i in range(NUMBER_OF_VEHICLES):
        #     if self.fleet[i].flagStateChange == 1:
        #         self.fleet[i].data[2] = [] # match the variable state
        #         self.fleet[i].data[3] = reward
        #         memory.push(*tuple(self.fleet[i].data))
        #
        # # remove ?
        # return num_vehicles_in_area, distance_to_each_area, request_distribution, open_requests

        # Stats for [GLOBAL_DIVIDE x GLOBAL_DIVIDE] regions
        open_requests_global = [0 for _ in range(GLOBAL_DIVIDE ** 2)]
        request_distribution_global = [[0 for _ in range(4)] for __ in range(GLOBAL_DIVIDE ** 2)]
        num_vehicles_in_area_global = [0 for _ in range(GLOBAL_DIVIDE ** 2)]

        for region_code in range(MAP_DIVIDE ** 2):
            global_area_code = self.smallRegionToGlobalRegion[region_code]
            open_requests_global[global_area_code] += open_requests_in_area[region_code]
            for i in range(4):
                request_distribution_global[global_area_code][i] += request_distribution[region_code][i]

            num_vehicles_in_area_global[global_area_code] += num_vehicles_in_area[region_code]

        # Append stats of Global_divide to states
        open_requests_to_states = []
        open_requests_global_to_states = []
        num_vehicles_in_area_global_to_states = []

        statistics = [open_requests_global, request_distribution_global, num_vehicles_in_area_global]

        return states, open_requests_info_in_area, vehicles_should_get_rewards, vehicle_last_state, statistics

    def optimize_model(self, saved_actions, rewards):
        # this function trains the model with decay factor GAMMA
        # if len(memory) < BATCH_SIZE:
        #     return
        # transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
        #                               dtype=torch.uint8)
        # non_final_next_states = torch.cat([s for s in batch.next_state
        #                                    if s is not None])
        # open_req_last_batch = torch.stack(batch.open_req_last, 0)
        # num_veh_last_batch = torch.stack(batch.num_veh_last, 0)
        # his_req_last_batch = torch.stack(batch.his_req_last, 0)
        # action_batch = torch.cat(batch.action)
        # reward_batch = torch.cat(batch.reward)
        # open_req_new_batch = torch.stack(batch.open_req_new, 0)
        # num_veh_new_batch = torch.stack(batch.num_veh_new, 0)
        # his_req_new_batch = torch.stack(batch.his_req_new, 0)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        #state_action_values = self.policy_net(open_req_last_batch, num_veh_last_batch, his_req_last_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        #next_state_values = self.target_net(open_req_new_batch, num_veh_new_batch, his_req_new_batch).max(1)[0].detach()
        # Compute the expected Q values
        #expected_state_action_values = (next_state_values.view(-1, 1) * GAMMA) + reward_batch

        # FOR A2C


        # Compute Huber loss
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        # if self.time % 720 == 0:
        #     print("Loss:", loss.item())

        # for A2C net
        # episode, episode_reward = a2c_sample_episode()
        self.optimizer.zero_grad()
        actor_loss, critic_loss = self.a2c_compute_losses(saved_actions, rewards)
        loss = actor_loss + critic_loss
        loss.backward(retain_graph=True)
        # for param in self.a2c_model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        print("Loss:", loss.item())

        # Optimize the model
        # self.optimizer.zero_grad()
        # loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # self.optimizer.step()

    def reward_compute(self, vehicle, old_state):
        """
        This function computes the reward given the last state and current state
        :param vehicle: the vehicle to get reward. Type: Vehicle
        :param old_state: a constant representing the old state.
        :return: a score in double.
        """

        assert isinstance(vehicle, Vehicle)
        assert isinstance(old_state, int)

        reward = None
        if old_state == REBALANCE and (vehicle.status == STAY or vehicle.status == REBALANCE):  # end of rebalance: give deduction
            reward = (self.time - vehicle.rebalanceStartTime) * DISTANCE_COST + vehicle.penalty_for_not_pickup_for_this_time
        if vehicle.status == STAY and old_state == DRIVEWITHCUSTOMER:
            reward = PICKUP_REWARD + vehicle.penalty_for_not_pickup_for_this_time
            if vehicle.getPickupAtRebalance:
                reward += (vehicle.pickupEndTime - vehicle.rebalanceStartTime) * DISTANCE_COST
            else:
                reward += (vehicle.pickupEndTime - vehicle.pickupStartTime) * DISTANCE_COST
        if old_state == STAY:
            reward = 0 + vehicle.penalty_for_not_pickup_for_this_time
        if reward is None:
            raise ValueError('reward wrong')
        else:
            vehicle.penalty_for_not_pickup_for_this_time = vehicle.penalty_for_not_pickup_for_next_time
            vehicle.penalty_for_not_pickup_for_next_time = 0
        return reward

    def coordinate_change(self, direction, loc):
        if direction == 'TO_MODEL':
            if not (self.lngMin <= loc[0] <= self.lngMax and self.latMin <= loc[1] <= self.latMax):
                print(direction, loc)
                warnings.warn('Illegal location! Change to min/max reachable position')
                # Error handler
                if loc[0] < self.lngMin:
                    loc[0] = self.lngMin
                elif loc[0] > self.lngMax:
                    loc[0] = self.lngMax
                if loc[1] < self.latMin:
                    loc[1] = self.latMin
                elif loc[1] > self.latMax:
                    loc[1] = self.latMax
            return [(loc[0] - self.lngMin) / self.unitLongitude, (loc[1] - self.latMin) / self.unitLatitude]
        elif direction == 'TO_COMMAND':
            assert 0 <= loc[0] <= GRAPHMAXCOORDINATE and 0 <= loc[1] <= self.lat_scale
            converted  = [loc[0] * self.unitLongitude + self.lngMin, loc[1] * self.unitLatitude + self.latMin]
            if (converted[0] < self.lngMin or converted[0] > self.lngMax or converted[1] < self.latMin or converted[1] > self.latMax):
                raise ValueError
            return converted
        else:
            raise ValueError

    def areas_to_handle_requests(self, open_requests_in_area):
        areas = [False for _ in range(NUMBER_OF_VEHICLES ** 2)]
        for area1D in range(MAP_DIVIDE ** 2):
            if open_requests_in_area[area1D] > 0:
                area_code2D = [area1D // MAP_DIVIDE, area1D % MAP_DIVIDE]
                nine_regions = [convert_area(area_code2D, i, '2D', '2D') for i in range(9)]
                for area in nine_regions:
                    if 0 <= area[0] < MAP_DIVIDE and 0 <= area[1] < MAP_DIVIDE:
                        areas[area[0] * MAP_DIVIDE + area[1]] = True
        return areas

    def should_update_individual(self, vehicle, last_state):
        assert isinstance(vehicle, Vehicle)
        assert isinstance(last_state, int)
        if last_state == REBALANCE and vehicle.status == STAY:
            return True
        if last_state == DRIVEWITHCUSTOMER and vehicle.status == STAY:
            return True
        if last_state == STAY and vehicle.status == STAY and vehicle.lastStayTime - self.time >= STAY_TIMEOUT:
            return True
        return False

    def should_get_reward(self, vehicle, last_state):
        assert isinstance(vehicle, Vehicle)
        assert isinstance(last_state, int)
        if vehicle.last_action is None:
            return False
        if last_state == STAY and vehicle.status == STAY and vehicle.lastStayTime - self.time >= STAY_TIMEOUT:
            return True
        if last_state == REBALANCE and vehicle.status == STAY:
            return True
        if last_state == DRIVEWITHCUSTOMER and vehicle.status == STAY:
            return True
        else:
            return False

    def select_action(self, open_req, num_veh, his_req):
        sample = torch.randn(open_req.size(0))
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1  # need to change to global variable
        mask = (sample > eps_threshold).long()
        with torch.no_grad():
            actions = self.policy_net(open_req, num_veh, his_req).max(1)[1]
        actions_greedy = torch.randint_like(actions, 0, 19)
        actions = (mask * actions + (torch.ones_like(mask) - mask) * actions_greedy).to(self.device)
        return actions
        # torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    def a2c_select_action(self, open_req, num_veh, his_req, open_req_global, num_veh_global, his_req_global):
    #state = torch.from_numpy(state).float()
    # SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        probs, state_value = self.a2c_model(open_req, num_veh, his_req, open_req_global, num_veh_global, his_req_global)
        m = Categorical(probs)
        actions = m.sample()
        saved_actions = []
        for i in range(actions.size(0)):
            saved_actions.append(SavedAction(m.log_prob(actions[i])[i], state_value[i]))
        return actions, saved_actions


    def a2c_compute_losses(self, saved_actions, rewards):
        ####### TODO #######
        #### Compute the actor and critic losses
        eps = np.finfo(np.float32).eps.item()
        actor_losses, critic_losses = [], []
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            actor_losses.append((-log_prob * advantage))
            critic_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        actor_loss = torch.stack(actor_losses).sum()
        critic_loss = torch.stack(critic_losses).sum()

        return actor_loss, critic_loss
