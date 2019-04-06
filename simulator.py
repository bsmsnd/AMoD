# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:48:42 2019

@author: sunhu
"""

from constant import NUMBER_OF_VEHICLES
import DispatchingLogic_demo 
import random
import numpy as np
import matplotlib.pyplot as plt
from utils.RoboTaxiStatus import RoboTaxiStatus
import sys


# Initialize map
# map 0.05 degree ~ 5.5km 
lon = [0.0, 0.05]  # longitude range of the map 
lat = [0.0, 0.05]  # latitude range of the map
bottomLeft = [lon[0], lat[0]]  
topRight = [lon[1], lat[1]]

# Initialize request
std_num_request = 0.5  # variance for new request per 10 second
num_request = 0  # count the total number of request   
request_dic = {}  # save all the information about the request
req = [] # list of openrequest

# Initialize vehicle speed
speed_initial = 40 # km/h
speed = speed_initial/(3600 * 111.3196)

#
time = 0


class vehicle:
    def __init__(self):
        self.loc = [random.uniform(lon[0], lon[1]), random.uniform(lat[0], lat[1])]
        self.status = RoboTaxiStatus('STAY')
        self.destination = [0., 0.]
        self.destination_custome = [0., 0.]
    
    def pick_up(self, req_infor):
        if self.status is RoboTaxiStatus.DRIVEWITHCUSTOMER:
            raise ValueError('Can not change the state from DRIVEWITHCUSTOMER to pick up')
            sys.exit(1)
        self.status = RoboTaxiStatus('DRIVETOCUSTOMER')
        self.destination = req_infor[2]
        self.destination_custome = req_infor[3]
    
    def rebalance(self, dest):
        if self.status is RoboTaxiStatus.DRIVEWITHCUSTOMER:
            raise ValueError('Can not change the state from DRIVEWITHCUSTOMER to Rebalance')
            sys.exit(1)
        self.status = RoboTaxiStatus('REBALANCEDRIVE')
        self.destination = dest
    def state(self): return self.status

fleet = [vehicle() for _ in range(NUMBER_OF_VEHICLES)]    


def cal_dis(ori, des):
    return np.sqrt((ori[0]-des[0])**2 + (ori[1]-des[1])**2)


def cal_time(ori, des):
    # calculate the time from original location to the destination location
    global speed
    return cal_dis(ori, des)/speed
    

def generate_request():
    ############Generate random request#################
    # The number of request every 10 second is abs(gaussian) distribution with mean 0
    # variance std_num_request
    # The location of custome and destination are uniformly distributed in the whole map 
    # output: add new request to the global varialble req and 
    ####################################################
    global num_request
    global request_dic
    global req
    global time
    num_b = abs(int(round(np.random.normal(0,std_num_request))))
    if num_b == 0: return
    for i in range(num_b):
        ori_location = [random.uniform(lon[0], lon[1]), random.uniform(lat[0], lat[1])]
        dest_location = [random.uniform(lon[0], lon[1]), random.uniform(lat[0], lat[1])]
        req_time = np.random.uniform(time-10, time)
        req_temp = [num_request, req_time, ori_location, dest_location]
        req.append(req_temp)
        request_dic[num_request] = req_temp
        num_request += 1
    return




def fleet_update(action):
    ####################update state#######################
    # input: action from our netowrk
    # update: update all the vehicles state and generate new state after 10 second
    # delete open requse which has been resolved 
    #######################################################
    global fleet
    global speed
    global req
    pickup, rebalance = action[0], action[1]
    delete_dic = {}
    # update vehicle state for pick up
    for pick in pickup:
        vehicle_ID, request_ID = pick[0], pick[1]
        fleet[vehicle_ID].pick_up(request_dic[request_ID])
        delete_dic[request_ID] = 1
    
    # delete these open request which has been resolved  
    index = 0
    while True:
        if (index >= len(req)): 
            break
        if req[index][0] in delete_dic:
            del req[index]
            continue
        index += 1
    
    # update vehicle state to rebalance 
    for rebal in rebalance:
        vehicle_ID, destination = rebal[0], rebal[1]
        fleet[vehicle_ID].rebalance(destination)
    
    # generate new motion after 10 seconds basing on the vehicle state
    for i in range(len(fleet)):
        veh = fleet[i]
        print(veh.status)
        if veh.status is RoboTaxiStatus.STAY: 
            continue
        elif (veh.state() is RoboTaxiStatus.DRIVETOCUSTOMER or 
              veh.state() is RoboTaxiStatus.DRIVEWITHCUSTOMER 
              or veh.state() is RoboTaxiStatus.REBALANCEDRIVE):
            pre_time = cal_time(veh.loc, veh.destination)
            if pre_time < 10:  
            # there is a state change in the 10 second
                if (veh.state() is RoboTaxiStatus.DRIVEWITHCUSTOMER or 
                    veh.state() is RoboTaxiStatus.REBALANCEDRIVE):
                    fleet[i].status = RoboTaxiStatus.STAY
                    fleet[i].loc = veh.destination
                    continue
                else:
                    loc, des = veh.destination, veh.destination_custome
                    ratio = (10-pre_time)*speed/cal_dis(loc, des)
                    fleet[i].status = RoboTaxiStatus.DRIVEWITHCUSTOMER
                    fleet[i].destination = veh.destination_custome
            else:
            # there is no state change in the 10 second
                    loc, des = veh.loc, veh.destination
                    ratio = 10*speed/cal_dis(loc, des)
            # update new location        
            new_loc = [loc[0] + ratio * (des[0]-loc[0]), 
                       loc[1] + ratio * (des[1]-loc[1])]
            fleet[i].loc = new_loc
        else:
            raise ValueError('Error with vehicle state')
            sys.exit(1)

def plot():
    ###########Generate plot for the system################
    global fleet
    pass


if __name__ == "__main__":       
    dispatch = DispatchingLogic_demo.DispatchingLogic(bottomLeft, topRight)
    while True:
        time += 10
        state_vehicle = [[i, fleet[i].loc, fleet[i].state(), 1] for i in range(NUMBER_OF_VEHICLES)]
        generate_request()
#        print(state_vehicle)
        action = dispatch.of([time, state_vehicle, req, [0,0,0]])
#        print(action)
        fleet_update(action)
            
        
        
        
        
    