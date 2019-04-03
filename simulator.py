# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:48:42 2019

@author: sunhu
"""

from constant import NUMBER_OF_VEHICLES
import DispatchingLogic 
import random
import numpy as np
import matplotlib.pyplot as plt

lon = [0.0, 0.05]
lat = [0.0, 0.05]
bottomLeft = [lon[0], lat[0]]  # 5km by 5km
topRight = [lon[1], lat[1]]
std_num_request = 1 
num_request = 0
request_dic = {}
speed_initial = 40 # km/h
speed = speed_initial/(3600 * 111.3196)


class vehicle:
    def __init__(self):
        self.loc = [0., 0.]
        self.status = 'STAY'
        self.destination = [0., 0.]
        self.destination_custome = [0., 0.]
    
    def pick_up(self, req_infor):
        self.status = 'DRIVETOCUSTOMER'
        self.destination = req_infor[2]
        self.destination_custome = req_infor[3]
    
    def rebalance(self, dest):
        self.status = 'REBALANCEDRIVE'
        self.destination = dest
    
    def state(self): return self.status
    
fleet = [vehicle() for _ in range(NUMBER_OF_VEHICLES)]    
    

def cal_dis(ori, des):
    return np.sqrt((ori[0]-des[0])**2 + (ori[1]-des[1])**2)
    
def cal_time(ori, des):
    global speed
    return cal_dis(ori, des)/speed
    

def generate_request(req, all_req, time):
    global num_request
    global request_dis
    num_b = abs(int(round(np.random.normal(0,std_num_request))))
    if num_b == 0: return req
    for i in range(num_b):
        ori_location = [random.uniform(lon[0], lon[1]), random.uniform(lat[0], lat[1])]
        dest_location = [random.uniform(lon[0], lon[1]), random.uniform(lat[0], lat[1])]
        req_time = np.random.uniform(time-10, time)
        req_temp = [num_request, req_time, ori_location, dest_location]
        req.append(req_temp)
        request_dic[num_request] = req_temp
        num_request += 1
    return req


def fleet_update(action):
    global fleet
    global speed
    pickup, rebalance = action[0], action[1]
    for pick in pickup:
        vehicle_ID, request_ID = pick[0], pick[1]
        fleet[vehicle_ID].pick_up(request_dic[request_ID])
    for rebal in rebalance:
        vehicle_ID, destination = rebal[0], rebal[1]
        fleet[vehicle_ID].rebalance(destination)
    for i in range(len(fleet)):
        veh = fleet[i]
        if veh.state() == 'STAY': 
            continue
        elif veh.state() == 'DRIVETOCUSTOMER' or veh.state() == 'DRIVEWITHCUSTOMER' or veh.state() ==  'REBALANCEDRIVE':
            pre_time = cal_time(veh.loc, veh.destination)
            if pre_time < 10:
                if veh.state() == 'DRIVEWITHCUSTOMER' or veh.state() ==  'REBALANCEDRIVE':
                    fleet[i].status = 'STAY'
                    fleet[i].loc = veh.destination
                else:
                    loc, des = veh.destination, veh.destination_custome
                    ratio = (10-pre_time)*speed/cal_dis(loc, des)
                    fleet[i].status = 'DRIVEWITHCUSTOMER'
                    fleet[i].destination = veh.destination_custome
            else:      
                    loc, des = veh.loc, veh.destination
                    ratio = pre_time*speed/cal_dis(loc, des)
            new_loc = [loc[0] + ratio * (des[0]-loc[0]), 
                       loc[1] + ratio * (des[1]-loc[1])]
            fleet[i].loc = new_loc
        else:
            print('Error with vehicle state')

def plot():
    global fleet
    pass

if __name__ == "__main__":
    time = 0
    all_req = []
    req = []
    dispatch = DispatchingLogic.DispatchingLogic(bottomLeft, topRight)
    while True:
        time += 10
        state_vehicle = [[i, time, fleet[i].state(), 1] for i in range(NUMBER_OF_VEHICLES)]
        req = generate_request(req, num_request, time)
        action = dispatch.of([time, state_vehicle, req, [0,0,0]])
        fleet_update(action)
            
        
        
        
        
    