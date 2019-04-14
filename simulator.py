# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:48:42 2019

@author: sunhu
"""

from constant import NUMBER_OF_VEHICLES
from DispatchingLogic import DispatchingLogic 
import random
import numpy as np
import matplotlib.pyplot as plt
from utils.RoboTaxiStatus import RoboTaxiStatus
import sys
import matplotlib.pyplot as plt
import datetime
import os
import json
import pickle


time_p = 0
# Initialize map
# map 0.05 degree ~ 5.5km 
lon = [0.0, 0.1]  # longitude range of the map 
lat = [0.0, 0.05]  # latitude range of the map
bottomLeft = [lon[0], lat[0]]  
topRight = [lon[1], lat[1]]

# Initialize request
std_num_request = 0.3  # variance for new request per 10 second
num_request = 0  # count the total number of request   
request_dic = {}  # save all the information about the request
# all the index of request that have been responsed   
# but the vehicle is drivingtocustome
request_wait = []  
# list of openrequest
req = [] 

# Initialize vehicle speed
speed_initial = 30 # km/h
speed = speed_initial/(3600 * 111.3196)

# constant for plot and save 
flag_plot_enable = False
flag_save_enable = False
plot_period = 40
save_period = 20
pause_time = 0.01
curDT = datetime.datetime.now()

extend = '.txt'
filename = str(curDT.year) + '_' + str(curDT.month) + '_' + str(curDT.day)+ '_' 
filename = filename + str(curDT.hour)+ '_' + str(curDT.minute) + '_' + str(curDT.second)
filename = os.path.join('log', filename)


class vehicle:
    def __init__(self):
        self.loc = [random.uniform(lon[0], lon[1]), random.uniform(lat[0], lat[1])]
        self.status = RoboTaxiStatus('STAY')
        self.destination = [0., 0.]
        self.destination_custome = [0., 0.]
        self.requestID = -1
    
    def pick_up(self, req_infor):
        if self.status is RoboTaxiStatus.DRIVEWITHCUSTOMER:
            raise ValueError('Can not change the state from DRIVEWITHCUSTOMER to pick up')
            sys.exit(1)
        self.status = RoboTaxiStatus('DRIVETOCUSTOMER')
        self.destination = req_infor[2]
        self.destination_custome = req_infor[3]
        self.requestID = req_infor[0]
    
    def rebalance(self, dest):
        if self.status is RoboTaxiStatus.DRIVEWITHCUSTOMER:
            raise ValueError('Can not change the state from DRIVEWITHCUSTOMER to Rebalance')
            sys.exit(1)
        self.status = RoboTaxiStatus('REBALANCEDRIVE')
        self.destination = dest
        self.requestID = -1
        
    def state(self): return self.status

fleet = [vehicle() for _ in range(NUMBER_OF_VEHICLES)]   

fleet_save = [[[0,0],'Stay'] for i in range(NUMBER_OF_VEHICLES)]

def RoboToStr(x):
    if (x.status is RoboTaxiStatus.DRIVETOCUSTOMER):
        return 'DRIVETOCUSTOMER'
    elif (x.status is RoboTaxiStatus.DRIVEWITHCUSTOMER):
        return 'DRIVEWITHCUSTOMER'
    elif (x.status is RoboTaxiStatus.STAY):
        return 'STAY'
    else:
        return 'REBALANCEDRIVE'
    


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
    global time_p
    num_b = abs(int(round(np.random.normal(0,std_num_request))))
    if num_b == 0: return
    for i in range(num_b):
        ori_location = [random.uniform(lon[0], lon[1]), random.uniform(lat[0], lat[1])]
        dest_location = [random.uniform(lon[0], lon[1]), random.uniform(lat[0], lat[1])]
        req_time = np.random.uniform(time_p-10, time_p)
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
    global fleet_save
    global speed
    global req
    global request_wait
    global num_request
#    global request_dic
    pickup, rebalance = action[0], action[1]
    delete_dic = {}
    # update vehicle state for pick up
    for pick in pickup:
        vehicle_ID, request_ID = pick[0], pick[1]
        if vehicle_ID >= NUMBER_OF_VEHICLES:
            raise ValueError("vehicle_ID exceed NUMBER_OF_VEHICLES", vehicle_ID)
        if request_ID >= num_request:
            raise ValueError("request_ID exceed NUMBER_OF_VEHICLES", request_ID)
        fleet[vehicle_ID].pick_up(request_dic[request_ID])
        delete_dic[request_ID] = 1
        request_wait.append(request_ID)
    
    
    # update vehicle state to rebalance 
    for rebal in rebalance:
        vehicle_ID, destination = rebal[0], rebal[1]
        if destination[0]<lon[0] or destination[0] > lon[1]:
            print("HHHH")
            raise ValueError("lon exceed")
        if destination[1]<lat[0] or destination[1] > lat[1]:
            print("HHHH")
            raise ValueError("lat exceed")
        fleet[vehicle_ID].rebalance(destination)
    
    # generate new motion after 10 seconds basing on the vehicle state
    for i in range(len(fleet)):
        veh = fleet[i]
#        print(veh.status)
        fleet_save[i] = [veh.loc, RoboToStr(veh)]
        if veh.status is RoboTaxiStatus.STAY: 
            continue
        elif (veh.status is RoboTaxiStatus.DRIVETOCUSTOMER or 
              veh.status is RoboTaxiStatus.DRIVEWITHCUSTOMER 
              or veh.status is RoboTaxiStatus.REBALANCEDRIVE):
            pre_time = cal_time(veh.loc, veh.destination)
            if pre_time < 10:  
            # there is a state change in the 10 second
                if (veh.status is RoboTaxiStatus.DRIVEWITHCUSTOMER or 
                    veh.status is RoboTaxiStatus.REBALANCEDRIVE):
                    fleet[i].status = RoboTaxiStatus.STAY
                    fleet[i].loc = veh.destination
                    if (veh.status is RoboTaxiStatus.DRIVEWITHCUSTOMER):
                        fleet[i].requestID = -1
                    continue
                else:
                    # the state is changing from drivetocustome to drivewithcustome
                    loc, des = veh.destination, veh.destination_custome
                    ratio = (10-pre_time)*speed/cal_dis(loc, des)
                    fleet[i].status = RoboTaxiStatus.DRIVEWITHCUSTOMER
                    fleet[i].destination = veh.destination_custome
                    request_wait.remove(veh.requestID)
#                    if (veh.requestID != -1):
#                        delete_dic[veh.requestID] = 1
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
    
    
    # delete these open request which has been resolved  
    index = 0
    while True:
        if (index >= len(req)): 
            break
        if req[index][0] in delete_dic:
            del req[index]
            continue
        index += 1
        

def plot():
    ###########Generate plot for the system################
    # '*' means request, large red * : open request   
    # small blue *: request are responsed
    # 'o' means vehicle
    # green 'o' : stay or rebalance 
    # magenta 'o': drive to custome
    # red 'o': drive with custome
    global fleet
    global req
    global request_dic
    global request_wait
    global pause_time
    plt.clf()
    for i in range(len(req)):
        plt.plot(req[i][2][0], req[i][2][1], marker='*', markersize=17, color="r")
    for i in range(len(request_wait)):
        req_temp = request_dic[request_wait[i]][2]
        plt.plot(req_temp[0], req_temp[1], marker='*', markersize=10, color="b")
        
    for i in range(len(fleet)):
        if fleet[i].status is RoboTaxiStatus.REBALANCEDRIVE:
            plt.plot(fleet[i].loc[0], fleet[i].loc[1], marker='o', markersize=6, color="y")
        elif fleet[i].status is RoboTaxiStatus.STAY:
            plt.plot(fleet[i].loc[0], fleet[i].loc[1], marker='o', markersize=6, color="y")
        elif fleet[i].status is RoboTaxiStatus.DRIVETOCUSTOMER:
            plt.plot(fleet[i].loc[0], fleet[i].loc[1], marker='o', markersize=6, color="m")
        elif fleet[i].status is RoboTaxiStatus.DRIVEWITHCUSTOMER:
            plt.plot(fleet[i].loc[0], fleet[i].loc[1], marker='o', markersize=6, color="r")
    plt.axis([lon[0], lon[1], lat[0], lat[1]])
    plt.pause(pause_time)
    

def save():
    ###########Save the information to a text file#############
    # One txt file contain all the fleet, open-request, responded-request information
    # Another file contain a dictionary of all the 
    global time_p
    global fleet_save
    global req
    global request_dic
    global request_wait
    global filename
    global extend
    temp_data = [fleet_save, req, request_wait]
    fleet_jason = json.dumps(temp_data)
    file = open(filename + extend, "a")
    file.write(fleet_jason+"\n")
    file.close()
    with open(filename+ '_fleet_dic' + '.pkl', "wb") as fp:
        pickle.dump(request_dic, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
        
if __name__ == "__main__":       
    dispatch = DispatchingLogic(bottomLeft, topRight)
    plt.ion()
    if flag_plot_enable:
        plt.figure(1)
    file = open(filename, "w")
    file.close()
    while True:
        time_p += 10
        state_vehicle = [[i, fleet[i].loc, fleet[i].state(), 1] for i in range(NUMBER_OF_VEHICLES)]
        generate_request()
#        print(state_vehicle)
        action = dispatch.of([time_p, state_vehicle, req, [0,0,0]])
#        print(action[0])
        fleet_update(action)
        
        if flag_plot_enable and time_p % plot_period == 0:
            plot()
        if flag_save_enable and time_p % save_period == 0:
            save()
            
        
        
        
    