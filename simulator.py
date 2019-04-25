# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:48:42 2019

@author: sunhu
"""

from constant import NUMBER_OF_VEHICLES
from DispatchingLogic import DispatchingLogic  # To change, please also change the import in generic.py
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
import scipy.stats


time_p = 0
# Initialize map
# map 0.05 degree ~ 5.5km 
lon = [0.0, 0.3]  # longitude range of the map 
lat = [0.0, 0.2]  # latitude range of the map
bottomLeft = [lon[0], lat[0]]  
topRight = [lon[1], lat[1]]

# Initialize request
std_num_request = 0.25  # variance for new request per 10 second
num_request = 0  # count the total number of request   
flag_dist_enable = False
time_trafic= [9,18]
var_trafic = [1,1]
alpha = 0.55
loc_house = [[0.02, 0.01], [0.02, 0.015], [0.07, 0.045], [0.08, 0.01]]
loc_downtown = [[0.04, 0.035]]
loc_static_distri = [[0.2,0.1]]
request_dic = {}  # save all the information about the request
# all the index of request that have been responsed   
# but the vehicle is drivingtocustome
request_wait = []  
# list of openrequest
req = [] 

# Initialize vehicle speed
speed_initial = 64.37 # km/h  40mile/h
speed = speed_initial/(3600 * 111.3196)

# constant for plot and save 
flag_plot_enable = True
flag_save_enable = False
plot_period = 30
save_period = 20
pause_time = 0.01
curDT = datetime.datetime.now()

extend = '.txt'
filename = str(curDT.year) + '_' + str(curDT.month) + '_' + str(curDT.day)+ '_' 
filename = filename + str(curDT.hour)+ '_' + str(curDT.minute) + '_' + str(curDT.second)
filename = os.path.join('log', filename)

# average waiting time 
win_size = 500  # the sliding window size for average waiting time 
wait_time = []
wait_time_sum = 0
wait_time_total = 0
wait_time_count = 0


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
    
def get_localtion(loc_):
    global lon
    global lat
    index = np.random.randint(len(loc_))
    lon_current = [loc_[index][0]-0.01, loc_[index][0]+0.01]
    if (lon_current[0]<lon[0]): lon_current[0] = lon[0]
    if (lon_current[1]>lon[1]): lon_current[1] = lon[1]
    
    lat_current = [loc_[index][1]-0.01, loc_[index][1]+0.01]
    if (lat_current[0]<lat[0]): lat_current[0] = lat[0]
    if (lat_current[1]>lat[1]): lat_current[1] = lat[1]

    return lon_current, lat_current



def generate_request_from_distr(num_distr, loc_1, loc_2):
    global req
    global request_dic
    global num_request
    global time_p
    if num_distr == 0: return  
    for i in range(num_distr):
        lon_current, lat_current = get_localtion(loc_1)
        lon_current_dest, lat_current_dest = get_localtion(loc_2)
        
        ori_location = [random.uniform(lon_current[0], lon_current[1]),
                        random.uniform(lat_current[0], lat_current[1])]
        dest_location = [random.uniform(lon_current_dest[0], lon_current_dest[1]), 
                         random.uniform(lat_current_dest[0], lat_current_dest[1])]
        req_time = np.random.uniform(time_p-10, time_p)
        req_temp = [num_request, req_time, ori_location, dest_location]
        req.append(req_temp)
        request_dic[num_request] = req_temp
        num_request += 1
    return 


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
    global alpha
    global time_trafic
    global var_trafic
    global loc_house
    global loc_downtown
    global loc_static_distri
    time_day = time_p % (24*60*60)
    hour_day = time_day/(60*60)
    pdf1 = scipy.stats.norm(time_trafic[0], var_trafic[0]).pdf(hour_day)
    pdf2 = scipy.stats.norm(time_trafic[1], var_trafic[1]).pdf(hour_day)
    std_distr1 = alpha*pdf1
    std_distr2 = alpha*pdf2
    num_distr1 = abs(int(round(np.random.normal(0,std_distr1))))
    num_distr2 = abs(int(round(np.random.normal(0,std_distr2))))
    num_b = abs(int(round(np.random.normal(0,std_num_request))))
    num_b_ = abs(int(round(np.random.normal(0,std_num_request*3))))
    
    if flag_dist_enable:
        generate_request_from_distr(num_distr1, loc_house, loc_downtown)
        generate_request_from_distr(num_distr2, loc_downtown, loc_house)
    if num_b != 0: 
      for i in range(num_b):
          ori_location = [random.uniform(lon[0], lon[1]), random.uniform(lat[0], lat[1])]
          dest_location = [random.uniform(lon[0], lon[1]), random.uniform(lat[0], lat[1])]
          req_time = np.random.uniform(time_p-10, time_p)
          req_temp = [num_request, req_time, ori_location, dest_location]
          req.append(req_temp)
          request_dic[num_request] = req_temp
          num_request += 1
          
    if num_b_ != 0:
        for i in range(num_b):
          ori_location1 = [random.uniform(loc_static_distri[0][0]-0.03, loc_static_distri[0][0]+0.03), 
                          random.uniform(loc_static_distri[0][1]-0.03, loc_static_distri[0][1]+0.03)]
          dest_location1 = [random.uniform(lon[0], lon[1]), random.uniform(lat[0], lat[1])]
          req_time1 = np.random.uniform(time_p-10, time_p)
          req_temp1 = [num_request, req_time1, ori_location1, dest_location1]
          req.append(req_temp1)
          request_dic[num_request] = req_temp1
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
    global request_dic
    global wait_time
    global wait_time_sum
    global wait_time_count
    global wait_time_total
    global win_size
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
#        delete_dic[request_ID] = 1
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
                    wait_time_p = time_p - request_dic[veh.requestID][1]
                    wait_time.append(wait_time_p)
                    wait_time_sum += wait_time_p
                    wait_time_total += wait_time_p
                    wait_time_count += 1
                    if (len(wait_time)>win_size):
                        wait_time_sum -= wait_time[0]
                        del wait_time[0]
                        
                    if (veh.requestID != -1):
                        delete_dic[veh.requestID] = 1
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
        plt.plot(req[i][2][0], req[i][2][1], marker='*', markersize=25, color="r")
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
    plt.title('%d day %d:%d' % (time_p // 86400, time_p % 86400 // 3600, time_p % 3600 // 60))
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
    average_wait_time = []
    average_wait_time_global = []
    time_passes = []
    if flag_plot_enable:
        plt.figure(1)
    if flag_save_enable:
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
        if time_p % 1800 == 0 and len(wait_time) and wait_time_count:
            print('Total {0} request---- average wait time for {1} request: {2} ----global avg waiting time: {3}'.format(len(request_dic), 
                  len(wait_time), (wait_time_sum / len(wait_time)),(wait_time_total/wait_time_count)))
            time_passes.append(time_p / 60)
            average_wait_time.append(wait_time_sum / len(wait_time))
            average_wait_time_global.append(wait_time_total / wait_time_count)
        if flag_plot_enable and time_p % plot_period == 0:
            plot()
        if flag_save_enable and time_p % save_period == 0:
            save()

        if time_p % (2*24*60*60) == 0:
            break

    plt.figure()
    plt.plot(time_passes, average_wait_time)
    plt.xlabel('time/minute')
    plt.ylabel('average waiting time')
    # plt.show()
    plt.savefig('Time_Average_Greedy_PICK_DQN_REBALANCE.png')
    
    plt.figure()
    plt.plot(time_passes, average_wait_time)
    plt.xlabel('time/minute')
    plt.ylabel('average waiting time/s')
    # plt.show()
    plt.savefig('Time_Average_Global_Greedy_PICK_DQN_REBALANCE.png')
    