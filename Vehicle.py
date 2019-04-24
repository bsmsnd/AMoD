import math
from utils.RoboTaxiStatus import RoboTaxiStatus
from constant import *
from generic import *


class Vehicle:
    def __init__(self):
        self.loc = [0., 0.]  # [longitude --> (0,100), latitude --> (0, ?)]
        self.status = STAY
        self.area = 0
        self.request = 0  # 1 -- has requests
        self.rebalanceTo = 0  # to which area
        self.rebalanceStartTime = 0
        self.rebalanceArrivalTime = 0
        self.pickupStartTime = 0
        self.pickupEndTime = 0
        self.getPickupAtRebalance = False
        # self.data = [None]*4 # [s, a, s', r]  # Now changed to last_state(s), action(a)
        self.last_state = None  # update in 'of' (not in pre-process)
        self.last_action = None
        self.penalty_for_not_pickup_for_this_time = 0
        self.penalty_for_not_pickup_for_next_time = 0
        self.lastStayTime = 0
        
        # FOR A2C
        self.a2c_reward = []
        self.a2c_saved_actions = [] 

    def get_distance_to(self, x, y):
        return math.sqrt((x - self.loc[0]) ** 2 + (y - self.loc[1]) ** 2)

    def update(self, new_loc, new_status, time):
        self.loc = new_loc
        self.area = which_area(self.loc[0], self.loc[1])
        self.change_status(new_status, time)

    def change_status(self, new_status, time):
        # Reset some parameters as well
        if new_status is RoboTaxiStatus.STAY:
            if self.status is REBALANCE:
                self.rebalanceArrivalTime = time
            self.status = STAY
        if new_status is RoboTaxiStatus.DRIVETOCUSTOMER:
            self.status = DRIVETOCUSTOMER
        if new_status is RoboTaxiStatus.DRIVEWITHCUSTOMER:
            if self.status == DRIVETOCUSTOMER:
                self.pickupEndTime = time
            self.status = DRIVEWITHCUSTOMER
        if new_status is RoboTaxiStatus.REBALANCEDRIVE:
            self.status = REBALANCE

    def update_stay(self, time):
        self.lastStayTime = time
        self.last_action = 0  # in action space, 0 = stay
        # self.status = STAY

    def update_rebalance(self, time, rebalance_to, cmd):
        self.rebalanceTo = rebalance_to
        self.rebalanceStartTime = time
        self.last_action = cmd


        # self.status = REBALANCE
