import math
from utils.RoboTaxiStatus import RoboTaxiStatus
from constant import *
from generic import *

class Vehicle:
    def __init__(self):
        self.loc = [0., 0.]  # [longitude, latitude]
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

        self.lastStayTime = 0


    def get_distance_to(self, x, y):
        return math.sqrt((x - self.loc[0]) ** 2 + (y - self.loc[1]) ** 2)

    def update(self, new_loc, new_status, time):
        self.loc = new_loc
        self.area = which_area(self.loc[0], self.loc[1])
        self.change_Status(new_status)

    def change_Status(self, new_status, time):
        # TODO: NEED TO RESET SOME PARAMS

        # flag will be 1 if Rebalance -> not Rebalance or STAY ->

        if self.status==REBALANCE and new_status is not RoboTaxiStatus.REBALANCEDRIVE:
            self.flagStateChange = 1
        if self.status==STAY:
            self.flagStateChange = 1
        if new_status is RoboTaxiStatus.STAY:
            if self.status is REBALANCE:
                self.rebalanceArrivalTime = time
            self.status = STAY
        if new_status is RoboTaxiStatus.DRIVETOCUSTOMER:
            self.status = DRIVETOCUSTOMER
        if new_status is RoboTaxiStatus.DRIVEWITHCUSTOMER:
            if self.status == DRIVETOCUSTOMER:
                self.pickupEndTime
            self.status = DRIVEWITHCUSTOMER
        if new_status is RoboTaxiStatus.REBALANCEDRIVE:
            self.status = REBALANCE

    def update_stay(self, time):
        self.lastStayTime = time
        self.status = STAY

    def update_rebalance(self, time, rebalance_to):
        self.rebalanceTo = rebalance_to
        self.rebalanceStartTime = time
        self.status = REBALANCE

