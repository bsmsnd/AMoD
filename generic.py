from constant import *
from utils.RoboTaxiStatus import RoboTaxiStatus


def which_area(x, y):
    # this function determines the area for location (x,y)
    # x,y should be coordinates in model.
    lng_area = int(x // (GRAPHMAXCOORDINATE / MAP_DIVIDE))
    lat_area = int(y // (GRAPHMAXCOORDINATE / MAP_DIVIDE))
    return MAP_DIVIDE * lat_area + lng_area


def arg_max(x):
    # find the index of the max value for an iterable
    return max(enumerate(x), key=lambda x: x[1])[0]


def arg_min(x):
    # find the index of the max value for an iterable
    return min(enumerate(x), key=lambda x: x[1])[0]


def vehicle_status_converter(status, direction):
    # values for direction: 'TO_MODEL' or 'TO_HOST'
    if direction == 'TO_MODEL':
        if status == RoboTaxiStatus.STAY:
            return STAY
        elif status == RoboTaxiStatus.DRIVEWITHCUSTOMER:
            return DRIVEWITHCUSTOMER
        elif status == RoboTaxiStatus.DRIVETOCUSTOMER:
            return DRIVETOCUSTOMER
        elif status == RoboTaxiStatus.REBALANCEDRIVE:
            return REBALANCE
        else:
            print('illegal vehicle status, direction = TO_MODEL')
            raise ValueError
    elif direction == 'TO_HOST':
        if status == STAY:
            return RoboTaxiStatus.STAY
        elif status == DRIVETOCUSTOMER:
            return RoboTaxiStatus.DRIVETOCUSTOMER
        elif status == DRIVEWITHCUSTOMER:
            return RoboTaxiStatus.DRIVEWITHCUSTOMER
        elif status == REBALANCE:
            return RoboTaxiStatus.REBALANCEDRIVE
        else:
            print('illegal vehicle status, direction = TO_HOST')
            raise ValueError
    else:
        print('illegel direction: should be TO_HOST or TO_MODEL')
        raise ValueError
