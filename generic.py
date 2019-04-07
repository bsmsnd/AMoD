from constant import *
from utils.RoboTaxiStatus import RoboTaxiStatus


def which_area(x, y):
    # this function determines the area for location (x,y)
    # x,y should be coordinates in model.
    lng_area = int(x // (GRAPHMAXCOORDINATE / MAP_DIVIDE))  # 0 - 100
    lat_area = int(y // (LNG_SCALE / MAP_DIVIDE))  # 0 - scale
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


def convert_area(location=None, location_delta=None, input_form='1D', output_form='2D'):
    """
    This function convert location(area code) into the other form.
    :param location: the location to convert, 1D or 2D specified by input_form
    :param location_delta: the location to add to original location (1 - 9); Use None if no delta term
    :param input_form: either '1D' or '2D'
    :param output_form: either '1D' or '2D'
    :return: the location in the form specified by output_form
    """
    if input_form == '1D':
        assert isinstance(location, int)
        ori_2d = _convert_1d_to_2d(location)
        if location_delta is None:
            location_delta = 4
        assert 0 <= location_delta < 9
        new_2d = [ori_2d[0] + NINE_REGIONS[location_delta][0], ori_2d[1] + NINE_REGIONS[location_delta][1]]
        bad_point_flag = False
        if new_2d[0] < 0 or new_2d[0] >= MAP_DIVIDE or new_2d[1] < 0 or new_2d[1] >= MAP_DIVIDE:
           bad_point_flag = True
        if output_form == '1D':
            if bad_point_flag:
                return ILLEGAL_AREA
            else:
                return _convert_2d_to_1d(new_2d)
        elif output_form == '2D':
            if bad_point_flag:
                return [ILLEGAL_AREA,ILLEGAL_AREA]
            else:
                return new_2d
        else:
            raise ValueError('output_form should be either "1D" or "2D')
    elif input_form == '2D':
        if (not isinstance(location, list)) or (not isinstance(location[0], int)) or (not isinstance(location[1], int)):
            raise ValueError()
        if location[0] < 0 or location[0] >= MAP_DIVIDE or location[1] < 0 or location[1] >= MAP_DIVIDE:
            raise ValueError('Location in 2D value error. Should be within 0 and %d' % MAP_DIVIDE)
        if location_delta is None:
            location_delta = 4
        assert 0 <= location_delta < 9
        new_2d = [location[0] + NINE_REGIONS[location_delta][0], location[1] + NINE_REGIONS[location_delta][1]]
        bad_point_flag = False
        if new_2d[0] < 0 or new_2d[0] >= MAP_DIVIDE or new_2d[1] < 0 or new_2d[1] >= MAP_DIVIDE:
            bad_point_flag = True
        if output_form == '1D':
            if bad_point_flag:
                return ILLEGAL_AREA
            else:
                return _convert_2d_to_1d(new_2d)
        elif output_form == '2D':
            if bad_point_flag:
                return [ILLEGAL_AREA,ILLEGAL_AREA]
            else:
                return new_2d
        else:
            raise ValueError('output_form should be either "1D" or "2D')


def _convert_1d_to_2d(loc):
    """
    This function converts 1D location to 2D
    :param loc: a location in 1D
    :return: a list of 2 numbers, [x, y]
    """
    return [loc // MAP_DIVIDE, loc % MAP_DIVIDE]


def _convert_2d_to_1d(loc):
    """
    This function converts 2D location to 1D
    :param loc: a location in 2D
    :return: a integer indicating its area code
    """
    return loc[0] * MAP_DIVIDE + loc[1]