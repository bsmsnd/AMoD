from constant import *

def which_area(x, y):
    # this function determines the area for location (x,y)
    # x,y should be coordinates in model.
    lngArea = int(x // (GRAPHMAXCOORDINATE / MAP_DIVIDE))
    latArea = int(y // (GRAPHMAXCOORDINATE / MAP_DIVIDE))
    return MAP_DIVIDE * latArea + lngArea

def arg_max(x):
    # find the index of the max value for an iterable
    return max(enumerate(x), key=lambda x: x[1])[0]

def arg_min(x):
    # find the index of the max value for an iterable
    return min(enumerate(x), key=lambda x: x[1])[0]