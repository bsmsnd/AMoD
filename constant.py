BATCH_SIZE = 128  # Some hyper-parameters to adjust!
GRAPHMAXCOORDINATE = 100
NUMBER_OF_VEHICLES = 20  # Change AidoGuest.py as well!
MAP_DIVIDE = 5
LNG_SCALE = 100  # This value should be reset in init ONLY ONCE

# Vehicle Status
STAY = 0
DRIVETOCUSTOMER = 1
DRIVEWITHCUSTOMER = 2
REBALANCE = 3

MID_POINTS = [[GRAPHMAXCOORDINATE / MAP_DIVIDE / 2 + GRAPHMAXCOORDINATE / MAP_DIVIDE * (n % MAP_DIVIDE),
               LNG_SCALE / MAP_DIVIDE / 2 + LNG_SCALE / MAP_DIVIDE * (n // MAP_DIVIDE)] for n in
              range(MAP_DIVIDE ** 2)]

NINE_REGIONS = [[-1, 1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]

# Set Hyper-parameters
PICKUP_REWARD = 100
DISTANCE_COST = -5

STAY_TIMEOUT = 60