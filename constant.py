BATCH_SIZE = 128  # Some hyper-parameters to adjust!
GRAPHMAXCOORDINATE = 100  # This sets the max. coordinate of
NUMBER_OF_VEHICLES = 20  # Change AidoGuest.py as well!
MAP_DIVIDE = 5  # Set this value to change the grid of the map N x N

global LNG_SCALE
LNG_SCALE = 100

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
DISTANCE_COST = -0.05

STAY_TIMEOUT = 60
N_FEATURE = 27
N_ACTION = 19

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

ILLEGAL_AREA = -1
R_ILLEGAL = -1000