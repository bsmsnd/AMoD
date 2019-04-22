GRAPHMAXCOORDINATE = 100  # This sets the max. coordinate of
NUMBER_OF_VEHICLES = 20  # Change AidoGuest.py as well!
MAP_DIVIDE = 3  # Set this value to change the grid of the map N x N
GLOBAL_DIVIDE = 2
LEARNING_RATE = 1e-6
global LAT_SCALE
# LAT_SCALE = 100

# Vehicle Status
STAY = 0
DRIVETOCUSTOMER = 1
DRIVEWITHCUSTOMER = 2
REBALANCE = 3


# MID_POINTS = [[GRAPHMAXCOORDINATE / MAP_DIVIDE / 2 + GRAPHMAXCOORDINATE / MAP_DIVIDE * (n % MAP_DIVIDE),
#                LAT_SCALE / MAP_DIVIDE / 2 + LAT_SCALE / MAP_DIVIDE * (n // MAP_DIVIDE)] for n in
#               range(MAP_DIVIDE ** 2)]

NINE_REGIONS = [[1, -1], [1, 0], [1, 1], [0, -1], [0, 0], [0, 1], [-1, -1], [-1, 0], [-1, 1]]

# Set Hyper-parameters
PICKUP_REWARD = 100
DISTANCE_COST = -0.01
NO_PICKUP_PENALTY = -500

STAY_TIMEOUT = 20

N_FEATURE = 39
# change the number of action from 19 to 19+4
N_ACTION = 23

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 10

ILLEGAL_AREA = -1
R_ILLEGAL = -1000
SAVE_PERIOD = 3600
LOAD_FLAG = False
SAVE_FLAG = False
SAVE_PATH = './weight/dqn_weight.pt'
LOAD_PATH = './weight/dqn_weight.pt'
PRINT_REWARD_PERIOD = 120
MEMORY_SIZE = 5000

SLIDE_WIN_SIZE = 15000
const_bound = 1

