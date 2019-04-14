#BATCH_SIZE = 32  # Some hyper-parameters to adjust!
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
DISTANCE_COST = -0.03

STAY_TIMEOUT = 60
N_FEATURE = 27
N_ACTION = 19

EPS_START = 0.3
EPS_END = 0.05
EPS_DECAY = 2000
BATCH_SIZE = 64
GAMMA = 0.999
TARGET_UPDATE = 10

ILLEGAL_AREA = -1
R_ILLEGAL = -1000
SAVE_PERIOD = 3600
LOAD_FLAG = True
SAVE_FLAG = False
SAVE_PATH = './weight/dqn_weight.pt'
LOAD_PATH = './weight/dqn_weight.pt'
PRINT_REWARD_PERIOD = 120
MEMORY_SIZE = 5000

SLIDE_WIN_SIZE = 5000


