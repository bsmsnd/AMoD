import random
import math
import scipy.optimize as op
from utils.RoboTaxiStatus import RoboTaxiStatus

PROB_GLOBAL_REBALANCE = 0.5

class DispatchingLogic:
    """
    dispatching logic in the AidoGuest demo to compute dispatching instructions that are forwarded to the AidoHost
    """
    def __init__(self, bottomLeft, topRight):
        """
        :param bottomLeft: {lngMin, latMin}
        :param topRight: {lngMax, latMax}
        """
        self.lngMin = bottomLeft[0]
        self.lngMax = topRight[0]
        self.latMin = bottomLeft[1]
        self.latMax = topRight[1]


        print("minimum longitude in network: ", self.lngMin)
        print("maximum longitude in network: ", self.lngMax)
        print("minimum latitude  in network: ", self.latMin)
        print("maximum latitude  in network: ", self.latMax)

        # Example:
        # minimum longitude in network: -71.38020297181387
        # maximum longitude in network: -70.44406349551404
        # minimum latitude in network: -33.869660953686626
        # maximum latitude in network: -33.0303523690584

        self.matchedReq = set()
        self.matchedTax = set()

    def of(self, status):
        assert isinstance(status, list)

        pickup = []
        rebalance = []

        time = status[0]
        if time % 60 == 0:  # every minute
            # index = 0

            # sort requests according to submission time
            requests = sorted(status[2].copy(), key=lambda request: request[1])

            # store all vehicles in STAY mode
            stay_vehicle = []
            for vehicle in status[1]:
                # vehicle: [label, location, status, 1]
                if vehicle[2] is RoboTaxiStatus.STAY or vehicle[2] is RoboTaxiStatus.REBALANCEDRIVE:
                    stay_vehicle.append(vehicle)

            # store unmatched (unresponded) requests
            unmatched_requests = []
            for request in requests:
                if request[0] not in self.matchedReq:
                    unmatched_requests.append(request)


            # Calculate distances between requests and vehicles in STAY mode
            dist_table = [[[] for _ in range(len(stay_vehicle))] for _ in range(len(unmatched_requests))]    # Req * Vehicle
            for i in range(len(unmatched_requests)):
                for j in range(len(stay_vehicle)):
                    request = unmatched_requests[i]
                    vehicle = stay_vehicle[j]
                    dist_table[i][j] = self.get_distance(request[2][0], request[2][1], vehicle[1][0], vehicle[1][1])


            if dist_table:
                row, col = op.linear_sum_assignment(dist_table)
                # row: number of request in requests; col: number of vehicle in stay_vehicle
                for i in range(len(row)):
                    pickup.append([stay_vehicle[col[i]][0], unmatched_requests[row[i]][0]])
                    self.matchedReq.add(unmatched_requests[row[i]][0])
                    self.matchedTax.add(stay_vehicle[col[i]][0])


            for roboTaxi in status[1]:
                if roboTaxi[2] is RoboTaxiStatus.STAY and roboTaxi[0] not in self.matchedTax:
                    rebalance_flag = random.uniform(0, 1)
                    if rebalance_flag > PROB_GLOBAL_REBALANCE:
                        rebalanceLocation = self.getRandomRebalanceLocation()
                        rebalance.append([roboTaxi[0], rebalanceLocation])

            self.matchedTax = set()

            '''
            # for each unassigned request, add a taxi in STAY mode
            for request in requests:
                if request[0] not in self.matchedReq:
                    while index < len(status[1]):
                        roboTaxi = status[1][index]
                        if roboTaxi[2] is RoboTaxiStatus.STAY:
                            pickup.append([roboTaxi[0], request[0]]) # [which vehicle, which request]
                            self.matchedReq.add(request[0])
                            self.matchedTax.add(roboTaxi[0])
                            index += 1
                            break
                        index += 1
            '''

            '''
            # rebalance 1 of the remaining and unmatched STAY taxis
            for roboTaxi in status[1]:
                if roboTaxi[2] is RoboTaxiStatus.STAY and roboTaxi[0] not in self.matchedTax:
                    rebalanceLocation = self.getRandomRebalanceLocation()
                    rebalance.append([roboTaxi[0], rebalanceLocation])
                    break
            '''

        return [pickup, rebalance]

    def get_distance(self, req_x, req_y, vehi_x, vehi_y):
        return math.sqrt((req_x - vehi_x) ** 2 + (req_y - vehi_y) ** 2)


    def getRandomRebalanceLocation(self):
        """
        ATTENTION: AMoDeus internally uses the convention (longitude, latitude) for a WGS:84 pair, not the other way
        around as in some other cases.
        """
        return [random.uniform(self.lngMin, self.lngMax),
                random.uniform(self.latMin, self.latMax)]