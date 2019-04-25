import random
import math
import scipy.optimize as op
from utils.RoboTaxiStatus import RoboTaxiStatus
from generic import *
from constant import *
import warnings
from distance_on_unit_sphere import *


PROB_LOCAL_REBALANCE = 0.5

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

        self.unitLongitude = (self.lngMax - self.lngMin) / GRAPHMAXCOORDINATE
        self.map_width = distance_on_unit_sphere(self.latMin, self.lngMin, self.latMin, self.lngMax)
        self.map_length = distance_on_unit_sphere(self.latMax, self.lngMax, self.latMin, self.lngMax)
        self.lat_scale = GRAPHMAXCOORDINATE * self.map_length / self.map_width
        self.unitLatitude = (self.latMax - self.latMin) / self.lat_scale
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

            '''
                9-Surroundings Greedy Model
            '''
            # TODO: Combine store and assign
            # store unmatched (unresponded) requests into related area
            requests_each_area = [[] for _ in range(MAP_DIVIDE ** 2)]
            response_order = set()
            for request in requests:
                if request[0] not in self.matchedReq:
                    location = self.coordinate_change("TO_MODEL", request[2])
                    area = self.which_area(location[0], location[1])
                    requests_each_area[area].append(request)
                    if area not in response_order:
                        response_order.add(area)

            # Store vehicles in stay mode into areas
            stay_vehicle_each_area = [[] for _ in range(MAP_DIVIDE**2)]
            for vehicle in status[1]:
                # vehicle: [label, location, status, 1]
                if vehicle[2] is RoboTaxiStatus.STAY:
                    # TODO: Calculate area number given real location
                    location = self.coordinate_change("TO_MODEL", vehicle[1])
                    area = self.which_area(location[0], location[1])
                    stay_vehicle_each_area[area].append(vehicle)

            # Calculate distance table in each area
            # Apply op.linear_sum_assignment (Hungary Algorithm)
            # requests_each_area, stay_vehicle_each_area
            for area in response_order:
                if not requests_each_area[area]:
                    continue
                dist_table, nine_region_vehicles = self.calculateDistance(area, requests_each_area[area],
                                                                          stay_vehicle_each_area)
                if dist_table:
                    row, col = op.linear_sum_assignment(dist_table)
                    for i in range(len(row)):
                        request = requests_each_area[area][row[i]]
                        vehicle = nine_region_vehicles[col[i]]
                        pickup.append([vehicle[0], request[0]])
                        self.matchedReq.add(request[0])
                        self.matchedTax.add(vehicle[0])
                        # IMPORTANT: pop the responded label
                        for k in range(len(stay_vehicle_each_area)):
                            if not stay_vehicle_each_area[k]:
                                continue
                            if vehicle in stay_vehicle_each_area[k]:
                                stay_vehicle_each_area[k].pop(stay_vehicle_each_area[k].index(vehicle))

            for roboTaxi in status[1]:
                if roboTaxi[2] is RoboTaxiStatus.STAY and roboTaxi[0] not in self.matchedTax:
                    location = self.coordinate_change("TO_MODEL", roboTaxi[1])
                    area = self.which_area(location[0], location[1])
                    rebalance_flag = random.uniform(0, 1)
                    if rebalance_flag > PROB_LOCAL_REBALANCE:
                        rebalanceLocation = self.getRandomRebalanceLocation(area, MAP_DIVIDE)
                        rebalance.append([roboTaxi[0], rebalanceLocation])

            self.matchedTax = set()

            '''
            # rebalance 1 of the remaining and unmatched STAY taxis
            for roboTaxi in status[1]:
                if roboTaxi[2] is RoboTaxiStatus.STAY and roboTaxi[0] not in self.matchedTax:
                    rebalanceLocation = self.getRandomRebalanceLocation()
                    rebalance.append([roboTaxi[0], rebalanceLocation])
                    break
            '''

            '''
            # Global Greedy Model
            # store all vehicles in STAY mode
            stay_vehicle = []
            for vehicle in status[1]:
                # vehicle: [label, location, status, 1]
                if vehicle[2] is RoboTaxiStatus.STAY:
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
            '''


            '''
            # Original Version
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


    def getRandomRebalanceLocation(self, area, MAP_DIVIDE):
        """
        ATTENTION: AMoDeus internally uses the convention (longitude, latitude) for a WGS:84 pair, not the other way
        around as in some other cases.
        """
        LNG_DIVIDE = (self.lngMax - self.lngMin) / MAP_DIVIDE
        LAT_DIVIDE = (self.latMax - self.latMin) / MAP_DIVIDE
        if area == 0:
            return [random.uniform(self.lngMin, self.lngMin + 2 * LNG_DIVIDE),
                    random.uniform(self.latMin, self.latMin + 2 * LAT_DIVIDE)]
        elif area == MAP_DIVIDE - 1:
            return [random.uniform(self.lngMax - 2 * LNG_DIVIDE, self.lngMax),
                    random.uniform(self.latMin, self.latMin + 2 * LAT_DIVIDE)]
        elif area == MAP_DIVIDE * (MAP_DIVIDE - 1):
            return [random.uniform(self.lngMin, self.lngMin + 2 * LNG_DIVIDE),
                    random.uniform(self.latMax - 2 * LAT_DIVIDE, self.latMax)]
        elif area == MAP_DIVIDE ** 2 - 1:
            return [random.uniform(self.lngMax - 2 * LNG_DIVIDE, self.lngMax),
                    random.uniform(self.latMax - 2 * LAT_DIVIDE, self.latMax)]
        elif area % MAP_DIVIDE == 0:
            k = area // MAP_DIVIDE
            return [random.uniform(self.lngMin, self.lngMin + 2 * LNG_DIVIDE),
                    random.uniform(self.latMin + (k-1) * LAT_DIVIDE, self.latMin + (k+2) * LAT_DIVIDE)]
        elif 0 < area < MAP_DIVIDE - 1:
            return [random.uniform(self.lngMin + (area-1) * LNG_DIVIDE, self.lngMin + (area+2) * LNG_DIVIDE),
                    random.uniform(self.latMin, self.latMin + 2 * LAT_DIVIDE)]
        elif area % MAP_DIVIDE == MAP_DIVIDE - 1:
            k = area // MAP_DIVIDE
            return [random.uniform(self.lngMax - 2 * LNG_DIVIDE, self.lngMax),
                    random.uniform(self.latMin + (k-1) * LAT_DIVIDE, self.latMin + (k+2) * LAT_DIVIDE)]
        elif 0 < area % MAP_DIVIDE < MAP_DIVIDE - 1:
            k = area % MAP_DIVIDE
            return [random.uniform(self.lngMin + (k-1) * LNG_DIVIDE, self.lngMin + (k+2) * LNG_DIVIDE),
                    random.uniform(self.latMax - 2 * LAT_DIVIDE, self.latMax)]
        else:
            k = area % MAP_DIVIDE
            return [random.uniform(self.lngMin + (k-1) * LNG_DIVIDE, self.lngMin + (k+2) * LNG_DIVIDE),
                    random.uniform(self.latMin + (k-1) * LAT_DIVIDE, self.latMin + (k+2) * LAT_DIVIDE)]


    def calculateDistance(self, area, requests, stay_vehicles):
        """
        :param area: Area Number (0~MAP_DIVIDE**2)
        :param requests: requests in area
        :param stay_vehicles: stay_vehicle_each_area
        :return: distance table
        """
        if area == 0:
            vehicles = []
            for vehicle in stay_vehicles[area]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+MAP_DIVIDE+1]: vehicles.append(vehicle)
        elif area == MAP_DIVIDE-1:
            vehicles = []
            for vehicle in stay_vehicles[area-1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area-1+MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+MAP_DIVIDE]: vehicles.append(vehicle)
        elif area == MAP_DIVIDE * (MAP_DIVIDE - 1):
            vehicles = []
            for vehicle in stay_vehicles[area-MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area-MAP_DIVIDE+1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+1]: vehicles.append(vehicle)
        elif area == MAP_DIVIDE ** 2 - 1:
            vehicles = []
            for vehicle in stay_vehicles[area-1-MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area-MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area-1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area]: vehicles.append(vehicle)
        elif area % MAP_DIVIDE == 0:
            vehicles = []
            for vehicle in stay_vehicles[area-MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area-MAP_DIVIDE+1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+MAP_DIVIDE+1]: vehicles.append(vehicle)
        elif 0 < area < MAP_DIVIDE-1:
            vehicles = []
            for vehicle in stay_vehicles[area-1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area-1+MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+MAP_DIVIDE+1]: vehicles.append(vehicle)
        elif 0 < area % MAP_DIVIDE < MAP_DIVIDE-1:
            vehicles = []
            for vehicle in stay_vehicles[area - 1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area + 1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area - 1 - MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area - MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area - MAP_DIVIDE + 1]: vehicles.append(vehicle)
        elif area % MAP_DIVIDE == MAP_DIVIDE - 1:
            vehicles = []
            for vehicle in stay_vehicles[area - 1 - MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area - MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area - 1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area + MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area + MAP_DIVIDE - 1]: vehicles.append(vehicle)
        else:
            vehicles = []
            for vehicle in stay_vehicles[area-1-MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area-MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+1-MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area-1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+1]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area-1+MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+MAP_DIVIDE]: vehicles.append(vehicle)
            for vehicle in stay_vehicles[area+1+MAP_DIVIDE]: vehicles.append(vehicle)

        dist_table = [[[] for _ in range(len(vehicles))] for _ in range(len(requests))]    # Req * Vehicle
        for i in range(len(requests)):
            for j in range(len(vehicles)):
                request = requests[i]
                vehicle = vehicles[j]
                dist_table[i][j] = self.get_distance(request[2][0], request[2][1], vehicle[1][0], vehicle[1][1])
        return dist_table, vehicles


    def coordinate_change(self, direction, loc):
        if direction == 'TO_MODEL':
            if not (self.lngMin <= loc[0] <= self.lngMax and self.latMin <= loc[1] <= self.latMax):
                print(direction, loc)
                warnings.warn('Illegal location! Change to min/max reachable position')
                # Error handler
                if loc[0] < self.lngMin:
                    loc[0] = self.lngMin
                elif loc[0] > self.lngMax:
                    loc[0] = self.lngMax
                if loc[1] < self.latMin:
                    loc[1] = self.latMin
                elif loc[1] > self.latMax:
                    loc[1] = self.latMax
            return [(loc[0] - self.lngMin) / self.unitLongitude, (loc[1] - self.latMin) / self.unitLatitude]
        elif direction == 'TO_COMMAND':
            assert 0 <= loc[0] <= GRAPHMAXCOORDINATE and 0 <= loc[1] <= self.lat_scale
            converted = [loc[0] * self.unitLongitude + self.lngMin, loc[1] * self.unitLatitude + self.latMin]
            if (converted[0] < self.lngMin or converted[0] > self.lngMax or converted[1] < self.latMin or
                    converted[1] > self.latMax):
                raise ValueError
            return converted
        else:
            raise ValueError

    def which_area(self, x, y):
        # this function determines the area for location (x,y)
        # x,y should be coordinates in model.
        lng_area = int(x // (GRAPHMAXCOORDINATE / MAP_DIVIDE))  # 0 - 100
        lat_area = int(y // (self.lat_scale / MAP_DIVIDE))  # 0 - scale
        return MAP_DIVIDE * lat_area + lng_area