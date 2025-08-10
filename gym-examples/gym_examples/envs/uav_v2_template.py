from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import math
import numpy as np
from shapely import Point

from sensor_template import SensorTemplate
from controller_template import ControllerTemplate
from dynamics_template import DynamicsTemplate

class UAV_v2_template(ABC):
    """
    Abstract Base Class (ABC) for a UAV (Unmanned Aerial Vehicle) template.

    This class defines the structure and behavior expected for UAV implementations. 
    It includes attributes and methods for assigning missions, retrieving states, 
    sensing the environment, and controlling UAV actions.

    Attributes:
        id (int): Unique identifier for the UAV instance.
        radius (float): Radius of the UAV's physical body.
        nmac_radius (float): Radius for Near Mid-Air Collision (NMAC) detection.
        detection_radius (float): Radius for detecting other UAVs.
        sensor (SensorTemplate): Sensor object to collect data about the environment.
        dynamics (DynamicsTemplate): Dynamics model governing UAV's motion.
        controller (ControllerTemplate): Controller object for UAV's decision-making.
        mission_complete_distance (float): Distance threshold for mission completion.
        current_speed (float): Current speed of the UAV.
        current_position (Point): Current position of the UAV.
        current_heading (float): Current heading of the UAV in radians.
    """
    
    def __init__(self, controller, dynamics, sensor, radius, nmac_radius, detection_radius):
        """
        Initialize the UAV with its controller, dynamics, sensor, and parameters.

        Args:
            controller (ControllerTemplate): Controller object for decision-making.
            dynamics (DynamicsTemplate): Dynamics object governing motion.
            sensor (SensorTemplate): Sensor object for environment sensing.
            radius (float): Radius of the UAV's physical body.
            nmac_radius (float): NMAC detection radius.
            detection_radius (float): Radius for detecting other UAVs.
        """
        self.id = id(self) 
        self.radius = radius
        self.nmac_radius = nmac_radius
        self.detection_radius = detection_radius
        self.sensor: SensorTemplate = sensor
        self.dynamics: DynamicsTemplate = dynamics
        self.controller: ControllerTemplate = controller
        self.mission_complete_distance = 40 # Increased from 10 to 40 to account for UAV overshotting goal between updates
        self.current_speed = 0
        self.max_speed = 80
        self.max_acceleration = 1 # Passed to DynamicsPointMass for action renormalization
        self.max_heading_change = math.pi # Passed to DynamicsPointMass for action renormalization
        self.rotor_speed = 1 #! this is temp value, we need to find a way to calculate and update this method
        self.nmac_count = 0

    @abstractmethod
    def assign_start_end(self, start: Point, end: Point):
        """
        Assign the start and end points for the UAV's mission.

        Args:
            start (Point): Starting position of the UAV.
            end (Point): Target position of the UAV.
        """
        # TODO: 
        # collect start-end information from vertiport location
        # adjust the arguments to accept two vertiports
        # from each vertiport collect their location, and assign that as start/end point  


        self.mission_complete_status = False
        self.start = start
        self.end = end
        self.current_position = start
        
        # ADDED odometer attr:
        self.previous_position = start
        # TODO: change current heading at start to a random direction. let controller change current heading over time
        # performing this change will have impact on ORCA agent visualization
        # TODO: fix arrow visualization for ORCA agents - the arrow of ORCA agents should get updated as current heading is changes  
        #self.current_heading = math.atan2((end.y - start.y), (end.x - start.x))
        self.current_heading = np.random.uniform(-math.pi, math.pi)
        self.final_heading = math.atan2((end.y - self.current_position.y), (end.x - self.current_position.x))
        self.body = self.current_position.buffer(self.radius)
        return None

    @abstractmethod
    def get_mission_status(self) -> bool:
        """
        Check whether the mission is complete.

        Returns:
            bool: True if the UAV is within the mission completion distance from the target, False otherwise.
        """
        if self.current_position.distance(self.end) <= self.mission_complete_distance:
            mission_complete_status = True
        else:
            mission_complete_status = False
        return mission_complete_status

    @abstractmethod
    def set_mission_complete_status(self, mission_complete_status: bool) -> None:
        """
        Set the mission completion status.

        Args:
            mission_complete_status (bool): True if the mission is complete, False otherwise.
        """
        self.mission_complete_status = mission_complete_status
        return None

    @abstractmethod
    def get_state(self) -> Dict:
        """
        Retrieve the current state of the UAV.

        Returns:
            Dict: A dictionary containing state information such as distance to goal, current speed, heading, and radius.
        """
        ref_prll, ref_orth = self.get_ref()
        return {'id':self.id,
                'current_position':self.current_position,
                'distance_to_end': self.start.distance(self.end),
                'distance_covered': self.start.distance(self.current_position), #TODO: wrong def 
                'distance_to_goal': self.current_position.distance(self.end),
                'max_dist': self.start.distance(self.end),
                'min_dist': 0,
                'min_speed': 0,
                'max_speed': self.max_speed, 
                'current_speed': self.current_speed,
                'current_heading': self.current_heading,
                'final_heading': self.final_heading,
                'end':self.end,
                'radius': self.radius,
                'ref_prll':ref_prll,
                'ref_orth':ref_orth,
                'detection_radius': self.detection_radius
                }

    @abstractmethod
    def get_sensor_data(self) -> Tuple[List, List]:
        """
        Collect data from the sensor about other UAVs and restricted airspace in the environment.

        Returns:
            Tuple[List, List]: Sensor data about other UAVs, and restricted area data.
        """
        
        return self.sensor.get_data(self)

    def get_obs(self) -> Tuple[Dict, Tuple[List, List]]:  
        """
        Retrieve the observation, combining the UAV's state with sensor data.

        Returns:
            Tuple[Dict, Tuple[List, List]]
            Tuple containing Dict which contains own state informati 
            and another Tuple that has two lists, one for other_uavs, and another for restricted airspaces
        """
        
        # get self information
        own_data = self.get_state()
        # add self obs with other_uav obs
        sensor_data = self.get_sensor_data()
        
        return (own_data, sensor_data) # TODO: named tuple would be better for sensor data.

    def get_action(self, observation):
        """
        Retrieve an action based on the given observation.
        The input argument 'observation' should be same as the observation of agent in env. 
        This needs to be maintained in order to perform supervised learning, as pre-training for the AutoUAV.

        Args:
            observation: The input observation used by the controller.

        Returns:
            The action determined by the controller.
        """
        self.action = self.controller(observation=observation)
        return self.action

    def get_ref(self):
        """
        Calculate and return the reference directions (parallel and orthogonal) 
        for the UAV relative to its goal.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Parallel and orthogonal reference directions.
        """
        goal_dir = np.array([self.end.x - self.current_position.x, 
                             self.end.y - self.current_position.y])
        self.dist_to_goal = self.current_position.distance(self.end)

        if self.dist_to_goal > 1e-8:
            ref_prll = goal_dir / self.dist_to_goal
        else:
            ref_prll = goal_dir
        
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])

        return ref_prll, ref_orth
    
    def reset_odometer(self):
        self.odometer_reading = 0
        return None
    
    def update_odometer(self):
        distance = self.current_position.distance(self.previous_position)
        self.odometer_reading += distance
        self.previous_position = self.current_position

        

