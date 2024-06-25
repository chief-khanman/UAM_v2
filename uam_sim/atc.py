'''
Future: Use GNNs to send pre collision advisory to uavs.

Currently, as uavs move, when another uav or buildings detection zones intersect,
uavs start collision avoidance procedure. 

My plan is to use GNN which will take 
current state - all uavs position, speed, current_heading, ref_final_heading as input, I do see uavs in airspace as nodes of graph, and buildings as static nodes,
if GNNs can be used for studying graph evolution, I believe I can use it for evolving the current state of the system, 
to see if there will be collision in the near future (need to define what I mean by near future - as in how many minutes according to clock time)
Now the forward and backward process, 

The forward process is to detect the collision by using evolution of graph system,
this forward pass needs to produce collision or no collision, within future time horizon,
If collision detected;
The Backward pass needs to calculate what should be the correction heading for uavs that have collided in future. 
'''
import numpy as np
import random
from shapely import Point
import shapely
from geopandas import GeoSeries
from typing import List, Dict
from airspace import Airspace
from vertiport import Vertiport
from uav import UAV
from uav_basic import UAV_Basic
from autonomous_uav import Autonomous_UAV
from uam_sim.cas import CollisionController
import copy
#from autonomous_uav import Autonomous_UAV


class ATC:

    def __init__(self, airspace:Airspace, ):
        '''ATC (Air Traffic Controller) - maintains information on UAVs and Vertiports.

        Args:
            airspace (Airspace): The airspace object associated with the ATC.

        Attributes:
            airspace (Airspace): The airspace object associated with the ATC.
            reg_uav_list (List[UAV]): The list of registered UAVs.
            vertiports_in_airspace (List[Vertiport]): The list of vertiports in the airspace.
        '''

        self.airspace = airspace
        self.reg_uav_list:List[UAV_Basic] = [] #:List[UAV]
        self.vertiports_in_airspace:List[Vertiport] = [] #:List[Vertiport]
        #self.controller = controller
        
    
    
    #* This method needs to be run once to initialize the sim 
    def create_n_random_vertiports(self, num_vertiports):
        """
        Creates a specified number of random vertiports within the airspace.

        Args:
            num_vertiports (int): The number of vertiports to create.

        Returns:
            None

        Side Effects:
            - Creates the vertiports and updates the vertiports in the airspace list.
        """
        
        # Need to remove the hospital regions first so that vertiports are not sampled from hospital and buffer zones
        sample_space = self.airspace.location_utm_gdf.iloc[0,0].difference(self.airspace.location_utm_hospital_buffer.unary_union)
        sample_space_gdf = GeoSeries(sample_space)
        sample_vertiport:GeoSeries = sample_space_gdf.sample_points(num_vertiports)
        sample_vertiport_array:np.ndarray = shapely.get_parts(sample_vertiport[0])

        for location in sample_vertiport_array:
            self.vertiports_in_airspace.append(Vertiport(location=location, uav_list=[])) # location-> shapely.Point
    
    #TODO - break the task of creating UAVs and assigning start end vertiport 
    #* This method needs to be run once to initialize the sim 
    def create_n_reg_uavs(self, num_uavs, ):
        """
        Creates a specified number of regular UAVs and assigns them random start and end vertiports.

        Args:
            num_uavs (int): The number of UAVs to create.

        Returns:
            None
        """
        
        start_vertiport_list = copy.deepcopy(self.vertiports_in_airspace)
        end_vertiport_list = copy.deepcopy(self.vertiports_in_airspace)

        for _ in range(num_uavs):
            uav_start_vertiport = random.choice(start_vertiport_list)
            uav_end_vertiport = random.choice(end_vertiport_list)
            while uav_start_vertiport.location == uav_end_vertiport.location:
                uav_end_vertiport = random.choice(end_vertiport_list)
            uav = UAV_Basic(uav_start_vertiport, uav_end_vertiport)
            start_vertiport_list.pop(start_vertiport_list.index(uav_start_vertiport))
            end_vertiport_list.pop(end_vertiport_list.index(uav_end_vertiport))
            self.reg_uav_list.append(uav)

            
    
    def _vertiport_filtering(self, some_vertiport):
        '''Internal method. Used for selecting end vertiports for UAVs,
           such that at the beginning of the simulation,
           all UAVs have different vertiports.

        Parameters:
        some_vertiport (Vertiport): The vertiport to filter out.

        Returns:
        list: A list of filtered vertiports.
        '''
        filtered_vertiport = []
        for vertiport in self.vertiports_in_airspace:
            if vertiport != some_vertiport:
                filtered_vertiport.append(vertiport)
        return filtered_vertiport
    

    def has_reached_end_vertiport(self, uav:UAV_Basic):
        '''Checks if a UAV has reached its end_vertiport.
        
        This method checks if a UAV has reached its end_vertiport. If it did reach,
        it calls the landing_procedure method to update relevant objects.
        
        Args:
            uav (UAV): The UAV object to check.
        
        Returns:
            None
        '''
        
        if (uav.current_position.distance(uav.end_point) <= uav.landing_proximity) and (uav.reaching_end_vertiport == False):
            #uav.reached_end_vertiport = True
            self._landing_procedure(uav)
    
    
    def has_left_start_vertiport(self, uav: UAV_Basic):
        '''Checks if a UAV has left its start_vertiport.

        This method checks if a UAV has left its start_vertiport. If it did leave,
        then it calls the clearing_procedure to take care of updating objects.

        Args:
            uav (UAV): The UAV object to check.

        Returns:
            None
        '''
        if (uav.current_position.distance(uav.start_point) > 100) and (uav.leaving_start_vertiport == False):
            self._clearing_procedure(uav)
            uav.leaving_start_vertiport = True

    
 
    def provide_vertiport(self, ) -> Vertiport:
        '''
        Randomly selects and returns a vertiport from the list of available vertiports.
        This method facilitates the allocation of vertiports by choosing one at random from the `vertiports_in_airspace` list.
        It is useful for scenarios requiring a random vertiport selection without specific criteria.

        Returns:
            Vertiport: A Vertiport object selected randomly from the available vertiports.
        '''
        sample_vertiport = random.choice(self.vertiports_in_airspace)
        return sample_vertiport
    
    
    def _reassign_end_vertiport_of_uav(self, uav:UAV_Basic):
        '''Reassigns the end vertiport of a UAV.
        
        This method samples a vertiport from the ATC vertiport list. 
        If the sampled vertiport is the same as the UAV's current start_vertiport, it resamples until a different vertiport is obtained. 
        The sampled end_vertiport is then assigned as the UAV's end_vertiport. 
        Finally, the UAV's end_point is updated.
        
        Args:
            uav (UAV): The UAV object for which the end vertiport needs to be reassigned.
        '''
        sample_end_vertiport = self.provide_vertiport()
        while sample_end_vertiport.location == uav.start_vertiport.location:
            sample_end_vertiport = self.provide_vertiport()
        uav.end_vertiport = sample_end_vertiport
        uav.update_end_point()
    

    def _update_start_vertiport_of_uav(self, vertiport:Vertiport, uav:UAV_Basic):
        '''This method accepts a vertiport (end-vertiport of uav)
        and updates the start_vertiport attribute of UAV
        to the provided vertiport. This method works in conjunction with landing_procedure.

        Args:
            vertiport (Vertiport): The vertiport representing the end-vertiport of the UAV.
            uav (UAV): The UAV whose start_vertiport attribute needs to be updated.

        Returns:
            None

        '''
        uav.start_vertiport = vertiport
        uav.update_start_point()
    

    def _landing_procedure(self, landing_uav:UAV_Basic):
        '''
        Performs the landing procedure for a given UAV.
        Args:
            landing_uav (UAV): The UAV that is landing.
        Returns:
            None
        Raises:
            None
        '''
        landing_vertiport = landing_uav.end_vertiport
        landing_vertiport.uav_list.append(landing_uav) 
        landing_uav.refresh_uav()
        self._reassign_end_vertiport_of_uav(landing_uav)

    

    def _clearing_procedure(self, outgoing_uav:UAV_Basic): #! rename to _takeoff_procedure()
        '''
        Performs the clearing procedure for a given UAV.
        Args:
            outgoing_uav (UAV): The UAV that is outgoing(leaving the start_vertiport).
        Returns:
            None
        Raises:
            None
        '''
        outgoing_uav_id = outgoing_uav.id
        for uav in outgoing_uav.start_vertiport.uav_list:
            if uav.id == outgoing_uav_id:
                outgoing_uav.start_vertiport.uav_list.remove(uav)

    
    def set_start_end_uav(self, list_uav_airspace):
        '''Assign start-end point to all uavs in airspace'''
        pass


    def create_auto_uav(self,) -> Autonomous_UAV:
        #! need to provide the start and end vertiport
        self.auto_uav = Autonomous_UAV()

    def create_n_uavs(self, percent_auto):
        '''This method will create a mix of smart and regular uavs.
        The mix is controlled by percentage argument.'''
        pass

    def create_vertiport_at_location(self, position):
        '''Create a vertiport at position.
            Position is Point type, and 
            if position not within location, will return 0'''
        pass
    
        





    

     
    
   



    
   



         
    
    

    
