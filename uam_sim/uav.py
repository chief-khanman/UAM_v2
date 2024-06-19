#Deterministic uavs
from random import sample as random_sample
import numpy as np
from shapely.geometry import Point
from geopandas import GeoSeries
from vertiport import Vertiport
from uam_sim.cas import CollisionController
#TODO - abstract controller, basic collision controller 
#from collision_avoidance_controller_basic import uav_collision_detection, uav_nmac_detection, static_collision_detection, static_nmac_detection


class UAV:
    '''Representation of UAV in airspace. UAV motion represented in 2D plane. 
     Object is to move from start vertiport to end vertiport.
     A UAV instance requires a start and end vertiport.
     '''
    
    def __init__(self,
                 start_vertiport ,  
                 end_vertiport,
                 landing_proximity = 50., 
                 max_speed = 43,
                 ):
        
        
        #UAV builtin properties 
        self.heading_deg = np.random.randint(-178,178) + np.random.rand() # random heading between -180 and 180
        self.uav_footprint = 17 #H175 nose to tail length of 17m,
        self.nmac_radius = 150 #NMAC radius
        self.detection_radius = 550
        self.uav_footprint_color = 'blue' # this color represents the UAV object 
        self.uav_nmac_radius_color = 'orange'
        self.uav_detection_radius_color = 'green'
        self.uav_collision_controller = None
        
        #UAV technical properties
        self.id = id(self)
        self.current_speed = 0
        self.max_speed:float = max_speed
        self.max_acceleration = 1 # m/s^2, this has been obtained from internet 
        self.landing_proximity = landing_proximity # UAV object considered landed when within this radius
        
        #UAV soft properties
        self.leaving_start_vertiport = False
        self.reaching_end_vertiport = False

        #Vertiport assignement
        self.start_vertiport = start_vertiport
        self.end_vertiport = end_vertiport
        
        #UAV position properties
        self.start_point:Point = self.start_vertiport.location
        self.end_point:Point = self.end_vertiport.location
        self.current_position:Point = self.start_point
        
        #UAV heading properties
        self.current_heading_deg:float = self.heading_deg
        self.current_heading_radians = np.deg2rad(self.current_heading_deg)
        
        #Final heading calculation
        self.current_ref_final_heading_rad = np.arctan2(self.end_point.y - self.current_position.y, 
                                                                    self.end_point.x - self.current_position.x)
        self.current_ref_final_heading_deg = np.rad2deg(self.current_ref_final_heading_rad)


    def uav_polygon(self, dimension):
        return GeoSeries(self.current_position).buffer(dimension).iloc[0]
    
    def uav_polygon_plot(self, dimension):
        return GeoSeries(self.current_position).buffer(dimension)
    
    def refresh_uav(self, ):
        '''Update the start vertiport of a UAV 
        to a new start vertiport, 
        argument of this method '''
        self.leaving_start_vertiport = False
        self.reaching_end_vertiport = False 


    def update_end_point(self,):
        '''Updates the UAV end point using its own end_vertiport location'''
        self.end_point = self.end_vertiport.location

    def update_start_point(self,):
        self.start_point = self.start_vertiport.location

    def speed_controller(self,):

        if self.current_speed == 0:
            acc = self.max_acceleration
        elif self.current_position.distance(self.end_point) <= 500:
            acc = - ((self.max_speed)**2 / (2*500))
        elif self.current_speed <= self.max_speed:
            acc = self.max_acceleration
        else:
            acc = 0
    
        return acc 

    
    def _update_position(self,d_t:float,):
        '''Internal method. Updates current_position of the UAV after d_t seconds.
           This uses a first order Euler's method to update the position.
           '''
        update_x = self.current_position.x + self.current_speed * np.cos(self.current_heading_radians) * d_t 
        update_y = self.current_position.y + self.current_speed * np.sin(self.current_heading_radians) * d_t 
        self.current_position = Point(update_x,update_y)
    

    def _update_speed(self, d_t, acceleration_from_controller):
        '''
        Arg: acceleration_from_controller is an input from controller/das_system
        '''
        base_acc = self.speed_controller()
        '''
        if acceleration from controller, meaning controller is sending some acceleration value,
        then acceleration from controller will over-ride acceleration from speed controller
        
        '''
        if acceleration_from_controller != 0:
            final_acc = acceleration_from_controller
        else:
            final_acc = base_acc
        
        self.current_speed = self.current_speed + (final_acc * d_t)



    def _update_ref_final_heading(self, ): 
        '''Internal method. Updates the heading of the aircraft, pointed towards end_point'''
        self.current_ref_final_heading_rad = np.arctan2(self.end_point.y - self.current_position.y, 
                                                        self.end_point.x - self.current_position.x)
        self.current_ref_final_heading_deg = np.rad2deg(self.current_ref_final_heading_rad)
        
    #TODO - update_theta needs to be reworked NOW  
    def _update_theta_d(self, heading_correction_das_controller = 0): 
        #TODO - update method to include theta_dd and d_t
        '''Internal method. Updates heading of the aircraft, pointed towards ref_final_heading_deg''' 
        if heading_correction_das_controller == 0:
            avg_rate_of_turn = 20 #degree/s, collected from google - https://skybrary.aero/articles/rate-turn#:~:text=Description,%C2%B0%20turn%20in%20two%20minutes.

            #! need to find how to dynamically slow down the turn rate as we get close to the ref_final_heading
            if np.abs(self.current_ref_final_heading_deg - self.current_heading_deg) < avg_rate_of_turn:
                avg_rate_of_turn = 1 #degree Airbus H175

            if np.abs(self.current_ref_final_heading_deg - self.current_heading_deg) < 0.5:
                avg_rate_of_turn = 0.
            
            #* logic for heading update 
            if (np.sign(self.current_ref_final_heading_deg)==np.sign(self.current_heading_deg)==1):
                # and (ref_final_heading > current_heading_deg)) or ((np.sign(ref_final_heading)==np.sign(current_heading_deg)== -1) and (np.abs(ref_final_heading)<(np.abs(current_heading_deg)))):
                if self.current_ref_final_heading_deg > self.current_heading_deg:
                    self.current_heading_deg += avg_rate_of_turn #counter clockwise turn
                    self.current_heading_radians = np.deg2rad(self.current_heading_deg) 
                elif self.current_ref_final_heading_deg < self.current_heading_deg:
                    self.current_heading_deg -= avg_rate_of_turn #clockwise turn
                    self.current_heading_radians = np.deg2rad(self.current_heading_deg)
                else:
                    pass  
            
            elif np.sign(self.current_ref_final_heading_deg) == np.sign(self.current_heading_deg) == -1:
                if np.abs(self.current_ref_final_heading_deg) < np.abs(self.current_heading_deg):
                    self.current_heading_deg += avg_rate_of_turn #counter clockwise turn
                    self.current_heading_radians = np.deg2rad(self.current_heading_deg)
                elif np.abs(self.current_ref_final_heading_deg) > np.abs(self.current_heading_deg):
                    self.current_heading_deg -= avg_rate_of_turn #clockwise turn
                    self.current_heading_radians = np.deg2rad(self.current_heading_deg)
                else:
                    pass
                    
            elif np.sign(self.current_ref_final_heading_deg) == 1 and np.sign(self.current_heading_deg) == -1:
                self.current_heading_deg += avg_rate_of_turn #counter clockwise turn
                self.current_heading_radians = np.deg2rad(self.current_heading_deg)

            elif np.sign(self.current_ref_final_heading_deg) == -1 and np.sign(self.current_heading_deg) == 1:
                self.current_heading_deg -= avg_rate_of_turn #clockwise turn
                self.current_heading_radians = np.deg2rad(self.current_heading_deg)

            else:
                raise Exception('Error in heading correction')
        
        else:
            self.current_heading_deg += heading_correction_das_controller
            self.current_heading_radians = np.deg2rad(self.current_heading_deg)
                     

    def get_intruder_distance(self, other_uav):
        return self.current_position.distance(other_uav.current_position)
 

    def get_intruder_speed(self,other_uav):
        rel_heading = self.calculate_intruder_heading(other_uav)
        return self.current_speed - (np.cos(np.deg2rad(rel_heading)) * other_uav.current_speed)
    
    def get_intruder_heading(self,other_uav):
        return self.current_heading_deg - other_uav.current_heading_deg
    
    def get_other_uav_list(self, uav_list):
        other_uav_list = []
        for uav in uav_list:
            if uav.id != self.id:
                other_uav_list.append(uav)
        return other_uav_list

    def get_intruder_uav_list(self,uav_list, radius_str):
        '''
        Here the self.intruder_uav_list is created everytime as an empty list, 
        So everystep this attribute is an empty list and its populated with uavs that are within any(detection, nmac, collision) radius.
        There is no return from this method, the data is stored in the attribute and should be accessed immediately after calling this method.
        Any subsequent routines can call the attribute and use the attribute for data processing 
        '''
        if radius_str == 'detection':
            own_radius = other_radius = self.detection_radius
            
        elif radius_str == 'nmac':
            own_radius = other_radius = self.nmac_radius
            
        elif radius_str == 'collision':
            own_radius = other_radius = self.collision_radius
            
        else:
            raise RuntimeError('Unknown radius string passed.')

        self.intruder_uav_list = []
        other_uav_list = self.get_other_uav_list(uav_list)

        for other_uav in other_uav_list:
            if self.uav_polygon(own_radius).intersects(other_uav.uav_polygon(other_radius)):
                self.intruder_uav_list.append(other_uav)
        
        return self.intruder_uav_list



    #! 
    '''
    currently the get_state method only detects other uavs, 
    need to add logic for static_objects,
    might need to break down the logic inside get_state into two parts, 
    One for static object detection 
    One for dynamic object detection. 

    Then finally, the get state method should pull all the information and return a combined state information.


    Once get_state works for one intruder it has to work for n intruders.


    Looking ahead into the future static objects like buildings are clustered together, 
    there will be problem on how that observation will need to be dealt with. 
    '''


    def get_state_dynamic_obj(self, uav_list, radius_str = 'detection'):
        '''
        Get state of UAV based on radius string argument.
        This method will return UAVs that have been detected. 
        If no UAV is detected then we get a string back. - might need to change this 
        '''

        intruder_uav_list = self.get_intruder_uav_list(uav_list,radius_str)

        if len(intruder_uav_list) == 0:
            return None
        else:
            '''
            If there are more than one intruder, set first in the list as current intruder,
            and iterate through the list to find the intruder that is nearest. 
            State information, will be built using the nearest intruder. 
            '''
            current_intruder = intruder_uav_list[0] #set first uav in intruder_list as current intruder 
            #sort based on distance
            for ith_intruder in intruder_uav_list:
                if self.get_intruder_distance(ith_intruder) < self.get_intruder_distance(current_intruder):
                    current_intruder = ith_intruder #nearest intruder is current intruder
            
            intruder_state_info = {'own_id':self.id,
                                'own_pos': self.current_position,
                                'own_current_heading':self.current_heading_deg,
                                'intruder_id': current_intruder.id,
                                'intruder_pos': current_intruder.current_position,
                                'intruder_current_heading': current_intruder.current_heading_deg,
                                }
            
            return intruder_state_info
    
    def get_state_static_obj(self, building_gdf, radius_str = 'detection'):
        if radius_str == 'detection':
            own_radius = self.detection_radius
        elif radius_str == 'nmac':
            own_radius  = self.nmac_radius
        elif radius_str == 'collision':
            own_radius = self.collision_radius
        else:
            raise RuntimeError('Unknown radius string passed.')
        
        building_polygon_count = len(building_gdf)
        intersection_list = []
        
        for i in range(building_polygon_count):
            intersection_list.append(self.uav_polygon(own_radius).intersection(building_gdf.iloc[i]))
        
        intersection_with_building = any(intersection_list)
        own_state_info = self.current_heading_deg
        
        return intersection_with_building , own_state_info
                
        
    def get_state(self, uav_list, building_gdf, radius_str = 'detection'):
        static_state = self.get_state_static_obj(building_gdf, radius_str)
        dynamic_state = self.get_state_dynamic_obj(uav_list, radius_str) 


        return static_state, dynamic_state
        
         

    # the action argument should be a named_tuple acceleration and theta_dd
    # for simplicity only using acceleration now 
    def step(self,action):
        '''Updates the position of the UAV.'''
        if action is None:
            acceleration = None
        elif isinstance(action, tuple):
            acceleration = action[0]
            heading_correction = action[1]

        self._update_position(d_t=1, ) 
        self._update_speed(d_t=1, acceleration_from_controller=acceleration)
        self._update_ref_final_heading()
        self._update_theta_d(heading_correction)

        obs = self.current_position

        return obs
    



    #TODO - might need these later 
        
    def undef_state(self,uav_list ):
        # deviation = self.current_heading_deg - self.current_ref_final_heading_deg
        # speed = self.current_speed
        # num_intruder = self.calculate_intruder(uav_list)
        # intruder_distance = self.calculate_intruder_distance()
        # intruder_speed = self.calculate_intruder_speed()
        # intruder_heading = self.calculate_intruder_heading()

        # state = {'deviation':deviation,
        #               'speed':speed,
        #               'num_intruder':num_intruder,
        #               'intruder_distance':intruder_distance,
        #               'intruder_speed':intruder_speed,
        #               'intruder_heading':intruder_heading}
        # return state
        pass
    
    def get_global_state(self, uav_list):
        pass
    
        


    




        
    
    





