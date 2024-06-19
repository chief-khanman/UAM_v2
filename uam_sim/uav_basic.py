#Deterministic uavs
from random import sample as random_sample
import numpy as np
from shapely.geometry import Point
from geopandas import GeoSeries
from vertiport import Vertiport
from uam_sim.cas import CollisionController

class UAV_Basic:
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
        
        
        #UAV rendering-representation properties 
        self.uav_footprint_color = 'blue' # this color represents the UAV object 
        self.uav_nmac_radius_color = 'orange'
        self.uav_detection_radius_color = 'green'
        self.uav_collision_controller = None
        
        #UAV builtin properties
        self.uav_footprint = 17 #H175 nose to tail length of 17m,
        self.nmac_radius = 150 #NMAC radius
        self.detection_radius = 550
        
        #UAV technical properties
        self.id = id(self)
        self.heading_deg = np.random.randint(-178,178) + np.random.rand() # random heading between -180 and 180
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

    def acceleration_controller(self,):

        if self.current_speed == 0:
            acc = self.max_acceleration
        elif self.current_position.distance(self.end_point) <= 500:
            acc = - ((self.max_speed)**2 / (2*500))
        elif self.current_speed <= self.max_speed:
            acc = self.max_acceleration
        else:
            acc = 0
    
        return acc 

    
    def _update_speed(self, d_t, acceleration_from_controller):
        '''
        Arg: acceleration_from_controller is an input from controller/das_system
        '''
        base_acc = self.acceleration_controller()
        '''
        if acceleration from controller, meaning controller is sending some acceleration value,
        then acceleration from controller will over-ride acceleration from speed controller
        
        '''
        if acceleration_from_controller != 0:
            final_acc = acceleration_from_controller
        else:
            final_acc = base_acc
        
        self.current_speed = self.current_speed + (final_acc * d_t)


    def _update_position(self,d_t:float,):
        '''Internal method. Updates current_position of the UAV after d_t seconds.
           This uses a first order Euler's method to update the position.
           '''
        update_x = self.current_position.x + self.current_speed * np.cos(self.current_heading_radians) * d_t 
        update_y = self.current_position.y + self.current_speed * np.sin(self.current_heading_radians) * d_t 
        self.current_position = Point(update_x,update_y)
    

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
            print('Heading before correction: ', self.current_heading_deg)
            self.current_heading_deg += heading_correction_das_controller
            print('Heading after correction: ', self.current_heading_deg)
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

    def get_intruder_uav_list(self,uav_list, radius_str = 'detection'):
        #! This method is called by simulator
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
        
    def get_airspace_building_list(self, building_gdf):
        self.building_gdf = building_gdf


    def get_state_dynamic_obj(self,):
        '''
        Get state of UAV based on radius string argument.
        This method will return UAVs that have been detected. 
        If no UAV is detected then we get a string back. - might need to change this 
        '''

        intruder_uav_list = self.intruder_uav_list

        if len(intruder_uav_list) == 0:
            return None
        else:
            '''
            If there are more than one intruder, set first in the list as current intruder,
            and iterate through the list to find the intruder that is nearest. 
            State information, will be built using the nearest intruder. 
            '''
            current_intruder = intruder_uav_list[0] #set first uav in intruder_list as current intruder 
            #!sort based on distance
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
    
    def get_state_static_obj(self, radius_str = 'detection'):
        '''
        Currently, this method only detects a building and returns True/False.
        I have an algorithm, in Zoom whiteboard, 
        use that for better collision avoidance performance.
        '''


        if radius_str == 'detection':
            own_radius = self.detection_radius
        elif radius_str == 'nmac':
            own_radius  = self.nmac_radius
        elif radius_str == 'collision':
            own_radius = self.collision_radius
        else:
            raise RuntimeError('Unknown radius string passed.')
        
        building_polygon_count = len(self.building_gdf)
        intersection_list = []
        
        for i in range(building_polygon_count):
            intersection_list.append(self.uav_polygon(own_radius).intersection(self.building_gdf.iloc[i]))
        
        intersection_with_building = any(intersection_list)
        
        return intersection_with_building , self.current_heading_deg
                
        
    def get_state(self, ):
        static_state = self.get_state_static_obj()
        dynamic_state = self.get_state_dynamic_obj() 


        return static_state, dynamic_state
        
    @staticmethod
    def get_quadrant(theta):
        if (theta >= 0) and (theta < 90):
            return 1
        elif (theta >= 90) and (theta <= 180):
            return 2
        elif (theta < 0) and (theta >= -90):
            return 3
        elif (theta >= -180) and (theta < -90):
            return 4
        else:
            raise RuntimeError('DAS Error: Invalid heading')
    
    def get_action(self, state): 
        if state[0][0] is False and state[1] is None:
            acceleration = 0
            heading_correction = 0
        
        elif state[0][0] is False and isinstance(state[1], dict):
            own_pos = state[1]['own_pos']
            int_pos = state[1]['intruder_pos']
            own_heading = state[1]['own_current_heading']
            int_heading = state[1]['intruder_current_heading']

            del_x = int_pos.x - own_pos.x
            del_y = int_pos.y - own_pos.y
            own_quadrant = self.get_quadrant(own_heading)
            intruder_quadrant = self.get_quadrant(int_heading)
            if (del_x > 0) and (del_y > 0):
                if (own_quadrant == 1) and (intruder_quadrant == 4) :
                    heading_correction = 25
                    acceleration = -1
                else:
                    heading_correction = 0
                    acceleration = 0
            elif (del_x < 0) and (del_y > 0):
                if (own_quadrant == 2) and (intruder_quadrant == 3) :
                    heading_correction = -25
                    acceleration = -1
                else:
                    heading_correction = 0
                    acceleration = 0
            elif (del_x < 0) and (del_y < 0):
                if (own_quadrant == 4) and (intruder_quadrant == 1):
                    heading_correction = 25
                    acceleration = -1
                else:
                    heading_correction = 0
                    acceleration = 0
            elif (del_x > 0) and (del_y < 0):
                if (own_quadrant == 3) and (intruder_quadrant == 2):
                    heading_correction = -25
                    acceleration = -1
                else:
                    heading_correction = 0
                    acceleration = 0
            else:
                raise RuntimeError('Action not from scenario')
        
        elif state[0][0] is True and state[1] is None:
            acceleration = 0 
            current_heading = state[0][1]
            
            if (current_heading < 0):
                current_heading += 360
            elif (current_heading > 180):
                current_heading -= 360

            if 0 <= current_heading or current_heading <= 180:
                heading_correction = 5
            elif -180<=current_heading or current_heading<= 0: 
                heading_correction = -5
            else:
                raise RuntimeError(f'DAS module - state[0][0] is True and state[1] is None, current heading {state[0][1]}')

        elif state[0][0] is True and isinstance(state[1], dict):
            acceleration = 0
            heading_correction = 0
        
        else:
            raise RuntimeError('DAS module: static and dynamic states do not match the conditionals')

        return acceleration, heading_correction
    
    

    #TODO -  the action argument should be a named_tuple acceleration and theta_dd
    
    def step(self,):
        '''Updates the position of the UAV.'''
        
        state = self.get_state()
        action = self.get_action(state)

        if action is None:
            acceleration = None
        elif isinstance(action, tuple):
            acceleration = action[0]
            heading_correction = action[1]

        self._update_position(d_t=1, ) 
        self._update_speed(d_t=1, acceleration_from_controller=acceleration)
        self._update_theta_d(heading_correction)
        self._update_ref_final_heading()

        obs = self.current_position

        return obs
    





    




        
    
    





