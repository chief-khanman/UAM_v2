import numpy as np 
import matplotlib.pyplot as plt
from typing import List
import time
from geopandas import GeoSeries

import gymnasium as gym
from gymnasium import spaces

from assets import airspace, airtrafficcontroller 
from assets.uav_basic import UAV_Basic



#! Warning
'''
Remember this environment will only be used for training,
Therefore there will only be one auto_uav which is built into the uam_env 
'''

'''
    When we create the UAM env(subclass of gymEnv) it will build an instance that is similar to simulator.
    The initializer arguments of UAM_Env will be passed to the simulator, that is location_name, reg_uav_no, vertiport_no, and Auto_uav(only one for now)
    [** emphasizing, the above arguments are arguments of UAV_Env passed to simulator_env**]
    
    Inside the simulator there will be one instance of Auto_UAV, this Auto_UAV's argument is a tuple of actions defined in UAV_Env.
    The Auto_UAV navigates the airspace using these actions. 

    *** The "step" method of UAV_Env, is used to step every uav(meaning reg_uav and Auto_uav)

    ***Refer to uam_single_agent_env's TRAINING section for questions that need to be answered, for further documentation and clarification

     
'''
#! End 


class Uam_Uav_Env(gym.Env):
    metadata = {"render_mode":["human", "rgb_array"], "render_fps":4}

    def __init__(self, location_name, num_vertiport, num_reg_uav,sleep_time = 0.005, render_mode=None):
        
        uam_airspace = airspace.Airspace(location_name)
        uam_atc = airtrafficcontroller.ATC(airspace=uam_airspace)
        
        self.airspace = uam_airspace
        self.atc = uam_atc
        
        #! this might belong to reset() 
        #*
        self.atc.create_n_random_vertiports(num_vertiport)
        self.atc.create_n_reg_uavs(num_reg_uav)
        vertiports_point_array = [vertiport.location for vertiport in self.atc.vertiports_in_airspace]
        # sim data
        self.sim_vertiports_point_array = vertiports_point_array
        self.uav_basic_list:List[UAV_Basic] = self.atc.reg_uav_list
        #*

        #! find a vertiport from self.atc.vertiports 
        #   loop through 
        # i need empty veriport because i need to assign the auto uav to that vertiport  
        self.observation_space = {'agent_id': int,                              #!check if an int can be an observation space type
                                  'agent_pos': spaces.Box(),                    #!find what should be the low and high of these from airspace - there is a max and min distance for the airspace
                                  'agent_current_heading':spaces.Box(),         #!this will be between -180 and 180 
                                  'intruder_id':int,
                                  'intruder_pos':spaces.Box(),
                                  'intruder_current_heading': spaces.Box(),
                                  'intersection_with_building':bool}
        self.action_space = spaces.Box()                                        #!action space will have both acceleration and heading change so find what is necessary for this

        self.sleep_time = sleep_time




        #! create_auto_uav()

    def reset(self,):
        '''
        This method will reset the environment. NOT sure how this will be done, but it will be done.
        '''
        # this will first remove the existing vertiports 
        #                 remove the existing reg uavs
        #                 remove the auto_uav
        # then it will create new vertiport 
        #              create new reg_uavs
        # then this method will use one of the empty vertiports and assign it to the auto_uav
        
        '''
        Since reset is called after creating the env, in reset we will perform the following 
        
        1) create the auto_uav with start_vertiport -> empty_vertiport, end_vertiport -> choose a random vertiport from vertiport list this function will be performed by the ATC
        '''
        #! check the following things - 1) how and where is a uav_basic created, 
        #! how are its attributes assigned, 
        #! once this is clear -> create a script for auto_uav, and use that to create the auto_uav above in reset and assign all the necessary attributes to it

        self.auto_uav = self.atc.create_auto_uav() 

        pass 
    
    
    
    def uav_polygon(self, dimension):
        return GeoSeries(self.current_position).buffer(dimension).iloc[0]
    

    def get_intruder_uav_list(self, radius_str = 'detection'):
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

        for uav in self.uav_basic_list:
            if self.uav_polygon(own_radius).intersects(uav.uav_polygon(other_radius)):
                self.intruder_uav_list.append(uav)

    
    
    def get_observation_dynamic_obj(self,):
        intruder_uav_list = self.intruder_uav_list
        
        #! DYNAMIC OBSERVATION NEEDS TO BE HANDLED FOR NONE CASE
        if len(intruder_uav_list) == 0:
            return None
        
        else:
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
            
            return intruder_state_info #! THERE ARE TWO DIFFERENT RETURNS - 1)NONE 2) DICT - NEED TO FIND A WAY TO HANDLE NONE 
        
    def get_observation_static_obj(self, radius_str = 'detection'):
        if radius_str == 'detection':
            own_radius = self.detection_radius
        elif radius_str == 'nmac':
            own_radius  = self.nmac_radius
        elif radius_str == 'collision':
            own_radius = self.collision_radius
        else:
            raise RuntimeError('Unknown radius string passed.')
        
        building_polygon_count = len(self.airspace.location_utm_hospital_buffer)
        intersection_list = []
        
        for i in range(building_polygon_count):
            intersection_list.append(self.uav_polygon(own_radius).intersection(self.building_gdf.iloc[i]))
        
        intersection_with_building = any(intersection_list)
        
        return intersection_with_building , self.current_heading_deg #! RETURN - TUPLE[BOOL, FLOAT] 


    def _get_obs(self,):
        '''
        This method will collect observations for auto_uav only. 
        '''

        observation_static = self.get_observation_static_obj()
        intersection_with_building = observation_static[0] #! -> bool, True or False

        observation_dynamic = self.get_observation_dynamic_obj()

        if observation_dynamic is None:
            current_intruder_id = None
            current_intruder_current_position = 1000
            current_intruder_current_heading_deg = None
        else:
            current_intruder_id = observation_dynamic['intruder_id']
            current_intruder_current_position = observation_dynamic['intruder_position']
            current_intruder_current_heading_deg = observation_dynamic['intruder_current_heading']


        return {   'agent_id': self.id,                              #!check if an int can be an observation space type
                    'agent_pos': self.current_position,                    #!find what should be the low and high of these from airspace - there is a max and min distance for the airspace
                    'agent_current_heading':self.current_heading,         #!this will be between -180 and 180 
                    'intruder_id':current_intruder_id,
                    'intruder_pos':current_intruder_current_position,
                    'intruder_current_heading': current_intruder_current_heading_deg,
                    'intersection_with_building':intersection_with_building}
        
         

    def _get_info(self,):
        pass 



    def step(self,action):
        '''
        This method is used to step the environment, it will step the environment by one timestep.
        
        The action argument - will be passed to auto_uav's step method
        
        Regular UAVs will step without action. so I will need to modify regular uav in such a way that they will step without action. 
        This tells me that regular uav will need to have collision avoidance built into the UAV module, such that they can step without action. 
         
        '''
        #decomposing action tuple 
        #! these two need to be passed to auto_uav
        acceleration = action[0]
        heading_correction = action[1]
        
        #for uav in uav_basic_list step all uav_basic
        #! if i have auto_uav, then i can use the same structure from ATC and perform the following funtions without any problem
        for uav in self.uav_basic_list:
            self.atc.has_left_start_vertiport(uav)
            self.atc.has_reached_end_vertiport(uav)
            uav.step()
        
        self.auto_uav.step(acceleration, heading_correction) #! this will be created inside the reset method
        
        
        #! then lets call get_obs on auto_uav and connect that to env._get_obs()
        obs = self._get_obs()
        reward = self.get_reward(obs)
        done = None #! HOW should we check for truncation, and termination
        info = None #! DO we need info 

        return obs, reward, done, info

    def render(self,fig, ax, static_plot, sim, gpd):
        plt.cla()
        static_plot(sim, ax, gpd)
        # UAV PLOT LOGIC
        for uav_obj in self.uav_basic_list:
            uav_footprint_poly = uav_obj.uav_polygon_plot(uav_obj.uav_footprint)
            uav_footprint_poly.plot(ax=ax, color=uav_obj.uav_footprint_color, alpha=0.3)

            uav_nmac_poly = uav_obj.uav_polygon_plot(uav_obj.nmac_radius)
            uav_nmac_poly.plot(ax=ax, color=uav_obj.uav_nmac_radius_color, alpha=0.3)

            uav_detection_poly = uav_obj.uav_polygon_plot(uav_obj.detection_radius)
            uav_detection_poly.plot(ax=ax, color=uav_obj.uav_detection_radius_color,alpha=0.3)

        #TODO - render auto_uav here

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(self.sleep_time)

    def _render_frame(self,):
        pass

    def close(self,):
        pass
    