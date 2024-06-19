# Collision Avoidance System (CAS) module

from geopandas import GeoSeries
import numpy as np 
#from uav import UAV


'''
All controller's method -> get_action(state) : should accept state
                                      state  : should include all sort of information 
                                      now 
                                      1) state information can be empty                                                   if state is empty -> control_acceleration = 0, control_heading = 0
        this tells me get state 
        should have a new state observation
        for static objects as well    2) state information can have static object(buildings)                if state only has static object -> control_acceleration = 0, control_heading = static_object_heading_control
                                      3) state information can have dynamic object(other uavs)
                                      4) state information can a both static and dynamic 


                                    

'''
class CollisionController:
    '''
    Controllers are defined for individualy UAVs, 
    A controller will be used for a certain type of UAV, 
    The controller will accept the observation of one UAV
    and based on that specific UAVs observation, it will assign action to that specific UAV.
    '''

    def __init__(self) -> None:
        self.name = 'Baseline Collision Controller'

    def get_quadrant(self, theta):
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
        
    '''
    state -> static state                dynamic state                          get_action
            False                           None                                    
    '''
    #                    state -> ((bool,                float                  ), dict            | None)
    #                               static_obj_detected, uav.current_heading_deg , dynamic_obj_info| None
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
                heading_correction = 25
            elif -180<=current_heading or current_heading<= 0: 
                heading_correction = -25
            else:
                raise RuntimeError(f'DAS module - state[0][0] is True and state[1] is None, current heading {state[0][1]}')

        elif state[0][0] is True and isinstance(state[1], dict):
            acceleration = 0
            heading_correction = 0
        
        else:
            raise RuntimeError('DAS module: static and dynamic states do not match the conditionals')

        return acceleration, heading_correction
    


class ZeroController:
    '''
    Controllers are defined for individualy UAVs, 
    A controller will be used for a certain type of UAV, 
    The controller will accept the observation of one UAV
    and based on that specific UAVs observation, it will assign action to that specific UAV.
    '''
    def __init__(self) -> None:
        self.name = 'zero controller'

    def get_action(self, state):
        return 0,0 
    



#! add two new controllers - both should perform static object detection and avoidance, one should have dynamic obj detection and avoidance as well 