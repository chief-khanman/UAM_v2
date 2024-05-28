# Detection and avoidance system DAS 

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
class Collision_controller:
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
        

    def get_action(self, state):
        if state == None:
            return 0,0
        else:
            own_pos = state['own_pos']
            int_pos = state['intruder_pos']
            own_heading = state['own_current_heading']
            int_heading = state['intruder_current_heading']

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

            return acceleration, heading_correction
    


class Zero_controller:
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