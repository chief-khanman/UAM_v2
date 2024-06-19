'''
    FOR Training the RL algorithm - the simulation should run until the Auto_UAV reaches its endpoint
                                  - so simulation ends when Auto_UAV reaches its endpoint 
                                  - We run N number of simulations 
    
    '''


'''every class is one object - every function does only one thing for that object '''

from shapely import Point
from typing import List


class Vertiport:
    def __init__(self,location, uav_list=[]): #sim:simulator.Simulator,
        self.id = id(self)
        self.location = location
        self.uav_list:List = uav_list 

    # def __repr__(self,):
    #     return 'Vertiport({location}, {uav_list})'.format(location=self.location, uav_list=self.uav_list)