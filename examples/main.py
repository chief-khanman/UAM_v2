import os
import matplotlib.pyplot as plt
from simulator import Simulator
from simulator_basic import Simulator_basic
from matplotlib.animation import FuncAnimation
import geopandas as gpd
from utils import static_plot
from uam_sim.cas import CollisionController, ZeroController

 
#! - Complete the following checks 
#TODO - 1) make sure there is at least 2 vertiports at all time 
#TODO - 2) There is at least one uav - if 0 UAV, no need to run simulation(simulation should not run, but check to make sure)
#TODO - 3) make sure location name is valid
#TODO - 4) make sure vertiports are not on top of buildings and other structures 
#TODO - 5) run headless(without video) 
#TODO - 6) be able to dynamically add locations using strings  


if __name__ == '__main__':
    
    #controller = Zero_controller()
    controller = CollisionController()
    controller_predict = controller.get_action
    
    sim = Simulator_basic('Austin, Texas, USA', 20, 18,sleep_time=0, total_timestep=10000)
    
    #fig, ax initialization 
    plt.ion() 
    fig, ax = plt.subplots()

    #* remember this is a convinience function
    #sim.RUN_SIMULATOR(fig, ax, static_plot,sim, gpd,)
    
    for vertiports in sim.atc.vertiports_in_airspace:
        print('Vertiport id: ',vertiports.id)
        print('Vertiport uav list: ',vertiports.uav_list)
        

            



























    # *Plotting Logic
    # #TODO - Use FuncAnimation to animate the path of the UAV
    # #TODO - call a plotter function here that encapsulates this loop 





    # logfile = '</change /to /logfile /path>'
    # print('Simulation initialization -  ')
    # print('Current working dir', os.getcwd())
    # os.chdir(path='gym-examples/gym_examples/envs/assets')
    # print('Current working dir: ', os.getcwd())
    # print('state log files will be saved in ', logfile)