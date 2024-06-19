#Welcome to the simulator, this takes all the classes from all the modules, and builds an instance of the simulator
from airspace import Airspace
from airtrafficcontroller import ATC
from uav import UAV
from uav_basic import UAV_Basic
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import Point
import time
from typing import List



class Simulator_basic:
 
    def __init__(self, location_name, num_vertiports, num_reg_uavs, sleep_time, total_timestep): 
        """
        Initializes a Simulator object.

        Args:
            location_name (str): The name of the location for the simulation.
            num_vertiports (int): The number of vertiports to create in the simulation.
            num_reg_uavs (int): The number of regular UAVs to create in the simulation.
        """       
        # sim airspace and ATC
        self.airspace = Airspace(location_name=location_name)
        self.atc = ATC(self.airspace, )
        # Initialize sim's vertiports and uavs using ATC 
        #*
        #! These two statements will most probably be in reset()
        #! Think how a seed value can be added to the system, so that the system is reproduceable - the vertiports and the uav assignment fixed to some seed value. 
        self.atc.create_n_random_vertiports(num_vertiports)
        self.atc.create_n_reg_uavs(num_reg_uavs,)
        
        # unpacking atc.vertiports in airspace
        vertiports_point_array = [vertiport.location for vertiport in self.atc.vertiports_in_airspace]
        # sim data
        self.sim_vertiports_point_array = vertiports_point_array
        self.uav_list:List[UAV_Basic] = self.atc.reg_uav_list
        #* 
        # sim sleep time
        self.sleep_time = sleep_time
        #sim run time
        self.total_timestep = total_timestep


    def render(self,fig, ax, static_plot, sim, gpd):
        plt.cla()
        static_plot(sim, ax, gpd)
        # UAV PLOT LOGIC
        for uav_obj in self.uav_list:
            uav_footprint_poly = uav_obj.uav_polygon_plot(uav_obj.uav_footprint)
            uav_footprint_poly.plot(ax=ax, color=uav_obj.uav_footprint_color, alpha=0.3)

            uav_nmac_poly = uav_obj.uav_polygon_plot(uav_obj.nmac_radius)
            uav_nmac_poly.plot(ax=ax, color=uav_obj.uav_nmac_radius_color, alpha=0.3)

            uav_detection_poly = uav_obj.uav_polygon_plot(uav_obj.detection_radius)
            uav_detection_poly.plot(ax=ax, color=uav_obj.uav_detection_radius_color,alpha=0.3)

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(self.sleep_time)

    
    def _get_obs(self,uav_obj:UAV_Basic):
        state_info = uav_obj.get_state(self.uav_list, self.airspace.location_utm_hospital_buffer)
        
        return state_info


    def get_uav(self, uav_id):
        for uav in self.uav_list:
            if uav.id == int(uav_id):
                return uav
            else:
                continue
        raise RuntimeError('UAV not it list')


    def set_uav_intruder_list(self):
        for uav in self.uav_list:
            uav.get_intruder_uav_list(self.uav_list)
    
    def set_building_gdf(self):
        for uav in self.uav_list:
            uav.get_airspace_building_list(self.airspace.location_utm_hospital_buffer)
    
    
    def sim_step(self, ):
        for uav in self.uav_list:
            self.atc.has_left_start_vertiport(uav)
            self.atc.has_reached_end_vertiport(uav)
            uav.step()
        
        

    def RUN_SIMULATOR(self, fig, ax, static_plot, sim, gpd,): 
        """
        Runs the simulator. 
        This method packs rendering, and stepping into one method. 
        Generally, for RL the loop is written explictly. This method was written for convinience. 

        Args:
            fig (matplotlib.figure.Figure): The figure object for plotting.
            ax (matplotlib.axes.Axes): The axes object for plotting.
            static_plot (function): A function that plots the static elements of the simulation.

        Returns:
            None
        """
        
        self.set_uav_intruder_list()
        self.set_building_gdf()
        
        for _ in range(self.total_timestep):
            self.render(fig, ax, static_plot, sim, gpd)
            self.sim_step() #! how would step behave to action_list 
        
        print('Simulation complete.')





    # #TODO - this is necessary 
    # def reset(self,):
    #     pass 



    # #TODO - this is necessary 
    # def close(self,):
    #     pass

