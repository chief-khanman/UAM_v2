import math
import numpy as np
import xarray as xr
from geopandas import GeoDataFrame as GDF
from shapely import Point
from uav_v2_template import UAV_v2_template




def get_map_bounds(location_gdf:GDF):
    '''For a given map GDF(GeoDataFrame) return the bounds
    min_x, min_y, max_x, max_y.
    
    Args:
        location GDF
    
    Returns:
        (float, float, float, float)'''

    # units returned from location_gdf are in meters
    
    # get bounds from airspace 
    min_x, min_y, max_x, max_y = location_gdf.total_bounds
    # floor min_x and min_y 
    min_x , min_y = int(math.floor(min_x)), int(math.floor(min_y))
    # ceil max_x and max_y
    max_x, max_y = int(math.ceil(max_x)), int(math.ceil(max_y))

    return min_x, min_y, max_x, max_y


def create_noise_matrix(min_x, min_y, max_x, max_y, cell_dim = 100):
    ''' Create noise matrix for each time step.
        The rows and columns of this matrix depends on
        map dimensions. Given map bounds return noise matrix.
        
        Args: 
             minx, miny, maxx, maxy
        
        Returns:
            ndarray
        '''
    
    
    rows, cols = int((max_y - min_y)/cell_dim), int((max_x - min_x)/cell_dim)
    noise_matrix = np.zeros((rows, cols))
    
    return noise_matrix


def uav_array_window(window_ln, center_idx):
    ''' Given window length, and window center index,
        return indices surrounding UAV. The window indices are 
        indices of the noise matrix, around UAV.  
        
        Args: 
            int, (int, int)
        
        Returns:
        A list of array indices to store the noise intensity.
            List[Tuple[int, int]]'''
    
    window = []
    row_range = column_range = window_ln//2
    for i in range(center_idx[0]-row_range, center_idx[0]+row_range+1):
        for j in range(center_idx[1]-column_range, center_idx[1]+column_range+1):
            window.append((i,j))
    
    return window

# position to array index
def pos2idx(position:Point, minx:int, miny:int):
    ''' Using position within map, 
        return corresponding array index'''
    # if a UAV's x position is 3.9999
    # and its    y position is 6.7777
    # using floor x is 3 and 
    #             y is 6
    # to get COLUMN index in array (x - minx) -> 3-3 = 0
    # to get ROW    index in array (y - miny) -> 6-6 = 0
    # so, UAV current_pos belongs to array_pos (row, col) -> (0,0)
    #  
    
    x, y  = int(math.floor(position.x)), int(math.floor(position.y))
    col = x - minx
    row = y - miny
    index = (row, col)
    
    return index


def idx2pos(array_loc, minx, miny, cell_dim=100):
    '''Provided an array index(location), 
    this method will return centroid positions
    of the corresponding array index(location)'''
    # add half of cell dimension to find centroid location

    row, col = array_loc[0], array_loc[1]

    x = minx + col + cell_dim//2
    y = miny + row + cell_dim//2
    position = Point((x,y))
    
    return position


# This method needs to be called inside another method,
# where there is a mapping between noise_array and obs_location
# because, we will only determine the sound 
# for a given window around a UAVs current location  
def get_noise_intensity(uav_current_pos, uav_current_speed, uav_rotor_speed, obs_location, altitude, noise_model=None):
    '''
    Given the (altitude) current position, speed and rotor speed, 
    for a given observer location return the noise intensity. 
    The arguments will be passed to a NN to provide the noise intensity
    at observer location.

    Args: 
        uav_current_pos: Point,
        uav_current_speed: float,
        uav_rotor_speed: float,
        obs_location: Point,
        altitude: float
    Returns:
        float  
    '''
    if noise_model == None:
        raise NotImplementedError('noise model not passed to kwarg')
    
    # noise_model has to be a pytorch model
    # add a check to ensure noise_model is a pytorch model

    noise_intensity = noise_model(uav_current_pos, uav_current_speed, uav_rotor_speed, altitude, obs_location)

    return noise_intensity
    

def calculate_noise_window(uav:UAV_v2_template, win_len, minx, miny, noise_matrix, noise_intensity = get_noise_intensity) -> None:
    '''Calculate and update the noise matrix'''
    # for a noise matrix and a defined noise window associated with a UAV
    # calculate noise level for the UAV's noise window
    #  
    uav_idx = pos2idx(uav.current_position, minx, miny)
    uav_noise_window = uav_array_window(win_len,uav_idx)
    
    for arr_idx in uav_noise_window:
        if arr_idx == uav_idx:
            # when array index matches UAV's array index, we are using UAVs current position to calculate the noise intensity 
            noise_lvl = noise_intensity(uav.current_position, uav.current_speed, uav.rotorspeed, uav.current_position)
            noise_matrix[arr_idx] += noise_lvl
        else:

            obs_pos = idx2pos(arr_idx, minx, miny)
            noise_lvl = noise_intensity(uav.current_position, uav.current_speed, uav.rotor_speed, obs_pos)
            noise_matrix[arr_idx] += noise_lvl
            
    pass


def set_noise_matrix_2_xarray(current_time_step, ):
    # add the noise matrix of current time step to x_array

    pass


# How to implement these methods in env:
# initialize noise matrix
# in step():
#   for uav in uav_list:
#       make noise_window for UAV
#       calculate noise intensity @ UAV's window
#       super_impose noise window over noise matrix
# save noise_matrix in xr_array
# 
# How to visualize: 
# use the xr_array 
# for every timestep:
#   collect noise_matrix from xr_array @ current_time  
#   plot noise_matrix USING ??? (there is a matplotlib method for plotting heat map, need to make sure its able to superimpose over existing static and dynamic assets)

if __name__ == '__main__':
    from airspace import Airspace
    airspace = Airspace(12, "Austin, Texas, USA")

    minx, miny, maxx, maxy = get_map_bounds(airspace.location_utm_gdf)
    print(minx, miny, maxx, maxy)

    noise_matrix = create_noise_matrix(minx, miny, maxx, maxy)
    print(f'Noise matrix: {noise_matrix}')
    print(f'Noise matrix shape: {noise_matrix.shape}')
    
    uav_pos2index = pos2idx(Point(602201.377178908, 3330295.7490083342),minx, miny)
    print(f'index of UAV position:{uav_pos2index}')

    some_window = uav_array_window(5,uav_pos2index)
    print(f'window at current uav position: {some_window}')
    

    
    
