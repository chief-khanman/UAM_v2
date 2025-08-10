from sensor_template import SensorTemplate
from typing import List, Dict, Tuple, Any
import numpy as np
import math
import shapely
from uav_v2_template import UAV_v2_template
from auto_uav_v2 import Auto_UAV_v2
from airspace import Airspace
from atc import ATC
import geopandas as gpd

#FIX: rename to sensor_universal.py and SensorUniversal
class MapSensor(SensorTemplate):
    """
    Sensor for UAVs in a mapped environment with restricted airspace.
    Handles both dynamic (other UAVs) and static (restricted airspace) collision detection.
    """
    
    def __init__(self, airspace:Airspace, atc:ATC):
        """
        Initialize the MapSensor.
        
        Args:
            space: The space containing UAVs, vertiports, and airspace information
        """
        
        self.airspace = airspace
        self.atc = atc
        return None



    def get_data(self, self_uav) -> Tuple[List, List]:
        """
        Collect data of other UAVs in space within detection radius.
        
        Args:
            self_uav: The UAV using this sensor
        """

        other_uav_data = self.get_uav_detection(self_uav)
        ra_data = self.get_ra_detection(self_uav)
        #       List|List[Dict]    List|List[Dict]
        return other_uav_data, ra_data
    

    #### UAV ####
    def get_uav_detection(self, self_uav: UAV_v2_template) -> List[Dict]:
        """
        Return data of other UAVs in space within detection radius.
        
        Args:
            self_uav: The UAV using this sensor
        """
        # Clear previous data
        other_uav_data_list = []
        
        # Get self_uav detection radius
        self.detection_radius = self_uav.detection_radius
        
        # Get UAV list from space
        uav_list: List[UAV_v2_template] = self.atc.get_uav_list()
        
        # Collect data for each UAV within detection radius
        for uav in uav_list:
            if uav.id != self_uav.id:
                if self_uav.current_position.distance(uav.current_position) <= self.detection_radius:
                    other_uav_data = {
                        'other_uav_id': uav.id,
                        'other_uav_current_position': uav.current_position,
                        'other_uav_current_speed': uav.current_speed,
                        'other_uav_current_heading': uav.current_heading,
                        'other_uav_radius': uav.radius
                    }
                    other_uav_data_list.append(other_uav_data)
        
        return other_uav_data_list
    
    def get_nmac(self, self_uav: UAV_v2_template) -> Tuple[bool, List]:
        """
        Check for Near Mid-Air Collision (NMAC) with other UAVs.
        
        Args:
            self_uav: The UAV checking for NMAC
            
        Returns:
            Tuple of (nmac_detected, nmac_uav_list)
        """
        nmac_list = []
        uav_list = self.atc.get_uav_list()
        
        deactivation_flag = self.deactivate_nmac(self_uav)
        if deactivation_flag == True:
            return False, []
        
        for uav in uav_list:
            if uav.id == self_uav.id:
                continue
            
            # Check if NMAC radii intersect
            if self_uav.current_position.buffer(self_uav.nmac_radius).intersects(
                uav.current_position.buffer(uav.nmac_radius)
            ):
                nmac_list.append(uav)
                self_uav.nmac_count += 1
                
        if len(nmac_list) > 0:
            return True, nmac_list
        return False, nmac_list
    
    def get_uav_collision(self, self_uav: UAV_v2_template) -> Tuple[bool, Any]:
        """
        Check if there is a collision with another UAV.
        
        Args:
            self_uav: The UAV checking for collisions
            
        Returns:
            Tuple of (collision_detected, collision_info)
        """

        deactivate_collision_flag = self.deactivate_collision(self_uav)
        if deactivate_collision_flag == True:
            return False, [] 
        
        uav_list = self.atc.get_uav_list()
        
        for uav in uav_list:
            if uav.id == self_uav.id:
                continue
            
            # Check if UAV bodies intersect
            if self_uav.current_position.buffer(self_uav.radius).intersects(
                uav.current_position.buffer(uav.radius)
            ):
                return True, (self_uav.id, uav.id)
                
        return False, None
    
    
    
    
    #### Restricted Area ####
    # TODO: this method needs to be tested,
    # TODO: to see what is the output when there is detection of Restricted Airspace
    def get_ra_detection(self, self_uav: UAV_v2_template) -> List[Dict]:
        """
        Check if the UAV's detection radius intersects with restricted airspace.
        
        Args:
            self_uav: The UAV checking for static object detection
            
        Returns:
            Tuple of (detection, detection_info)
        """
        
        # ra_data
        ra_data = []
        # Get the buffer around the UAV's detection radius
        uav_detection_area = self_uav.current_position.buffer(self_uav.detection_radius)
        
        # Check intersection with each restricted area
        if hasattr(self.airspace, 'location_tags'):
            for tag_value in self.airspace.location_tags.keys():
                # Get restricted area buffers (hospitals, airports, etc)
                restricted_areas_buffer = self.airspace.location_utm_buffer[tag_value]
                
                # each restricted_areas_buffer is a polygon
                # Check for intersection with any restricted area buffer
                # restricted_areas_buffer is a list of polygon
                for i in range(len(restricted_areas_buffer)):
                    # restricted_area is a polygon
                    restricted_area = restricted_areas_buffer.iloc[i]
                    # Extract geometry object from GeoSeries/GeoDataFrame
                    if hasattr(restricted_area, 'geometry'):
                        restricted_geometry = restricted_area.geometry
                    else:
                        restricted_geometry = restricted_area

                    if uav_detection_area.intersects(restricted_geometry):
                        # Calculate distance to the restricted area
                        distance = self_uav.current_position.distance(restricted_geometry.boundary)
                        # radial angle of vector pointing from centroid of ra to UAVs current_position
                        ra_heading = math.atan2((self_uav.current_position.y - restricted_geometry.centroid.y), (self_uav.current_position.x - restricted_geometry.centroid.x))
                        # Return detection info
                        ra_data.append({
                            'type': tag_value,
                            'distance': distance,
                            'ra_heading':ra_heading,
                            'area': restricted_geometry
                        })
        
        return ra_data
    
    def get_ra_collision(self, self_uav: UAV_v2_template) -> Tuple[bool, Dict]:
        """
        Check if the UAV's body intersects with restricted airspace.
        
        Args:
            self_uav: The UAV checking for static object collision
            
        Returns:
            Tuple of (collision, collision_info)
        """
        # Get the buffer around the UAV's physical radius
        uav_body = self_uav.current_position.buffer(self_uav.radius)
        
        # Check intersection with each restricted area
        if hasattr(self.airspace, 'location_tags'):
            for tag_value in self.airspace.location_tags.keys():
                # Get actual restricted areas (not buffers)
                restricted_areas = self.airspace.location_utm[tag_value]
                
                # Check for intersection with any restricted area
                for i in range(len(restricted_areas)):
                    # Get the actual geometry from the GeoDataFrame/GeoSeries
                    restricted_area = restricted_areas.iloc[i]
                    if hasattr(restricted_area, 'geometry'):
                        restricted_geometry = restricted_area.geometry
                    else:
                        restricted_geometry = restricted_area
                    
                    try:
                        if uav_body.intersects(restricted_geometry):
                            return True, {
                                'type': tag_value,
                                'area': restricted_geometry
                            }
                    except TypeError:
                        # If an error occurs, log detailed info for debugging
                        print(f"Type error with: {type(restricted_area)}, {type(restricted_geometry)}")
                        print(f"UAV body type: {type(uav_body)}")
                        continue
        
        return False, {}
    
    def get_closest_restricted_area(self, self_uav: UAV_v2_template) -> Tuple[bool, Dict]:
        """
        Find the closest restricted area to the UAV within detection radius.
        
        Args:
            self_uav: The UAV checking for nearby restricted areas
            
        Returns:
            Tuple of (area_detected, area_info)
        """
        min_distance = float('inf')
        closest_area = None
        closest_type = None
        
        # Get the buffer around the UAV's detection radius
        uav_detection_area = self_uav.current_position.buffer(self_uav.detection_radius)
        
        # Check distance to each restricted area
        for tag_value in self.airspace.location_tags.keys():
            # Get restricted areas
            restricted_areas = self.airspace.location_utm[tag_value]
            
            # Check distance to each area
            for i in range(len(restricted_areas)):
                # Get the actual geometry
                restricted_area = restricted_areas.iloc[i]
                if hasattr(restricted_area, 'geometry'):
                    restricted_geometry = restricted_area.geometry
                else:
                    restricted_geometry = restricted_area
                
                # If within detection range
                try:
                    if uav_detection_area.intersects(restricted_geometry):
                        # Calculate distance
                        distance = self_uav.current_position.distance(restricted_geometry.boundary)
                        
                        # Update if closer than current closest
                        if distance < min_distance:
                            min_distance = distance
                            closest_area = restricted_geometry
                            closest_type = tag_value
                except TypeError:
                    continue
        
        if closest_area is not None:
            return True, {
                'type': closest_type,
                'distance': min_distance,
                'area': closest_area
            }
        
        return False, {}
    

    def deactivate_nmac(self, uav)->bool:
        if uav.current_position.distance(uav.start) <= 100 or uav.current_position.distance(uav.end)<=100:
            return True
        return False

    def deactivate_detection()->None:
        pass

    def deactivate_collision(self, uav)->bool:
        if uav.current_position.distance(uav.start) <= 100 or uav.current_position.distance(uav.end)<=100:
            return True  
        return False