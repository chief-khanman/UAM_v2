import numpy as np
import matplotlib.pyplot as plt
import shapely
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from osmnx import features as ox_features
from osmnx import geocode_to_gdf as geocode_to_gdf
from osmnx import projection as ox_projection
from typing import List, Tuple, Dict
import math
from shapely import Point
import random
from sklearn.cluster import KMeans as KM 

from vertiport import Vertiport
#FIX:
# this module will now handle creating objects in/on airspace
# vertiport creation
# restricted airspace creation
#  
class Airspace: #                                                                            airspace_tag_list: List[Tuple('building:commercial', ...)]
                #                                                                                                           
    def __init__(self, number_of_vertiports, location_name: str, buffer_radius: float = 500, airspace_tag_list=[], vertiport_tag_list=[], max_vertiports=25) -> None:   #! airspace feature has to be list of strings
        """Airspace - Defines the location of the map. Imports key information on hopitals(no fly zone)

        Args:
            location_name (string): Location of the Airspace ie. "Austin, Texas, USA"
            buffer_radius (int): distance around hospitals

        Attributes:
            location_name(string): Location of the Airspace ie. "Austin, Texas, USA"
            buffer_radius (int): distance around hospitals
            location_utm_gdf (gdp.GeoDataFrame): Location in UTM(Universal Transverse Mercator)
            location_utm_hospital ( ox_projection): location of hospital converted to UTM
            location_utm_hospital_buffer (UTM): buffer around the hospital
        """
        self.vertiport_tags = {}
        self.vertiport_feature_gdf = {}
        self.vertiport_utm = {}

        self.location_name = location_name  #'Austin, Texas, USA'
        self.buffer_radius = buffer_radius
        self.airspace_tag_list = airspace_tag_list
        self.vertiport_tag_list = vertiport_tag_list

        # location - this is the airspace where we are working 
        location_gdf = geocode_to_gdf(self.location_name)  # converts named geocode - 'Austin,Texas' location to gdf
        self.location_utm_gdf: gpd.GeoDataFrame = ox_projection.project_gdf(location_gdf)  # default projection - UTM projection #! GeoDataFrame has deprication warning - need quick fix
        self.location_utm_gdf["boundary"] = (self.location_utm_gdf.boundary)  # adding column 'boundary'
        
        if self.vertiport_tag_list:
            # Populate the dictionaries if tags were provided
            #                                                           tag     ,  tag_value
            #                          vertiport_tag_list: List[Tuple('building', 'commercial'), ... , ... , ... ]
            for tag, tag_value in self.vertiport_tag_list:
                self.vertiport_tags[tag_value] = tag
                self.vertiport_feature_gdf[tag_value] = ox_features.features_from_polygon(location_gdf["geometry"][0], tags={tag:tag_value})
                self.vertiport_utm[tag_value] = ox_projection.project_gdf(self.vertiport_feature_gdf[tag_value])




        #Airspace RA tag list
        if self.airspace_tag_list:
            # airspace features and restricted airspace 
            self.location_tags = {}
            self.location_feature_gdf = {}
            self.location_utm = {}
            self.location_utm_buffer = {}
            
            self.airspace_restricted_area_buffer_array = []
            self.airspace_restricted_area_array = []
            
            for tag, tag_value in self.airspace_tag_list:
                self.location_tags[tag_value] = tag
                self.location_feature_gdf[tag_value] = ox_features.features_from_polygon(location_gdf["geometry"][0], tags={tag:tag_value})
                self.location_utm[tag_value] = ox_projection.project_gdf(self.location_feature_gdf[tag_value])
                self.location_utm_buffer[tag_value] = self.location_utm[tag_value].buffer(self.buffer_radius)
                self.airspace_restricted_area_array.append(self.location_utm[tag_value])
                self.airspace_restricted_area_buffer_array.append(self.location_utm_buffer[tag_value])
            
            self.restricted_airspace_buffer_geo_series = pd.concat(self.airspace_restricted_area_buffer_array)
            self.restricted_airspace_geo_series = pd.concat(self.airspace_restricted_area_array)


        # Vertiport
        self.number_of_vertiports = number_of_vertiports
        self.vertiport_list:List = []
        self.max_vertiports = max_vertiports 
        self.vertiport_site_polygon_dict = {}

    def __repr__(self) -> str:
        return "Airspace({location_name})".format(location_name=self.location_name)

    def set_vertiport(self,vertiport):
        """
        Adds a vertiport to the vertiport list.

        Args:
            vertiport: The vertiport to add.
        
        Returns:
            None
        """
        if len(self.vertiport_list) < self.max_vertiports:
            self.vertiport_list.append(vertiport)
        else:
            print('Max number of vertiports reached, additonal vertiports will not be added')
        return None 
    
    def set_random_sample_vertiports(self, vertiports:List[Vertiport], sample_number=None):
        '''Given a list of vertiports, 
            add randomly sampled 'sample_number' of vertiports 
            to airspace's vertiport_list'''
        
        if sample_number:
            sampled_vertiports = random.sample(vertiports, sample_number)
        
        for vertiport in sampled_vertiports:
            self.set_vertiport(vertiport)
        
        return None
    
    


    def get_vertiport_list(self):
        """
        Returns the list of vertiports.

        Returns:
            List: The list of vertiports.
        """
        return self.vertiport_list



    def create_vertiport_at_location(self, position:Tuple)-> None:
        """Create a vertiport at position(x,y)."""
        position = Point(position[0], position[1])
        
        if self.airspace_tag_list:
            for tag_value in self.location_tags.keys():
                sample_space = self.location_utm_gdf.iloc[0,0].difference(
                    self.location_utm_buffer[tag_value].unary_union
                )
            sample_space_gdf = GeoSeries(sample_space)
        else: 
            sample_space = self.location_utm_gdf
            sample_space_gdf = sample_space.geometry


        sample_space_array: np.ndarray = shapely.get_parts(sample_space_gdf)

        for sample in sample_space_array:
            if sample.contains(position):
                print('Valid location for vertiport')
                _vertiport = Vertiport(position)
                return _vertiport
        
        print('Not a valid position for vertiport')

        return None
    


    def create_vertiport_from_site_polygon(self,polygon:shapely.Polygon) -> Vertiport:
        '''Given a polygon, find the centeroid of the polygon, 
        and place a vertiport at that polygon'''
        
        poly_centeroid = polygon.centroid
        return Vertiport(poly_centeroid)
        

    def create_all_vertiports_from_site_polygons(self,polygon_list:List[shapely.Polygon]) -> List[Vertiport]:
        '''Use polygons from polygon_list to create vertiports at each polygon'''
        
        vertiport_list = []
        for polygon in polygon_list:
            vertiport_list.append(self.create_vertiport_from_site_polygon(polygon))
        return vertiport_list
        

    def make_vertiport_site_polygon_dict(self, tag_str):
        #TODO: check if tag_str in tag_list
        # if True, then use tag_str as key for dict
        
        '''Add polygons of specific "tag_str" to an instance dictionary called self.poly_dict.
        These polygons will be used to create vertiports using OSMNx tags'''

        #                                                               tag_str: 'commercial' etc. 
        try:
            assert tag_str in self.vertiport_tags.keys()
        except:
            raise AssertionError('airspace - make_vertiport_site_polygon_dict() is not using correct tag_str')
        
        self.vertiport_site_polygon_dict[tag_str] = [obj for obj in self.vertiport_utm[tag_str].geometry if isinstance(obj, shapely.Polygon)]

        return None
    


    def assign_region_to_vertiports(self, vertiport_list:List[Vertiport], num_regions) -> List[Vertiport]:
        '''Assign regions to each vertioport from vertiport list. '''

        location_tuple = [(vertiport.x, vertiport.y) for vertiport in vertiport_list]
        
        #! region build process needs to be separated 
        # region build process - 
        #           n_clusters needs to be a variable 
        kmeans = KM(n_clusters=num_regions, random_state=0, n_init="auto").fit(location_tuple)
        
        # print(f' These are the labels: {np.unique(kmeans.labels_)}')


        for i in range(len(kmeans.labels_)):
            vertiport = vertiport_list[i]
            vertiport.region = kmeans.labels_[i]

    
        return vertiport_list


    #! This method is used to build self.regions_dict
    def assign_vertiports_to_regions(self, vertiport_list:List[Vertiport], num_regions:int) -> Dict:

        region_vertiport_dict = {}
        for region_id in range(num_regions):
            region_vertiport_dict[region_id] = []
            for vertiport in vertiport_list:
                if vertiport.region == region_id:
                    region_vertiport_dict[region_id].append(vertiport)
        

        return region_vertiport_dict
                    

    def sample_vertiport_from_region(self, region_dict:Dict, n_sample_from_region:int = 1):
        '''From the dictionary of regions with vertiports, 
        sample "n_sample_from_region" number of vertiports from vertiports list of that region'''

        sampled_vertiports = []
        
        for region in region_dict.keys():
            sampled_vertiports += random.sample(region_dict[region], n_sample_from_region)
        
        return sampled_vertiports
    


    # VERTIPORT CREATION - OPTION 1
    def create_n_random_vertiports(self, num_vertiports: int, seed = None) -> None:
        """
        Creates a specified number of random vertiports within the airspace.

        Args:
            num_vertiports (int): The number of vertiports to create.

        Returns:
            None

        Side Effects:
            - Creates the vertiports and updates the vertiports in the airspace list.
        """

        # Set seed if provided
        if seed is not None:
            print(f"Creating vertiports with seed: {seed}")
            random.seed(seed)
            np.random.seed(seed)

        if num_vertiports > self.number_of_vertiports:
            raise RuntimeError('Exceeds vertiport number defined for initialization')

        if self.airspace_tag_list:
            for tag_value in self.location_tags.keys():
                sample_space = self.location_utm_gdf.iloc[0,0].difference(
                    self.location_utm_buffer[tag_value].unary_union
                )
            sample_space_gdf = GeoSeries(sample_space)
        else: 
            sample_space = self.location_utm_gdf
            sample_space_gdf = sample_space.geometry

        
        sample_vertiport: GeoSeries = sample_space_gdf.sample_points(num_vertiports, rng=seed)#TODO: change seed to rng, to avoid warning 
        sample_vertiport_array: np.ndarray = shapely.get_parts(sample_vertiport[0])

        for location in sample_vertiport_array:
            self.vertiport_list.append(
                Vertiport(location=location, uav_list=[])
            )

        print(f"Created {len(self.vertiport_list)} vertiports with seed {seed}")

    # VERTIPORT CREATION - OPTION 2
    # NOT used for VP design problem 
    def create_vertiports_from_regions(self, tag_str, num_regions, n_sample_from_region):
        '''create vertiports and update vertiport_list by adding,
            n_sample_from_region vertiports to vertiport_list'''
        
        #TODO: place a check
        # check if self.vertiport_site_polygon_dict is an attribute if not DO SOMETHING -- ??
        try: 
            assert hasattr(self, 'vertiport_site_polygon_dict')
        except:
            AttributeError("Missing vertiport_site_polygon_dict, __init__'s vertiport_tag_list is empty")
        #step 1 - makes self.poly_dict
        self.make_vertiport_site_polygon_dict(tag_str)
        #step 2
        vertiport_list = self.create_all_vertiports_from_site_polygons(self.vertiport_site_polygon_dict[tag_str])
        
        
        #step 3
        vertiport_list_with_region = self.assign_region_to_vertiports(vertiport_list, num_regions)
        
        #step 4
        regions_dict = self.assign_vertiports_to_regions(vertiport_list_with_region, num_regions)
        # for region in regions_dict.keys():
        #     print(f'Region {region} has {len(regions_dict[region])} vertiports')

        #step 5
        self.vertiport_list += self.sample_vertiport_from_region(regions_dict, n_sample_from_region)
        
        return None
    



    # 1. Make sure vertiports are present in vertiport list 





    # 2. Write a method for region builder 
    #       use map and subdivide into number of regions
    #       how to take a map polygon and divide using a primitive shape like rectangle or hexagon 



    # 3. combine the above in assign_region_to_vertiports method
    #       have a special case for K-Means since it does not divide the map area into regions rather uses vertiport locations 


    #! CHANGE: 
    # 1. use vertiports_list and regions_list 
    # 2. provide mapping_criteria - map vertiport to region 

    def build_pattern(self, center:Tuple[float], diag_dist:float):
        """make verticies of square from center at sqrt2 dist"""
        loc_list:List[Tuple[float,float]] = []
        for n in range(4):
            _loc_x, _loc_y = math.ceil(center[0] + diag_dist*math.cos(math.pi/2*n + math.pi/4)), math.ceil(center[1] + diag_dist*math.sin(math.pi/2*n + math.pi/4))
            loc_list.append((float(_loc_x), float(_loc_y)))
        return loc_list

    def make_regions_dict_vp_des_test_mode(self) -> None:
        '''In test_mode from a given center location we are making 4 regions at the vertices of square centered at point. 
        With each vertex as new center of region we are making four vertiports around that center again. 
        This gives us 4 regions with 4 vertiports each in each region. '''
        self.regions_dict = {}
        centeroid = self.location_utm_gdf.centroid
        center = (centeroid.x, centeroid.y)
        region_center_list = self.build_pattern(center, 5000)
        region = 0
        for region_center in region_center_list:
            _vertiport_list = []
            vertiport_centers_list = self.build_pattern(region_center, 1000)
            for i,vertiport_center in enumerate(vertiport_centers_list):
                _vp = Vertiport(Point(vertiport_center[0], vertiport_center[1]))
                _vp.region = region
                _vp.vp_id_for_region = i
                _vertiport_list.append(_vp)
            self.regions_dict[region] = _vertiport_list
            region += 1
        
        self.num_regions = len(self.regions_dict.keys())
           


        pass 
    #! ** this method will be removed and replaced with new method **
    def make_regions_dict_vp_des(self, tag_str, num_regions):
        '''Using tag_str, and num_region, make an airspace dict attribute that hold regions and vertiports'''
        #TODO: place a check
        # check if self.vertiport_site_polygon_dict is an attribute if not DO SOMETHING -- ??
        try: 
            assert hasattr(self, 'vertiport_site_polygon_dict')
        except:
            AttributeError("Missing vertiport_site_polygon_dict, __init__'s vertiport_tag_list is empty")
        #step 1 - makes self.poly_dict
        self.make_vertiport_site_polygon_dict(tag_str)
        #step 2
        vertiport_list = self.create_all_vertiports_from_site_polygons(self.vertiport_site_polygon_dict[tag_str])
        
        
        #step 3
        vertiport_list_with_region = self.assign_region_to_vertiports(vertiport_list, num_regions)
        
        #step 4
        self.regions_dict =  self.assign_vertiports_to_regions(vertiport_list_with_region, num_regions)
        
        self.num_regions = len(self.regions_dict.keys())
        
        return None


    def get_random_vertiport_from_region(self, region):
        vertiport_list_of_region = self.regions_dict[region]
        return random.sample(vertiport_list_of_region, k=1)    
    


    def random_fill_remaining_vertiport_slots(self, partial_vertiport_list):
        """Fill remaining spots in the vertiport list 
        with random vertiports selected from remaining regions
        and return a complete vertiport list"""
        # find how many regions there are for this env
        required_vertiports = self.num_regions
        # determine how many vertiports need to be collected 
        region_index_for_sampling =  len(partial_vertiport_list)
        #! WHY IS THIS conditional HERE - its never used 
        if region_index_for_sampling: 
            for region in range(region_index_for_sampling, required_vertiports):
                vertiport = random.sample(self.regions_dict[region], k=1)[0] #! random.sample() returns a list
                partial_vertiport_list.append(vertiport)

        complete_list_vertiport = partial_vertiport_list
        # print(f'In file airspace.random_fill_remaining_vertiport_slots(), printing complete_list_vertiport{complete_list_vertiport}')
        return complete_list_vertiport

        # using the previous information about 
        # how many vertiports there are in partial_vertiport_list and
        # how many more I need 
        # 
        # I will determine the current region to sample from 
        # and fill the remaining requirement for vertiport 





    def set_vertiport_list_vp_design(self, complete_vertiport_list):
        #! why is this +=, that would mean argument is added to previous self.vertiport, 
        #! complete_vertiport_list consists of all required vertiports for running map_env simulation 
        #! complete_vertiport_list comes from airspace.random_fill_remaining_vertiport_slots()
        self.vertiport_list = complete_vertiport_list 
        return None 
    
    def get_vertiports_of_region(self, region):
        vertiports = self.regions_dict[region]
        return vertiports

    
    # def make_region_dict(self, vertiport_list:List[Vertiport], num_regions:int) -> Dict:
    #     '''Return a dictionary, with keys as regions and values as list of vertiports of that region.
    #     This will be used later to sample vertiport from each region'''

    #     region_vertiport_dict = {}
    #     for region_id in range(num_regions):
    #         region_vertiport_dict[region_id] = []
    #         for vertiport in vertiport_list:
    #             if vertiport.region == region_id:
    #                 region_vertiport_dict[region_id].append(vertiport)
        

    #     return region_vertiport_dict
if __name__ == '__main__':
    airspace = Airspace(1, "Austin, Texas, USA", airspace_tag_list=[], vertiport_tag_list=[]) #('building', 'commercial')
    
    # airspace.create_vertiports_from_regions('commercial', num_regions=5, n_sample_from_region=2)
    # for vertiport in airspace.get_vertiport_list():
    #     print(vertiport)
    #     print(vertiport.region)

    airspace.make_regions_dict_vp_des_test_mode()
    print(airspace.regions_dict)
    x_arr = []
    y_arr = []
    for region, vp_list in airspace.regions_dict.items():
        for vp in vp_list:
            x_arr.append(vp.x)
            y_arr.append(vp.y)
    plt.scatter(x_arr,y_arr)
    for x, y in zip(x_arr, y_arr):
        plt.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords='offset points', xytext=(4, 4), fontsize=8)
    plt.show()

    print(airspace.num_regions)
    # random_vp = Vertiport(Point(12,13))
    # random_vp.region = 0
    # complete_vp_list = airspace.random_fill_remaining_vertiport_slots([random_vp])
    # for vp in complete_vp_list:
    #     print(vp)
    #     print(vp.region)
    
    # print(f'Before filling vertiports: {airspace.vertiport_list}')
    # airspace.set_vertiport_list_vp_design(complete_vp_list)
    # print(f'After filling vertiports: {airspace.vertiport_list}')
    # print(airspace.number_of_vertiports)
    

    
    

