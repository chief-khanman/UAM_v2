import geopandas as gpd
from osmnx import features as ox_features
from osmnx import geocode_to_gdf as geocode_to_gdf 
from osmnx import projection as ox_projection



class Airspace:
    def __init__(self, location_name:str, buffer_radius = 500):

        self.location_name = location_name #'Austin, Texas, USA'
        self.buffer_radius = buffer_radius
        
        #TODO - Buffer area needs to be added to all location of interest as they are added to the airspace 
        #self.buffer_area = buffer_area
        
        #location
        location_gdf = geocode_to_gdf(location_name) #converts named geocode - 'Austin,Texas' location to gdf 
        self.location_utm_gdf:gpd.GeoDataFrame = ox_projection.project_gdf(location_gdf) #default projection - UTM projection #! GeoDataFrame has deprication warning - need quick fix
        self.location_utm_gdf['boundary'] = self.location_utm_gdf.boundary #adding column 'boundary' 
        
        #hospital        
        location_hospital = ox_features.features_from_polygon(location_gdf['geometry'][0], tags={'building':'hospital'})
        self.location_utm_hospital = ox_projection.project_gdf(location_hospital)
        self.location_utm_hospital_buffer = self.location_utm_hospital.buffer(self.buffer_radius) # 500 meter buffer area
        
        #self._location_property -> (private property) lists the properties of the location, which is a gpd.GeoDataFrame
    
    def __repr__(self) -> str:
        return ('Airspace({location_name})'.format(location_name = self.location_name))
    

    #TODO - Look at system design principles to choose one of the ways to populate the area with 1) hospitals 2) airport and airspace 3) school etc.