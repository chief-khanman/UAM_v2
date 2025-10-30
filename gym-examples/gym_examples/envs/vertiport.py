

from shapely import Point
from typing import List


class Vertiport:
    def __init__(self, location: Point, uav_list: list = []) -> None:
        self.id = id(self)
        self.location = location
        self.uav_list: List = uav_list
        # vertiport capacity 
        self.landing_takeoff_capacity = 4
        # vertiport region id/number
        self.region = None
        self.vp_id_for_region = None
        # passenger arrival rate - an exponential distribution learned from metro data
        

    def __repr__(
        self,
    ) -> str:
        return "Vertiport({location}, {uav_list})".format(
            location=self.location, uav_list=self.uav_list
        )
    
    # Add x and y properties that delegate to the location Point object for rendering
    @property
    def x(self):
        return self.location.x
    
    @property
    def y(self):
        return self.location.y

    def landing_queue(self):
        pass


    def takeoff_queue(self):
        pass


    def get_uav_list():
        pass

    

if __name__ == '__main__':
    random_vertiport = Vertiport(Point(12,13))
    print(random_vertiport.location)