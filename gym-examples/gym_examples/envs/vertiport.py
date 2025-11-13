

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
        self.set_metrics(0,0,0)
        # metrics
        self.metrics = {'total_throughput':self.total_throughput, 
                        'avg_wait_time':self.avg_wait_time,
                        'avg_delay':self.avg_delay}
        

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


    def set_metrics(self, total_throughput, avg_wait_time, avg_delay):
        
        self.total_throughput = 0
        self.avg_wait_time = 0
        self.avg_delay = 0
        pass


    def get_metrics(self):
        return self.total_throughput, self.avg_wait_time, self.avg_delay 
    

if __name__ == '__main__':
    # vp1 = Vertiport(Point(12,13))
    # vp2 = Vertiport(Point(14,14))
    # vp3 = Vertiport(Point(16,17))
    # vp4 = Vertiport(Point(21,31))

    vp1 = Vertiport(Point(10.,10.))
    vp2 = Vertiport(Point(11.,11.))
    vp3 = Vertiport(Point(12.,12.))
    vp4 = Vertiport(Point(13.,13.))



    vp_list = [vp1, vp2, vp3, vp4]

    best_locations = [(10,10), (11,11), (12,12), (13,13)]
    best_count = 0
    for _vp in vp_list:
        tempxy = (_vp.x, _vp.y)
        if tempxy in best_locations:
            best_count +=1
    if best_count == 4:
        print('Found best')
    else:
        print('Found nothing')
