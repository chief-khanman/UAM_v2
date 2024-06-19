from uav import UAV
from vertiport import Vertiport
from shapely import Point

start_v = Vertiport(Point(0,0))
end_v = Vertiport(Point(5,5))

uav1 = UAV(start_v,end_v)
uav2 = UAV(start_v,end_v)

print(uav1.uav_polygon(uav1.detection_radius).intersects(uav2.make_polygon(uav2.detection_radius)))
