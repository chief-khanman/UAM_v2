import numpy as np
import math
from typing import List, Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import shapely
from shapely.geometry import Point, Polygon, LineString
from sensor_template import SensorTemplate
from uav_v2_template import UAV_v2_template
from auto_uav_v2 import Auto_UAV_v2
from airspace import Airspace  
from atc import ATC


@dataclass
class SensorDetection:
    """Enhanced detection data class for camera and lidar sensors"""
    detection_id: str
    detection_type: str  # 'uav', 'restricted_area', 'unknown'
    distance: float
    bearing: float  # angle from sensor to detected object
    confidence: float
    timestamp: float
    position: Tuple[float, float]
    sensor_type: str  # 'camera' or 'lidar'
    additional_data: Dict[str, Any] = None


class Camera2DSensor(SensorTemplate):
    """
    2D Camera sensor that follows the SensorTemplate interface.
    Provides realistic camera field of view detection for UAM applications.
    """
    
    def __init__(self, airspace:Airspace, atc:ATC, **camera_params):
        """
        Initialize Camera2D sensor with template compliance.
        
        Args:
            airspace: Airspace object containing restricted areas
            atc: ATC object for UAV management
            **camera_params: Camera-specific parameters
        """
        self.airspace = airspace
        self.atc = atc
        
        # Camera-specific parameters
        self.fov = camera_params.get('fov', np.pi / 3)  # 60 degrees default
        self.range = camera_params.get('range', 200.0)
        self.resolution = camera_params.get('resolution', (640, 480))
        self.min_object_size = camera_params.get('min_object_size', 5.0)
        self.detection_threshold = camera_params.get('detection_threshold', 0.1)
        self.pixels_per_degree = camera_params.get('pixels_per_degree', 10.0)
        
        # Detection data storage
        self.last_detections: List[SensorDetection] = []
        self.detection_history: List[List[SensorDetection]] = []
        
    def get_data(self, self_uav) -> Tuple[List, List]:
        """
        Main data collection method following template interface.
        
        Args:
            self_uav: The UAV using this sensor
            
        Returns:
            Tuple of (other_uav_data, ra_data) as Lists of Dicts
        """
        # Get detections from camera's field of view
        uav_detections = self.get_uav_detection(self_uav)
        ra_detections = self.get_ra_detection(self_uav)
        
        return uav_detections, ra_detections
    
    def get_uav_detection(self, self_uav) -> List[Dict]:
        """
        Detect other UAVs within camera's field of view.
        
        Args:
            self_uav: The UAV using this sensor
            
        Returns:
            List of dictionaries containing UAV detection data
        """
        other_uav_data_list = []
        uav_list = self.atc.get_uav_list()
        
        for uav in uav_list:
            if uav.id != self_uav.id:
                # Check if UAV is within camera range
                distance = self_uav.current_position.distance(uav.current_position)
                if distance <= self.range:
                    # Check if UAV is within field of view
                    if self._is_in_fov(self_uav, uav):
                        # Calculate bearing relative to UAV's heading
                        bearing = self._calculate_bearing(self_uav, uav)
                        confidence = self._calculate_confidence(distance, uav)
                        
                        # Convert to pixel coordinates
                        pixel_x = self._world_to_pixel_x(bearing)
                        pixel_y = self._world_to_pixel_y(distance, getattr(uav, 'height', 20))
                        
                        other_uav_data = {
                            'other_uav_id': uav.id,
                            'other_uav_current_position': uav.current_position,
                            'other_uav_current_speed': uav.current_speed,
                            'other_uav_current_heading': uav.current_heading,
                            'other_uav_radius': uav.radius,
                            'detection_distance': distance,
                            'detection_bearing': bearing,
                            'detection_confidence': confidence,
                            'pixel_coordinates': (pixel_x, pixel_y),
                            'sensor_type': 'camera'
                        }
                        other_uav_data_list.append(other_uav_data)
        
        return other_uav_data_list
    
    def get_ra_detection(self, self_uav) -> List[Dict]:
        """
        Detect restricted areas within camera's field of view.
        
        Args:
            self_uav: The UAV using this sensor
            
        Returns:
            List of dictionaries containing restricted area detection data
        """
        ra_data = []
        
        # Create camera FOV polygon for intersection testing
        camera_fov_polygon = self._create_fov_polygon(self_uav)
        
        # Check intersection with each restricted area
        if hasattr(self.airspace, 'location_tags'):
            for tag_value in self.airspace.location_tags.keys():
                restricted_areas_buffer = self.airspace.location_utm_buffer[tag_value]
                
                for i in range(len(restricted_areas_buffer)):
                    restricted_area = restricted_areas_buffer.iloc[i]
                    
                    # Extract geometry
                    if hasattr(restricted_area, 'geometry'):
                        restricted_geometry = restricted_area.geometry
                    else:
                        restricted_geometry = restricted_area
                    
                    # Check if restricted area intersects with camera FOV
                    if camera_fov_polygon.intersects(restricted_geometry):
                        distance = self_uav.current_position.distance(restricted_geometry.boundary)
                        
                        # Calculate bearing from UAV to restricted area centroid
                        ra_heading = math.atan2(
                            (restricted_geometry.centroid.y - self_uav.current_position.y),
                            (restricted_geometry.centroid.x - self_uav.current_position.x)
                        )
                        
                        # Convert to relative bearing (relative to UAV heading)
                        relative_bearing = ra_heading - self_uav.current_heading
                        relative_bearing = math.atan2(math.sin(relative_bearing), math.cos(relative_bearing))
                        
                        confidence = self._calculate_ra_confidence(distance)
                        
                        ra_data.append({
                            'type': tag_value,
                            'distance': distance,
                            'ra_heading': ra_heading,
                            'relative_bearing': relative_bearing,
                            'area': restricted_geometry,
                            'detection_confidence': confidence,
                            'sensor_type': 'camera'
                        })
        
        return ra_data
    
    def get_nmac(self, self_uav) -> Tuple[bool, List]:
        """
        Check for Near Mid-Air Collision within camera's detection capability.
        Enhanced with camera-specific detection logic.
        """
        nmac_list = []
        uav_list = self.atc.get_uav_list()
        
        deactivation_flag = self.deactivate_nmac(self_uav)
        if deactivation_flag:
            return False, []
        
        for uav in uav_list:
            if uav.id == self_uav.id:
                continue
            
            # Check if NMAC radii intersect AND within camera's field of view
            nmac_intersection = self_uav.current_position.buffer(self_uav.nmac_radius).intersects(
                uav.current_position.buffer(uav.nmac_radius)
            )
            
            if nmac_intersection and self._is_in_fov(self_uav, uav):
                nmac_list.append(uav)
        
        return len(nmac_list) > 0, nmac_list
    
    def get_uav_collision(self, self_uav) -> Tuple[bool, Any]:
        """
        Check for UAV collision with camera-enhanced detection.
        """
        deactivate_collision_flag = self.deactivate_collision(self_uav)
        if deactivate_collision_flag:
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
    
    def _is_in_fov(self, self_uav, target_uav) -> bool:
        """Check if target UAV is within camera's field of view"""
        # Calculate angle from self_uav to target_uav
        angle_to_target = math.atan2(
            target_uav.current_position.y - self_uav.current_position.y,
            target_uav.current_position.x - self_uav.current_position.x
        )
        
        # Calculate relative angle to UAV's heading
        relative_angle = angle_to_target - self_uav.current_heading
        relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))
        
        # Check if within FOV
        return abs(relative_angle) <= self.fov / 2
    
    def _calculate_bearing(self, self_uav, target_uav) -> float:
        """Calculate bearing from self_uav to target_uav relative to self_uav's heading"""
        angle_to_target = math.atan2(
            target_uav.current_position.y - self_uav.current_position.y,
            target_uav.current_position.x - self_uav.current_position.x
        )
        return angle_to_target - self_uav.current_heading
    
    def _calculate_confidence(self, distance: float, target_uav) -> float:
        """Calculate detection confidence based on distance and target properties"""
        max_confidence = 0.95
        min_confidence = 0.3
        confidence_decay = 0.003
        
        confidence = max_confidence * np.exp(-distance * confidence_decay)
        
        # Adjust based on target size if available
        if hasattr(target_uav, 'radius'):
            size_factor = min(1.0, target_uav.radius / 10.0)
            confidence *= (0.5 + 0.5 * size_factor)
        
        return max(min_confidence, min(max_confidence, confidence))
    
    def _calculate_ra_confidence(self, distance: float) -> float:
        """Calculate restricted area detection confidence"""
        max_confidence = 0.90
        min_confidence = 0.4
        confidence_decay = 0.002
        
        confidence = max_confidence * np.exp(-distance * confidence_decay)
        return max(min_confidence, min(max_confidence, confidence))
    
    def _world_to_pixel_x(self, bearing: float) -> int:
        """Convert world bearing to pixel X coordinate"""
        normalized_bearing = (bearing + self.fov / 2) / self.fov
        return int(normalized_bearing * self.resolution[0])
    
    def _world_to_pixel_y(self, distance: float, object_height: float) -> int:
        """Convert world distance to pixel Y coordinate"""
        apparent_height = (object_height * self.range) / distance
        return int(self.resolution[1] / 2 - apparent_height / 2)
    
    def _create_fov_polygon(self, self_uav) -> Polygon:
        """Create a polygon representing the camera's field of view"""
        x, y = self_uav.current_position.x, self_uav.current_position.y
        heading = self_uav.current_heading
        
        # Create FOV triangle/sector
        angles = np.linspace(
            heading - self.fov / 2,
            heading + self.fov / 2,
            10  # Number of points for approximation
        )
        
        # FOV polygon points
        points = [(x, y)]  # Start at UAV position
        for angle in angles:
            px = x + self.range * np.cos(angle)
            py = y + self.range * np.sin(angle)
            points.append((px, py))
        points.append((x, y))  # Close polygon
        
        return Polygon(points)
    
    def deactivate_nmac(self, uav) -> bool:
        """Deactivate NMAC detection near start/end points"""
        return (uav.current_position.distance(uav.start) <= 100 or 
                uav.current_position.distance(uav.end) <= 100)
    
    def deactivate_collision(self, uav) -> bool:
        """Deactivate collision detection near start/end points"""
        return (uav.current_position.distance(uav.start) <= 100 or 
                uav.current_position.distance(uav.end) <= 100)
    
    def configure_camera(self, **params):
        """Update camera configuration parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)


class LiDAR2DSensor(SensorTemplate):
    """
    2D LiDAR sensor that follows the SensorTemplate interface.
    Provides 360-degree scanning with point cloud generation for UAM applications.
    """
    
    def __init__(self, airspace, atc, **lidar_params):
        """
        Initialize LiDAR2D sensor with template compliance.
        
        Args:
            airspace: Airspace object containing restricted areas
            atc: ATC object for UAV management
            **lidar_params: LiDAR-specific parameters
        """
        self.airspace = airspace
        self.atc = atc
        
        # LiDAR-specific parameters
        self.range = lidar_params.get('range', 300.0)
        self.num_rays = lidar_params.get('num_rays', 64)
        self.scan_angle = lidar_params.get('scan_angle', 2 * np.pi)  # 360 degrees
        self.noise_level = lidar_params.get('noise_level', 0.01)
        self.angular_resolution = self.scan_angle / self.num_rays
        
        # Point cloud and scan data
        self.point_cloud: List[Dict] = []
        self.scan_data: List[Dict] = []
        self.last_detections: List[SensorDetection] = []
        
    def get_data(self, self_uav) -> Tuple[List, List]:
        """
        Main data collection method following template interface.
        
        Args:
            self_uav: The UAV using this sensor
            
        Returns:
            Tuple of (other_uav_data, ra_data) as Lists of Dicts
        """
        # Perform LiDAR scan
        self._perform_scan(self_uav)
        
        # Get detections from point cloud analysis
        uav_detections = self.get_uav_detection(self_uav)
        ra_detections = self.get_ra_detection(self_uav)
        
        return uav_detections, ra_detections
    
    def _perform_scan(self, self_uav):
        """Perform LiDAR scan and generate point cloud"""
        self.point_cloud.clear()
        self.scan_data.clear()
        
        # Cast rays in all directions
        for i in range(self.num_rays):
            ray_angle = self_uav.current_heading + (i * self.angular_resolution) - (self.scan_angle / 2)
            ray_data = self._cast_ray(self_uav, ray_angle)
            self.scan_data.append(ray_data)
            
            if ray_data['hit']:
                point = {
                    'x': ray_data['hit_x'],
                    'y': ray_data['hit_y'],
                    'distance': ray_data['distance'],
                    'angle': ray_angle,
                    'intensity': ray_data['intensity'],
                    'object_type': ray_data.get('object_type', 'unknown'),
                    'object_id': ray_data.get('object_id', 'unknown')
                }
                self.point_cloud.append(point)
    
    def _cast_ray(self, self_uav, angle: float) -> Dict:
        """Cast a single LiDAR ray and find intersection"""
        ray_start = Point(self_uav.current_position.x, self_uav.current_position.y)
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        ray_end = Point(
            ray_start.x + ray_dir[0] * self.range,
            ray_start.y + ray_dir[1] * self.range
        )
        ray_line = LineString([ray_start, ray_end])
        
        closest_hit = None
        min_distance = self.range
        hit_object_type = 'unknown'
        hit_object_id = 'unknown'
        
        # Check intersection with other UAVs
        uav_list = self.atc.get_uav_list()
        for uav in uav_list:
            if uav.id != self_uav.id:
                uav_circle = uav.current_position.buffer(uav.radius)
                if ray_line.intersects(uav_circle):
                    intersection_points = ray_line.intersection(uav_circle.boundary)
                    if not intersection_points.is_empty:
                        # Get closest intersection point
                        if hasattr(intersection_points, 'geoms'):
                            # Multiple intersection points
                            distances = [ray_start.distance(pt) for pt in intersection_points.geoms]
                            min_idx = np.argmin(distances)
                            closest_distance = distances[min_idx]
                            closest_point = intersection_points.geoms[min_idx]
                        else:
                            # Single intersection point
                            closest_distance = ray_start.distance(intersection_points)
                            closest_point = intersection_points
                        
                        if closest_distance < min_distance:
                            min_distance = closest_distance
                            closest_hit = closest_point
                            hit_object_type = 'uav'
                            hit_object_id = uav.id
        
        # Check intersection with restricted areas
        if hasattr(self.airspace, 'location_tags'):
            for tag_value in self.airspace.location_tags.keys():
                restricted_areas = self.airspace.location_utm[tag_value]
                
                for i in range(len(restricted_areas)):
                    restricted_area = restricted_areas.iloc[i]
                    if hasattr(restricted_area, 'geometry'):
                        restricted_geometry = restricted_area.geometry
                    else:
                        restricted_geometry = restricted_area
                    
                    try:
                        if ray_line.intersects(restricted_geometry):
                            intersection = ray_line.intersection(restricted_geometry.boundary)
                            if not intersection.is_empty:
                                if hasattr(intersection, 'geoms'):
                                    distances = [ray_start.distance(pt) for pt in intersection.geoms]
                                    closest_distance = min(distances)
                                    closest_point = intersection.geoms[np.argmin(distances)]
                                else:
                                    closest_distance = ray_start.distance(intersection)
                                    closest_point = intersection
                                
                                if closest_distance < min_distance:
                                    min_distance = closest_distance
                                    closest_hit = closest_point
                                    hit_object_type = tag_value
                                    hit_object_id = f"{tag_value}_{i}"
                    except:
                        continue
        
        # Add noise to simulate real sensor
        noise = (np.random.random() - 0.5) * self.noise_level * min_distance
        noisy_distance = max(0, min_distance + noise)
        
        if closest_hit:
            # Calculate hit position with noise
            hit_x = ray_start.x + ray_dir[0] * noisy_distance
            hit_y = ray_start.y + ray_dir[1] * noisy_distance
            
            return {
                'angle': angle,
                'distance': noisy_distance,
                'hit': True,
                'hit_x': hit_x,
                'hit_y': hit_y,
                'intensity': self._calculate_intensity(hit_object_type, noisy_distance),
                'object_type': hit_object_type,
                'object_id': hit_object_id
            }
        else:
            return {
                'angle': angle,
                'distance': self.range,
                'hit': False,
                'intensity': 0.0,
                'object_type': 'none',
                'object_id': 'none'
            }
    
    def get_uav_detection(self, self_uav) -> List[Dict]:
        """
        Detect other UAVs from LiDAR point cloud analysis.
        """
        other_uav_data_list = []
        
        # Cluster points by UAV
        uav_clusters = {}
        for point in self.point_cloud:
            if point['object_type'] == 'uav':
                uav_id = point['object_id']
                if uav_id not in uav_clusters:
                    uav_clusters[uav_id] = []
                uav_clusters[uav_id].append(point)
        
        # Process each UAV cluster
        for uav_id, points in uav_clusters.items():
            if len(points) >= 2:  # Minimum points for reliable detection
                # Calculate cluster properties
                centroid_x = np.mean([p['x'] for p in points])
                centroid_y = np.mean([p['y'] for p in points])
                avg_distance = np.mean([p['distance'] for p in points])
                avg_intensity = np.mean([p['intensity'] for p in points])
                
                # Get UAV object for additional data
                uav_list = self.atc.get_uav_list()
                target_uav = None
                for uav in uav_list:
                    if uav.id == uav_id:
                        target_uav = uav
                        break
                
                if target_uav:
                    bearing = math.atan2(
                        centroid_y - self_uav.current_position.y,
                        centroid_x - self_uav.current_position.x
                    ) - self_uav.current_heading
                    
                    confidence = self._calculate_detection_confidence(len(points), avg_distance, avg_intensity)
                    
                    other_uav_data = {
                        'other_uav_id': uav_id,
                        'other_uav_current_position': target_uav.current_position,
                        'other_uav_current_speed': target_uav.current_speed,
                        'other_uav_current_heading': target_uav.current_heading,
                        'other_uav_radius': target_uav.radius,
                        'detection_distance': avg_distance,
                        'detection_bearing': bearing,
                        'detection_confidence': confidence,
                        'point_count': len(points),
                        'avg_intensity': avg_intensity,
                        'centroid': (centroid_x, centroid_y),
                        'sensor_type': 'lidar'
                    }
                    other_uav_data_list.append(other_uav_data)
        
        return other_uav_data_list
    
    def get_ra_detection(self, self_uav) -> List[Dict]:
        """
        Detect restricted areas from LiDAR point cloud analysis.
        """
        ra_data = []
        
        # Cluster points by restricted area type
        ra_clusters = {}
        for point in self.point_cloud:
            if point['object_type'] not in ['uav', 'none', 'unknown']:
                ra_type = point['object_type']
                if ra_type not in ra_clusters:
                    ra_clusters[ra_type] = []
                ra_clusters[ra_type].append(point)
        
        # Process each restricted area cluster
        for ra_type, points in ra_clusters.items():
            if len(points) >= 3:  # Minimum points for area detection
                # Calculate cluster properties
                centroid_x = np.mean([p['x'] for p in points])
                centroid_y = np.mean([p['y'] for p in points])
                avg_distance = np.mean([p['distance'] for p in points])
                avg_intensity = np.mean([p['intensity'] for p in points])
                
                # Calculate bearing to centroid
                ra_heading = math.atan2(
                    centroid_y - self_uav.current_position.y,
                    centroid_x - self_uav.current_position.x
                )
                relative_bearing = ra_heading - self_uav.current_heading
                
                confidence = self._calculate_detection_confidence(len(points), avg_distance, avg_intensity)
                
                ra_data.append({
                    'type': ra_type,
                    'distance': avg_distance,
                    'ra_heading': ra_heading,
                    'relative_bearing': relative_bearing,
                    'detection_confidence': confidence,
                    'point_count': len(points),
                    'avg_intensity': avg_intensity,
                    'centroid': (centroid_x, centroid_y),
                    'sensor_type': 'lidar'
                })
        
        return ra_data
    
    def get_nmac(self, self_uav) -> Tuple[bool, List]:
        """
        Check for Near Mid-Air Collision with LiDAR-enhanced detection.
        """
        nmac_list = []
        uav_list = self.atc.get_uav_list()
        
        deactivation_flag = self.deactivate_nmac(self_uav)
        if deactivation_flag:
            return False, []
        
        for uav in uav_list:
            if uav.id == self_uav.id:
                continue
            
            # Check if NMAC radii intersect
            nmac_intersection = self_uav.current_position.buffer(self_uav.nmac_radius).intersects(
                uav.current_position.buffer(uav.nmac_radius)
            )
            
            if nmac_intersection:
                # Additional check: verify LiDAR can actually detect the UAV
                distance = self_uav.current_position.distance(uav.current_position)
                if distance <= self.range:  # Within LiDAR range
                    nmac_list.append(uav)
        
        return len(nmac_list) > 0, nmac_list
    
    def get_uav_collision(self, self_uav) -> Tuple[bool, Any]:
        """
        Check for UAV collision with LiDAR verification.
        """
        deactivate_collision_flag = self.deactivate_collision(self_uav)
        if deactivate_collision_flag:
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
    
    def _calculate_intensity(self, object_type: str, distance: float) -> float:
        """Calculate return intensity based on object type and distance"""
        material_reflectivity = {
            'uav': 0.6,
            'building': 0.8,
            'hospital': 0.8,
            'airport': 0.9,
            'tree': 0.3,
            'unknown': 0.5
        }
        
        base_intensity = material_reflectivity.get(object_type, 0.5)
        distance_attenuation = np.exp(-distance / (self.range * 0.5))
        
        return base_intensity * distance_attenuation
    
    def _calculate_detection_confidence(self, point_count: int, distance: float, intensity: float) -> float:
        """Calculate detection confidence based on point cloud characteristics"""
        # Base confidence from point count
        point_confidence = min(0.9, point_count / 10.0)
        
        # Distance factor
        distance_factor = np.exp(-distance / (self.range * 0.3))
        
        # Intensity factor
        intensity_factor = min(1.0, intensity * 2.0)
        
        # Combined confidence
        confidence = 0.4 * point_confidence + 0.4 * distance_factor + 0.2 * intensity_factor
        
        return max(0.1, min(0.95, confidence))
    
    def deactivate_nmac(self, uav) -> bool:
        """Deactivate NMAC detection near start/end points"""
        return (uav.current_position.distance(uav.start) <= 100 or 
                uav.current_position.distance(uav.end) <= 100)
    
    def deactivate_collision(self, uav) -> bool:
        """Deactivate collision detection near start/end points"""
        return (uav.current_position.distance(uav.start) <= 100 or 
                uav.current_position.distance(uav.end) <= 100)
    
    def configure_lidar(self, **params):
        """Update LiDAR configuration parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # Recalculate angular resolution if parameters changed
        self.angular_resolution = self.scan_angle / self.num_rays
    
    def get_point_cloud(self) -> List[Dict]:
        """Get the current point cloud data"""
        return self.point_cloud.copy()
    
    def get_scan_data(self) -> List[Dict]:
        """Get the current scan data"""
        return self.scan_data.copy()


class MultiSensorFusion(SensorTemplate):
    """
    Complete sensor fusion implementation that combines Camera2D and LiDAR2D sensors
    while maintaining full SensorTemplate interface compatibility.
    
    Prevents duplicate detections and provides enhanced coverage through sensor fusion.
    """
    
    def __init__(self, camera_sensor: Camera2DSensor, lidar_sensor: LiDAR2DSensor):
        """
        Initialize MultiSensorFusion with camera and LiDAR sensors.
        
        Args:
            camera_sensor: Camera2DSensor instance
            lidar_sensor: LiDAR2DSensor instance
        """
        # Store sensor references
        self.camera = camera_sensor
        self.lidar = lidar_sensor
        
        # Inherit airspace and atc from sensors for SensorTemplate compliance
        self.airspace = camera_sensor.airspace
        self.atc = camera_sensor.atc
        
        # Fusion configuration
        self.camera_weight = 0.6  # Camera typically more reliable for identification
        self.lidar_weight = 0.4   # LiDAR better for distance/geometry
        self.proximity_threshold = 50.0  # meters - for matching RA detections
        
    def get_data(self, self_uav) -> Tuple[List[Dict], List[Dict]]:
        """
        Main data collection method - fuses data from both sensors.
        
        Args:
            self_uav: The UAV using this sensor system
            
        Returns:
            Tuple of (fused_uav_detections, fused_ra_detections) without duplicates
        """
        # Get data from both sensors
        camera_uav, camera_ra = self.camera.get_data(self_uav)
        lidar_uav, lidar_ra = self.lidar.get_data(self_uav)
        
        # Fuse detections
        fused_uav = self._fuse_uav_detections(camera_uav, lidar_uav)
        fused_ra = self._fuse_ra_detections(camera_ra, lidar_ra)
        
        return fused_uav, fused_ra
    
    def get_uav_detection(self, self_uav) -> List[Dict]:
        """Get fused UAV detections (part of SensorTemplate interface)"""
        fused_uav, _ = self.get_data(self_uav)
        return fused_uav
    
    def get_ra_detection(self, self_uav) -> List[Dict]:
        """Get fused restricted area detections (part of SensorTemplate interface)"""
        _, fused_ra = self.get_data(self_uav)
        return fused_ra
    
    def get_nmac(self, self_uav) -> Tuple[bool, List]:
        """
        Check for Near Mid-Air Collision using both sensors.
        Combines results to provide comprehensive NMAC detection.
        """
        # Get NMAC results from both sensors
        camera_nmac, camera_list = self.camera.get_nmac(self_uav)
        lidar_nmac, lidar_list = self.lidar.get_nmac(self_uav)
        
        # Combine NMAC lists without duplicates
        combined_nmac_uavs = []
        seen_uav_ids = set()
        
        # Add UAVs from both lists, avoiding duplicates
        for uav in camera_list + lidar_list:
            if uav.id not in seen_uav_ids:
                combined_nmac_uavs.append(uav)
                seen_uav_ids.add(uav.id)
        
        # NMAC detected if either sensor detected it
        nmac_detected = camera_nmac or lidar_nmac
        
        return nmac_detected, combined_nmac_uavs
    
    def get_uav_collision(self, self_uav) -> Tuple[bool, Any]:
        """
        Check for UAV collision. Since this is a geometric check,
        delegate to camera sensor (both would give same result).
        """
        return self.camera.get_uav_collision(self_uav)
    
    def get_ra_collision(self, self_uav) -> Tuple[bool, Dict]:
        """
        Check for restricted area collision. Delegate to camera sensor
        since collision is geometric and both sensors use same airspace data.
        """
        # Try camera first, fallback to lidar if camera has issues
        try:
            return self.camera.get_ra_collision(self_uav)
        except:
            # Fallback to lidar if camera fails
            return self.lidar.get_ra_collision(self_uav)
    
    def deactivate_nmac(self, uav) -> bool:
        """Deactivate NMAC detection - delegate to camera sensor"""
        return self.camera.deactivate_nmac(uav)
    
    def deactivate_detection(self) -> None:
        """Deactivate detection - delegate to both sensors"""
        if hasattr(self.camera, 'deactivate_detection'):
            self.camera.deactivate_detection()
        if hasattr(self.lidar, 'deactivate_detection'):
            self.lidar.deactivate_detection()
    
    def deactivate_collision(self, uav) -> bool:
        """Deactivate collision detection - delegate to camera sensor"""
        return self.camera.deactivate_collision(uav)
    
    def _fuse_uav_detections(self, camera_detections: List[Dict], lidar_detections: List[Dict]) -> List[Dict]:
        """
        Fuse UAV detections from camera and LiDAR, handling duplicates properly.
        
        Returns single detection per UAV with fused confidence and metadata.
        """
        fused_detections = []
        
        # Create lookup dictionaries by UAV ID
        camera_lookup = {det['other_uav_id']: det for det in camera_detections}
        lidar_lookup = {det['other_uav_id']: det for det in lidar_detections}
        
        # Get all unique UAV IDs detected by either sensor
        all_uav_ids = set(camera_lookup.keys()) | set(lidar_lookup.keys())
        
        for uav_id in all_uav_ids:
            camera_det = camera_lookup.get(uav_id)
            lidar_det = lidar_lookup.get(uav_id)
            
            if camera_det and lidar_det:
                # Both sensors detected - create fused detection
                fused_det = self._create_fused_uav_detection(camera_det, lidar_det)
                fused_detections.append(fused_det)
                
            elif camera_det:
                # Only camera detected - preserve camera detection
                camera_det = camera_det.copy()
                camera_det['sensor_type'] = 'camera_only'
                camera_det['fusion_confidence'] = camera_det['detection_confidence']
                fused_detections.append(camera_det)
                
            elif lidar_det:
                # Only LiDAR detected - preserve LiDAR detection
                lidar_det = lidar_det.copy()
                lidar_det['sensor_type'] = 'lidar_only' 
                lidar_det['fusion_confidence'] = lidar_det['detection_confidence']
                fused_detections.append(lidar_det)
        
        return fused_detections
    
    def _create_fused_uav_detection(self, camera_det: Dict, lidar_det: Dict) -> Dict:
        """
        Create a single fused detection from camera and LiDAR detections of the same UAV.
        """
        # Start with camera detection as base (has pixel coordinates, etc.)
        fused_det = camera_det.copy()
        
        # Fuse confidence using weighted average
        camera_conf = camera_det['detection_confidence']
        lidar_conf = lidar_det['detection_confidence']
        fused_confidence = (self.camera_weight * camera_conf + self.lidar_weight * lidar_conf)
        
        # Use distance average (both sensors measure this)
        camera_dist = camera_det['detection_distance']
        lidar_dist = lidar_det['detection_distance']
        fused_distance = (camera_dist + lidar_dist) / 2
        
        # Update fused detection with combined data
        fused_det.update({
            'detection_confidence': fused_confidence,
            'fusion_confidence': fused_confidence,
            'detection_distance': fused_distance,
            'sensor_type': 'fused',
            'camera_confidence': camera_conf,
            'lidar_confidence': lidar_conf,
            'camera_distance': camera_dist,
            'lidar_distance': lidar_dist,
            'lidar_point_count': lidar_det.get('point_count', 0),
            'lidar_avg_intensity': lidar_det.get('avg_intensity', 0.0),
            'lidar_centroid': lidar_det.get('centroid', None)
        })
        
        return fused_det
    
    def _fuse_ra_detections(self, camera_detections: List[Dict], lidar_detections: List[Dict]) -> List[Dict]:
        """
        Fuse restricted area detections, handling multiple instances of same type.
        
        Uses proximity matching to avoid losing multiple hospital/airport detections.
        """
        fused_detections = []
        used_lidar_indices = set()
        
        # Process camera detections and try to match with LiDAR
        for camera_det in camera_detections:
            best_lidar_match = None
            best_lidar_idx = None
            best_similarity_score = float('inf')
            
            # Find closest LiDAR detection of same type
            for idx, lidar_det in enumerate(lidar_detections):
                if (idx not in used_lidar_indices and 
                    camera_det['type'] == lidar_det['type']):
                    
                    # Improved similarity scoring
                    distance_diff = abs(camera_det.get('distance', 0) - lidar_det.get('distance', 0))
                    angle_diff = abs(camera_det.get('ra_heading', 0) - lidar_det.get('ra_heading', 0))
                    
                    # Normalize angle difference to [0, π]
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff
                    
                    # Combined similarity score (lower is better)
                    # Weight distance more heavily than angle for RA matching
                    similarity_score = 0.7 * distance_diff + 0.3 * (angle_diff * 100)
                    
                    if similarity_score < best_similarity_score and similarity_score < self.proximity_threshold:
                        best_similarity_score = similarity_score
                        best_lidar_match = lidar_det
                        best_lidar_idx = idx
            
            if best_lidar_match:
                # Create fused RA detection
                fused_ra = self._create_fused_ra_detection(camera_det, best_lidar_match)
                fused_detections.append(fused_ra)
                used_lidar_indices.add(best_lidar_idx)
            else:
                # Camera-only RA detection
                camera_det = camera_det.copy()
                camera_det['sensor_type'] = 'camera_only'
                camera_det['fusion_confidence'] = camera_det['detection_confidence']
                fused_detections.append(camera_det)
        
        # Add unmatched LiDAR detections
        for idx, lidar_det in enumerate(lidar_detections):
            if idx not in used_lidar_indices:
                lidar_det = lidar_det.copy()
                lidar_det['sensor_type'] = 'lidar_only'
                lidar_det['fusion_confidence'] = lidar_det['detection_confidence']
                fused_detections.append(lidar_det)
        
        return fused_detections
    
    def _create_fused_ra_detection(self, camera_det: Dict, lidar_det: Dict) -> Dict:
        """Create a single fused RA detection from camera and LiDAR detections."""
        fused_det = camera_det.copy()
        
        # Fuse confidence
        camera_conf = camera_det['detection_confidence']
        lidar_conf = lidar_det['detection_confidence']
        fused_confidence = (self.camera_weight * camera_conf + self.lidar_weight * lidar_conf)
        
        # Use more accurate distance (prefer LiDAR for distance accuracy)
        camera_dist = camera_det['distance']
        lidar_dist = lidar_det['distance']
        fused_distance = (0.3 * camera_dist + 0.7 * lidar_dist)  # Weight LiDAR distance more
        
        # Update fused detection
        fused_det.update({
            'detection_confidence': fused_confidence,
            'fusion_confidence': fused_confidence,
            'distance': fused_distance,
            'sensor_type': 'fused',
            'camera_confidence': camera_conf,
            'lidar_confidence': lidar_conf,
            'camera_distance': camera_dist,
            'lidar_distance': lidar_dist,
            'lidar_point_count': lidar_det.get('point_count', 0),
            'lidar_avg_intensity': lidar_det.get('avg_intensity', 0.0)
        })
        
        return fused_det
    
    def configure_fusion(self, camera_weight: float = 0.6, lidar_weight: float = 0.4, 
                        proximity_threshold: float = 50.0):
        """
        Configure fusion parameters.
        
        Args:
            camera_weight: Weight for camera confidence (0-1)
            lidar_weight: Weight for LiDAR confidence (0-1) 
            proximity_threshold: Max distance for matching RA detections
        """
        # Normalize weights
        total_weight = camera_weight + lidar_weight
        self.camera_weight = camera_weight / total_weight
        self.lidar_weight = lidar_weight / total_weight
        self.proximity_threshold = proximity_threshold
    
    def get_fusion_statistics(self, self_uav) -> Dict[str, Any]:
        """
        Get detailed statistics about sensor fusion performance.
        Useful for analysis and debugging.
        """
        # Get individual sensor data
        camera_uav, camera_ra = self.camera.get_data(self_uav)
        lidar_uav, lidar_ra = self.lidar.get_data(self_uav)
        
        # Get fused data
        fused_uav, fused_ra = self.get_data(self_uav)
        
        # Calculate statistics
        stats = {
            'camera_uav_detections': len(camera_uav),
            'lidar_uav_detections': len(lidar_uav),
            'fused_uav_detections': len(fused_uav),
            'camera_ra_detections': len(camera_ra),
            'lidar_ra_detections': len(lidar_ra),
            'fused_ra_detections': len(fused_ra),
            'uav_fusion_types': {},
            'ra_fusion_types': {},
            'duplicate_prevention': {
                'camera_uav_unique': len(set(d['other_uav_id'] for d in camera_uav)),
                'lidar_uav_unique': len(set(d['other_uav_id'] for d in lidar_uav)),
                'fused_uav_unique': len(set(d['other_uav_id'] for d in fused_uav)),
            }
        }
        
        # Count fusion types
        for det in fused_uav:
            sensor_type = det.get('sensor_type', 'unknown')
            stats['uav_fusion_types'][sensor_type] = stats['uav_fusion_types'].get(sensor_type, 0) + 1
            
        for det in fused_ra:
            sensor_type = det.get('sensor_type', 'unknown')
            stats['ra_fusion_types'][sensor_type] = stats['ra_fusion_types'].get(sensor_type, 0) + 1
        
        return stats
    
    def get_sensor_health(self) -> Dict[str, bool]:
        """Check if both sensors are functioning correctly"""
        return {
            'camera_operational': hasattr(self.camera, 'get_data') and callable(self.camera.get_data),
            'lidar_operational': hasattr(self.lidar, 'get_data') and callable(self.lidar.get_data),
            'fusion_operational': True,
            'airspace_available': self.airspace is not None,
            'atc_available': self.atc is not None
        }

# Example usage and testing
if __name__ == "__main__":
    # Mock classes for testing (replace with actual imports)
    class MockATC:
        def get_uav_list(self):
            return []
    
    class MockAirspace:
        def __init__(self):
            self.location_tags = {}
            self.location_utm = {}
            self.location_utm_buffer = {}
    
    # Example usage
    mock_atc = MockATC()
    mock_airspace = MockAirspace()
    
    # Initialize sensors
    camera = Camera2DSensor(mock_airspace, mock_atc, 
                           fov=np.pi/3, range=200, resolution=(640, 480))
    
    lidar = LiDAR2DSensor(mock_airspace, mock_atc,
                         range=300, num_rays=64, scan_angle=2*np.pi)
    
    # Initialize sensor fusion
    fusion = MultiSensorFusion(camera, lidar)
    
    print("Camera and LiDAR sensors initialized successfully!")
    print(f"Camera FOV: {camera.fov * 180 / np.pi:.1f} degrees")
    print(f"Camera Range: {camera.range} units")
    print(f"LiDAR Range: {lidar.range} units")
    print(f"LiDAR Resolution: {lidar.num_rays} rays")