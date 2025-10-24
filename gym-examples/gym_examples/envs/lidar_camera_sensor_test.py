#!/usr/bin/env python3
"""
Complete test script to validate Camera2D, LiDAR2D, and MultiSensorFusion sensors
for urban air mobility simulation environment.

FINAL VERSION: Grey background, consistent styling, proper label positioning
"""

import numpy as np
import math
import time
from typing import List, Dict, Any
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Import matplotlib with proper backend handling
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as patches

# Import the sensor classes (assuming they're in the same directory)
from lidar_camera_sensors import Camera2DSensor, LiDAR2DSensor, MultiSensorFusion


class MockUAV:
    """Mock UAV class with minimum required properties for sensor testing"""
    
    def __init__(self, x: float, y: float, uav_id: str = None, heading: float = 0.0):
        self.id = uav_id or f"uav_{np.random.randint(1000, 9999)}"
        self.current_position = Point(x, y)
        self.current_heading = heading
        self.current_speed = 20.0  # m/s
        self.radius = 17.0
        self.nmac_radius = 200.0
        self.detection_radius = 500.0
        self.mission_complete_distance = 50.0
        
        # Mock start/end points for deactivation logic
        self.start = Point(x - 1000, y - 1000)  # Far away to not trigger deactivation
        self.end = Point(x + 1000, y + 1000)
    
    def move_to(self, x: float, y: float, heading: float = None):
        """Move UAV to new position"""
        self.current_position = Point(x, y)
        if heading is not None:
            self.current_heading = heading


class MockATC:
    """Mock ATC class for managing UAV list"""
    
    def __init__(self):
        self.uav_list: List[MockUAV] = []
    
    def add_uav(self, uav: MockUAV):
        self.uav_list.append(uav)
    
    def get_uav_list(self) -> List[MockUAV]:
        return self.uav_list


class MockAirspace:
    """Mock Airspace class with restricted areas"""
    
    def __init__(self):
        self.location_tags = {
            'hospital': ['hospital_1', 'hospital_2'],
            'airport': ['airport_1']
        }
        
        # Create some mock restricted areas as GeoDataFrame-like structures
        self.location_utm = {
            'hospital': self._create_mock_geodataframe([
                Polygon([(300, 300), (400, 300), (400, 400), (300, 400)]),  # Hospital 1
                Polygon([(700, 600), (800, 600), (800, 700), (700, 700)])   # Hospital 2
            ]),
            'airport': self._create_mock_geodataframe([
                Polygon([(500, 100), (700, 100), (700, 200), (500, 200)])   # Airport
            ])
        }
        
        # Create buffer zones (slightly larger areas)
        self.location_utm_buffer = {
            'hospital': self._create_mock_geodataframe([
                Polygon([(280, 280), (420, 280), (420, 420), (280, 420)]),  # Hospital 1 buffer
                Polygon([(680, 580), (820, 580), (820, 720), (680, 720)])   # Hospital 2 buffer
            ]),
            'airport': self._create_mock_geodataframe([
                Polygon([(480, 80), (720, 80), (720, 220), (480, 220)])     # Airport buffer
            ])
        }
    
    def _create_mock_geodataframe(self, polygons: List[Polygon]):
        """Create a mock GeoDataFrame-like structure"""
        class MockILoc:
            def __init__(self, geometries):
                self.geometries = geometries
            
            def __getitem__(self, index):
                # Return a mock row that has a geometry attribute
                class MockRow:
                    def __init__(self, geometry):
                        self.geometry = geometry
                return MockRow(self.geometries[index])
        
        class MockGeoDataFrame:
            def __init__(self, geometries):
                self.geometries = geometries
                self._iloc = MockILoc(geometries)
            
            def __len__(self):
                return len(self.geometries)
            
            @property
            def iloc(self):
                return self._iloc
        
        return MockGeoDataFrame(polygons)


class SensorTester:
    """Test harness for validating sensor functionality"""
    
    def __init__(self, enable_plotting=False, save_plots=False):
        # Set matplotlib backend based on usage
        if enable_plotting:
            try:
                matplotlib.use('TkAgg')  # Interactive backend
                plt.ion()  # Interactive mode
                print("✓ Interactive plotting enabled (TkAgg)")
            except:
                try:
                    matplotlib.use('Qt5Agg')  # Alternative interactive backend
                    plt.ion()
                    print("✓ Interactive plotting enabled (Qt5Agg)")
                except:
                    print("⚠ No interactive backend available, using save-only mode")
                    enable_plotting = False
                    save_plots = True
                    matplotlib.use('Agg')
        elif save_plots:
            matplotlib.use('Agg')  # Non-interactive backend for saving
            print("✓ Save-plots mode enabled (Agg backend)")
            
        # Create mock environment
        self.atc = MockATC()
        self.airspace = MockAirspace()
        self.enable_plotting = enable_plotting
        self.save_plots = save_plots
        
        # Create test UAV (the one carrying sensors) - this is our "agent"
        self.test_uav = MockUAV(100, 300, "test_uav", heading=0.0)
        
        # Create other UAVs to detect - KEEP USER'S MODIFICATIONS
        self.other_uavs = [
            MockUAV(400, 350, "uav_1", heading=math.pi/2),
            MockUAV(200, 500, "uav_2", heading=math.pi),
            MockUAV(600, 300, "uav_3", heading=-math.pi/4),
        ]
        
        # Add UAVs to ATC
        self.atc.add_uav(self.test_uav)
        for uav in self.other_uavs:
            self.atc.add_uav(uav)
        
        # Initialize sensors
        self.camera = Camera2DSensor(
            self.airspace, 
            self.atc,
            fov=math.pi/3,  # 60 degrees
            range=300.0,
            resolution=(640, 480)
        )
        
        self.lidar = LiDAR2DSensor(
            self.airspace,
            self.atc,
            range=400.0,
            num_rays=32,  # Reduced for clearer visualization
            scan_angle=2*math.pi
        )
        
        # Set up plotting only if requested
        self.fig = None
        self.axes = None
        if self.enable_plotting or self.save_plots:
            try:
                self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
                self.fig.suptitle('UAM Sensor Testing: Camera (Left) and LiDAR (Right)', 
                                fontsize=16, fontweight='bold')
                print("✓ Figure initialized successfully")
            except Exception as e:
                print(f"✗ Could not initialize plotting: {e}")
                self.enable_plotting = False
                self.save_plots = False
        
    def visualize_environment(self, ax, title: str, sensor_type: str = None):
        """
        Visualize the environment using rendering style consistent with map_renderer.py
        """
        ax.clear()
        ax.set_facecolor('#808080')  # Grey background like map renderer
        ax.set_xlim(-50, 900)
        ax.set_ylim(0, 800)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Draw restricted areas - CONSISTENT WITH map_renderer.py
        for area_type, areas in self.airspace.location_utm.items():
            color = 'red'
            for i in range(len(areas)):
                geometry = areas.iloc[i].geometry
                if hasattr(geometry, 'exterior'):
                    coords = list(geometry.exterior.coords)
                    polygon = patches.Polygon(coords, facecolor=color, alpha=0.7, 
                                            edgecolor=color, linewidth=1)
                    ax.add_patch(polygon)
        
        # Draw buffer areas - CONSISTENT WITH map_renderer.py
        for area_type, areas in self.airspace.location_utm_buffer.items():
            color = 'orange'
            for i in range(len(areas)):
                geometry = areas.iloc[i].geometry
                if hasattr(geometry, 'exterior'):
                    coords = list(geometry.exterior.coords)
                    polygon = patches.Polygon(coords, facecolor=color, alpha=0.3, 
                                            edgecolor=color, linewidth=1)
                    ax.add_patch(polygon)
        
        # Draw other UAVs - SIMPLIFIED (no detection/NMAC radii, no start-end lines)
        for uav in self.other_uavs:
            pos = uav.current_position
            
            # UAV body - matching map_renderer colors
            body = Circle((pos.x, pos.y), uav.radius, 
                    fill=True, color='blue', alpha=0.7)
            ax.add_patch(body)
            
            # UAV heading indicator - CONSISTENT WITH map_renderer.py
            heading_length = uav.radius * 5
            dx = heading_length * np.cos(uav.current_heading)
            dy = heading_length * np.sin(uav.current_heading)
            arrow = FancyArrowPatch((pos.x, pos.y),
                                (pos.x + dx, pos.y + dy),
                                color='black',
                                arrowstyle='->',
                                mutation_scale=10,
                                linewidth=2.5)
            ax.add_patch(arrow)
            
            # Label UAV
            ax.text(pos.x + 25, pos.y + 25, uav.id, fontsize=12, fontweight='bold')
        
        # Draw test UAV (agent) - SIMPLIFIED (sensors replace detection radii)
        agent_pos = self.test_uav.current_position
        
        # Agent body - matching map_renderer exact color
        agent_body = Circle((agent_pos.x, agent_pos.y),
                        self.test_uav.radius,
                        fill=True, color='#0000A0', alpha=0.9)
        ax.add_patch(agent_body)
        
        # Agent heading indicator - CONSISTENT WITH map_renderer.py
        heading_length = self.test_uav.radius * 5
        dx = heading_length * np.cos(self.test_uav.current_heading)
        dy = heading_length * np.sin(self.test_uav.current_heading)
        agent_arrow = FancyArrowPatch((agent_pos.x, agent_pos.y),
                                (agent_pos.x + dx, agent_pos.y + dy),
                                color='black',
                                arrowstyle='->',
                                mutation_scale=10,
                                linewidth=2.5)
        ax.add_patch(agent_arrow)
        
        # Draw sensor visualization
        if sensor_type == 'camera':
            self._draw_camera_fov(ax)
        elif sensor_type == 'lidar':
            self._draw_lidar_scan(ax)
        
        # Add legend with consistent styling - BOTTOM LEFT
        legend_elements = [
            patches.Patch(color='#0000A0', alpha=0.9, label='Test UAV (Agent)'),
            patches.Patch(color='blue', alpha=0.7, label='Other UAVs'),
            patches.Patch(color='red', alpha=0.7, label='Restricted Areas'),
            patches.Patch(color='orange', alpha=0.3, label='Buffer Zones')
        ]
        if sensor_type == 'camera':
            legend_elements.append(patches.Patch(color='green', alpha=0.2, label='Camera FOV'))
            legend_elements.append(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='lime', 
                                                     linewidth=4, label='Detected (Camera)'))
        elif sensor_type == 'lidar':
            legend_elements.append(patches.Patch(color='red', alpha=0.6, label='LiDAR Scan'))
            legend_elements.append(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', 
                                                     linewidth=4, label='Detected (LiDAR)'))
        
        ax.legend(handles=legend_elements, loc='lower left', fontsize=12)
    
    def _draw_camera_fov(self, ax):
        """Draw camera field of view with detection highlights"""
        x, y = self.test_uav.current_position.x, self.test_uav.current_position.y
        heading = self.test_uav.current_heading
        fov = self.camera.fov
        range_cam = self.camera.range
        
        # Create FOV triangle
        angles = np.linspace(heading - fov/2, heading + fov/2, 20)
        fov_x = [x] + [x + range_cam * np.cos(a) for a in angles] + [x]
        fov_y = [y] + [y + range_cam * np.sin(a) for a in angles] + [y]
        
        ax.fill(fov_x, fov_y, color='green', alpha=0.15)
        
        # Draw FOV boundary lines with thicker lines
        ax.plot([x, x + range_cam * np.cos(heading - fov/2)],
                [y, y + range_cam * np.sin(heading - fov/2)], 'g-', linewidth=2.5, alpha=0.8)
        ax.plot([x, x + range_cam * np.cos(heading + fov/2)],
                [y, y + range_cam * np.sin(heading + fov/2)], 'g-', linewidth=2.5, alpha=0.8)
        
        # Get detections and highlight detected objects
        camera_uav_data, camera_ra_data = self.camera.get_data(self.test_uav)
        
        # Highlight detected UAVs with thick border
        for detection in camera_uav_data:
            for uav in self.other_uavs:
                if uav.id == detection['other_uav_id']:
                    highlight = Circle((uav.current_position.x, uav.current_position.y),
                                     uav.radius + 8, fill=False, color='lime', 
                                     linewidth=4, alpha=0.9)
                    ax.add_patch(highlight)
                    # Draw detection line
                    ax.plot([x, uav.current_position.x], [y, uav.current_position.y],
                           'lime', linewidth=2, alpha=0.6, linestyle=':')
        
        # Highlight detected RAs with thick border (only the specific detected instance)
        for ra_detection in camera_ra_data:
            # The detection contains the actual geometry that was detected
            detected_geometry = ra_detection.get('area')
            if detected_geometry and hasattr(detected_geometry, 'exterior'):
                # Draw thick lime border around this specific detected RA
                coords = list(detected_geometry.exterior.coords)
                xs, ys = zip(*coords)
                ax.plot(xs, ys, 'lime', linewidth=4, alpha=0.8, linestyle='-')
                
                # Draw detection line to RA centroid
                centroid = detected_geometry.centroid
                ax.plot([x, centroid.x], [y, centroid.y],
                       'lime', linewidth=2, alpha=0.5, linestyle=':')
        
        # Show detection count
        detection_text = f'Camera: {len(camera_uav_data)} UAVs, {len(camera_ra_data)} RAs'
        ax.text(0.02, 0.98, detection_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=12, fontweight='bold')
    
    def _draw_lidar_scan(self, ax):
        """Draw LiDAR scan rays and point cloud"""
        # Perform scan to get point cloud
        lidar_uav_data, lidar_ra_data = self.lidar.get_data(self.test_uav)
        point_cloud = self.lidar.get_point_cloud()
        
        x, y = self.test_uav.current_position.x, self.test_uav.current_position.y
        
        # Draw rays to hit points with varying alpha based on intensity
        for point in point_cloud:
            alpha = 0.3 + 0.4 * point['intensity']  # Scale alpha with intensity
            ax.plot([x, point['x']], [y, point['y']], 'r-', alpha=alpha, linewidth=0.8)
        
        # Draw point cloud with color-coded intensities
        if point_cloud and len(point_cloud) > 0:
            pc_x = [p['x'] for p in point_cloud]
            pc_y = [p['y'] for p in point_cloud]
            intensities = [p['intensity'] for p in point_cloud]
            
            # Only create scatter plot if we have points
            scatter = ax.scatter(pc_x, pc_y, c=intensities, cmap='hot', 
                               s=30, alpha=0.8, edgecolors='darkred', linewidths=0.5,
                               vmin=0, vmax=1)
            
            # Add colorbar - handle properly to avoid rendering errors
            try:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.1)
                cbar = plt.colorbar(scatter, cax=cax)
                cbar.set_label('Intensity', fontsize=9)
            except Exception as e:
                # Fallback if axes_grid1 doesn't work
                try:
                    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Intensity', fontsize=9)
                except:
                    pass  # Skip colorbar if it fails
        
        # Highlight detected UAVs
        for detection in lidar_uav_data:
            for uav in self.other_uavs:
                if uav.id == detection['other_uav_id']:
                    highlight = Circle((uav.current_position.x, uav.current_position.y),
                                     uav.radius + 8, fill=False, color='red', 
                                     linewidth=4, alpha=0.9)
                    ax.add_patch(highlight)
                    # Draw detection line
                    ax.plot([x, uav.current_position.x], [y, uav.current_position.y],
                           'red', linewidth=2, alpha=0.6, linestyle=':')
        
        # For LiDAR RA detections, just draw a detection line (no border outline needed)
        for ra_detection in lidar_ra_data:
            centroid = ra_detection.get('centroid')
            if centroid:
                # Draw detection line to RA centroid only
                ax.plot([x, centroid[0]], [y, centroid[1]],
                       'red', linewidth=2, alpha=0.4, linestyle=':')
        
        # Show detection count and point cloud stats
        detection_text = f'LiDAR: {len(lidar_uav_data)} UAVs, {len(lidar_ra_data)} RAs\nPoints: {len(point_cloud)}'
        ax.text(0.02, 0.98, detection_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=12, fontweight='bold')
    
    def print_detection_results(self):
        """Print detailed detection results"""
        print("\n" + "="*60)
        print("SENSOR DETECTION RESULTS")
        print("="*60)
        
        camera_uav_data, camera_ra_data = self.camera.get_data(self.test_uav)
        lidar_uav_data, lidar_ra_data = self.lidar.get_data(self.test_uav)
        
        print(f"\nTest UAV Position: ({self.test_uav.current_position.x:.1f}, "
              f"{self.test_uav.current_position.y:.1f})")
        print(f"Test UAV Heading: {math.degrees(self.test_uav.current_heading):.1f}°")
        
        print("\n--- CAMERA SENSOR ---")
        print(f"UAV Detections: {len(camera_uav_data)}")
        for detection in camera_uav_data:
            print(f"  • UAV {detection['other_uav_id']}: "
                  f"dist={detection['detection_distance']:.1f}m, "
                  f"bearing={math.degrees(detection['detection_bearing']):.1f}°, "
                  f"confidence={detection['detection_confidence']:.2f}")
        
        print(f"RA Detections: {len(camera_ra_data)}")
        for detection in camera_ra_data:
            print(f"  • {detection['type']}: "
                  f"dist={detection['distance']:.1f}m, "
                  f"confidence={detection['detection_confidence']:.2f}")
        
        print("\n--- LIDAR SENSOR ---")
        print(f"UAV Detections: {len(lidar_uav_data)}")
        for detection in lidar_uav_data:
            print(f"  • UAV {detection['other_uav_id']}: "
                  f"dist={detection['detection_distance']:.1f}m, "
                  f"bearing={math.degrees(detection['detection_bearing']):.1f}°, "
                  f"confidence={detection['detection_confidence']:.2f}, "
                  f"points={detection['point_count']}")
        
        print(f"RA Detections: {len(lidar_ra_data)}")
        for detection in lidar_ra_data:
            print(f"  • {detection['type']}: "
                  f"dist={detection['distance']:.1f}m, "
                  f"confidence={detection['detection_confidence']:.2f}, "
                  f"points={detection['point_count']}")
        
        point_cloud = self.lidar.get_point_cloud()
        print(f"\nLiDAR Point Cloud: {len(point_cloud)} points")
        
        print("-"*60)
    
    def test_sensor_movement(self, positions: List[tuple], pause_time: float = 0.5):
        """Test sensors as UAV moves through different positions"""
        print("Starting sensor movement test...")
        print("="*60)
        
        for i, (x, y, heading) in enumerate(positions):
            print(f"\n▶ Position {i+1}/{len(positions)}: ({x}, {y}) heading {math.degrees(heading):.1f}°")
            
            # Move test UAV
            self.test_uav.move_to(x, y, heading)
            
            # Update visualizations
            if self.fig is not None and self.axes is not None:
                try:
                    self.visualize_environment(self.axes[0], f"Camera View - Position {i+1}", 'camera')
                    self.visualize_environment(self.axes[1], f"LiDAR Scan - Position {i+1}", 'lidar')
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    
                    if self.save_plots:
                        filename = f'sensor_test_position_{i+1}.png'
                        self.fig.savefig(filename, dpi=150, bbox_inches='tight', 
                                       facecolor='#E5E5E5', edgecolor='none')
                        print(f"  ✓ Saved: {filename}")
                    
                    if self.enable_plotting:
                        plt.draw()
                        plt.pause(pause_time)
                        
                except Exception as e:
                    print(f"  ✗ Plotting failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Print detection results
            self.print_detection_results()
            
            time.sleep(pause_time)
        
        print("\n" + "="*60)
        print("Movement test completed")
        print("="*60)
    
    def test_collision_detection(self):
        """Test NMAC and collision detection"""
        print("\n" + "="*60)
        print("COLLISION DETECTION TEST")
        print("="*60)
        
        target_uav = self.other_uavs[0]
        
        print("\n--- NMAC Detection ---")
        self.test_uav.move_to(target_uav.current_position.x + 150,
                             target_uav.current_position.y, 0.0)
        
        nmac_detected, nmac_list = self.camera.get_nmac(self.test_uav)
        print(f"Camera NMAC: {nmac_detected}, UAVs: {[uav.id for uav in nmac_list]}")
        
        nmac_detected, nmac_list = self.lidar.get_nmac(self.test_uav)
        print(f"LiDAR NMAC: {nmac_detected}, UAVs: {[uav.id for uav in nmac_list]}")
        
        print("\n--- Collision Detection ---")
        self.test_uav.move_to(target_uav.current_position.x + 10,
                             target_uav.current_position.y, 0.0)
        
        collision_detected, collision_info = self.camera.get_uav_collision(self.test_uav)
        print(f"Camera Collision: {collision_detected}, Info: {collision_info}")
        
        collision_detected, collision_info = self.lidar.get_uav_collision(self.test_uav)
        print(f"LiDAR Collision: {collision_detected}, Info: {collision_info}")


class FusionTester(SensorTester):
    """Extended tester for MultiSensorFusion with consistent rendering"""
    
    def __init__(self, enable_plotting=False, save_plots=False):
        # Call parent init but skip figure creation
        self.atc = MockATC()
        self.airspace = MockAirspace()
        self.enable_plotting = enable_plotting
        self.save_plots = save_plots
        
        # Set backend
        if enable_plotting:
            try:
                matplotlib.use('TkAgg')
                plt.ion()
                print("✓ Interactive plotting enabled")
            except:
                matplotlib.use('Agg')
                self.enable_plotting = False
                self.save_plots = True
        elif save_plots:
            matplotlib.use('Agg')
            print("✓ Save-plots mode enabled")
        
        # Create UAVs - KEEP USER'S MODIFICATIONS
        self.test_uav = MockUAV(100, 300, "test_uav", heading=0.0)
        self.other_uavs = [
            MockUAV(400, 350, "uav_1", heading=math.pi/2),
            MockUAV(200, 500, "uav_2", heading=math.pi),
            MockUAV(600, 300, "uav_3", heading=-math.pi/4),
        ]
        
        self.atc.add_uav(self.test_uav)
        for uav in self.other_uavs:
            self.atc.add_uav(uav)
        
        # Initialize sensors
        self.camera = Camera2DSensor(self.airspace, self.atc, fov=math.pi/3, range=300.0)
        self.lidar = LiDAR2DSensor(self.airspace, self.atc, range=400.0, num_rays=32)
        self.fusion_sensor = MultiSensorFusion(self.camera, self.lidar)
        
        # Create 3-panel figure for fusion with grey background
        self.fig = None
        self.axes = None
        if self.enable_plotting or self.save_plots:
            try:
                self.fig, self.axes = plt.subplots(1, 3, figsize=(24, 8))
                # self.fig.suptitle('UAM Sensor Fusion Testing', fontsize=16, fontweight='bold')
                print("✓ Fusion figure initialized")
            except Exception as e:
                print(f"✗ Figure init failed: {e}")
                self.enable_plotting = False
                self.save_plots = False
    
    def visualize_fusion_environment(self, position_idx: int):
        """Visualize all three sensor views"""
        if self.fig is None or self.axes is None:
            return
            
        try:
            # Camera view
            self.visualize_environment(self.axes[0], f"Camera", 'camera') #  - Pos {position_idx}
            
            # LiDAR view
            self.visualize_environment(self.axes[1], f"LiDAR", 'lidar') #  - Pos {position_idx}
            
            # Fusion combined view
            self._visualize_fusion_combined(self.axes[2], f"Fusion") #  - Pos {position_idx}
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if self.save_plots:
                filename = f'fusion_test_position_{position_idx}.png'
                self.fig.savefig(filename, dpi=600, bbox_inches='tight')
                print(f"  ✓ Saved: {filename}")
                
            if self.enable_plotting:
                plt.draw()
                plt.pause(0.5)
                
        except Exception as e:
            print(f"  ✗ Fusion plotting failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_fusion_combined(self, ax, title: str):
        """Combined visualization showing fusion results"""
        # Draw base environment (same as parent)
        self.visualize_environment(ax, title, sensor_type=None)
        
        # Get fusion data
        fused_uav, fused_ra = self.fusion_sensor.get_data(self.test_uav)
        
        x, y = self.test_uav.current_position.x, self.test_uav.current_position.y
        
        # Draw both FOV and scan lightly
        self._draw_camera_fov_light(ax)
        self._draw_lidar_scan_light(ax)
        
        # Highlight fusion detections with special markers
        fusion_legend_items = {
            'fused': {'count': 0, 'color': 'purple', 'label': 'FUSED'},
            'camera_only': {'count': 0, 'color': 'lime', 'label': 'CAM'},
            'lidar_only': {'count': 0, 'color': 'red', 'label': 'LDR'}
        }
        
        # Highlight UAV detections
        for det in fused_uav:
            for uav in self.other_uavs:
                if uav.id == det['other_uav_id']:
                    sensor_type = det.get('sensor_type', 'unknown')
                    
                    if sensor_type in fusion_legend_items:
                        color = fusion_legend_items[sensor_type]['color']
                        label = fusion_legend_items[sensor_type]['label']
                        fusion_legend_items[sensor_type]['count'] += 1
                    else:
                        color = 'gray'
                        label = '?'
                    
                    highlight = Circle((uav.current_position.x, uav.current_position.y),
                                     uav.radius + 10, fill=False, color=color, 
                                     linewidth=5, alpha=0.9)
                    ax.add_patch(highlight)
                    
                    # # Add label above UAV
                    # ax.text(uav.current_position.x, uav.current_position.y - 40,
                    #        f"{label}\n{uav.id}", color=color, fontweight='bold', fontsize=12,
                    #        ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Highlight RA detections using the actual detected geometry
        for det in fused_ra:
            sensor_type = det.get('sensor_type', 'unknown')
            
            if sensor_type in fusion_legend_items:
                color = fusion_legend_items[sensor_type]['color']
                label = fusion_legend_items[sensor_type]['label']
            else:
                color = 'gray'
                label = '?'
            
            # Use the geometry from the detection (camera detections have 'area', lidar has centroid)
            detected_geometry = det.get('area')
            if detected_geometry and hasattr(detected_geometry, 'exterior'):
                # Draw colored border around detected RA
                coords = list(detected_geometry.exterior.coords)
                xs, ys = zip(*coords)
                ax.plot(xs, ys, color, linewidth=5, alpha=0.9, linestyle='-')
                
                # # Add label on RA (centered like FUSED label)
                # centroid = detected_geometry.centroid
                # ax.text(centroid.x, centroid.y, f"{label}\n{det['type']}", 
                #        color=color, fontweight='bold', fontsize=12,
                #        ha='center', va='center',
                #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            elif 'centroid' in det:
                # For lidar-only detections without full geometry, position label at centroid
                centroid = det['centroid']
                # ax.plot(centroid[0], centroid[1], 'o', color=color, markersize=10, alpha=0.8)
                # # Position label at centroid (middle of where RA would be)
                # ax.text(centroid[0], centroid[1], f"{label}\n{det['type']}", 
                #        color=color, fontweight='bold', fontsize=12,
                #        ha='center', va='center',
                #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Show fusion statistics with consistent styling
        stats = self.fusion_sensor.get_fusion_statistics(self.test_uav)
        fusion_text = (f"Fusion: {stats['fused_uav_detections']} UAVs, {stats['fused_ra_detections']} RAs\n"
                      f"Types: CAM: {stats['uav_fusion_types'].get('camera_only', 0)}, "
                      f"LDR: {stats['uav_fusion_types'].get('lidar_only', 0)}")
        
        ax.text(0.02, 0.98, fusion_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=12, fontweight='bold')
        
        # Add custom legend for fusion types - BOTTOM LEFT
        fusion_legend = [
            Circle((0, 0), 1, fill=False, color='purple', linewidth=5, label='Both Sensors (Fused)'),
            Circle((0, 0), 1, fill=False, color='lime', linewidth=5, label='Camera Only'),
            Circle((0, 0), 1, fill=False, color='red', linewidth=5, label='LiDAR Only'),
        ]
        ax.legend(handles=fusion_legend, loc='lower left', fontsize=12, title='Fusion Detection Types')
    
    def _draw_camera_fov_light(self, ax):
        """Light camera FOV for fusion view"""
        x, y = self.test_uav.current_position.x, self.test_uav.current_position.y
        heading = self.test_uav.current_heading
        fov, range_cam = self.camera.fov, self.camera.range
        
        angles = np.linspace(heading - fov/2, heading + fov/2, 15)
        fov_x = [x] + [x + range_cam * np.cos(a) for a in angles] + [x]
        fov_y = [y] + [y + range_cam * np.sin(a) for a in angles] + [y]
        ax.fill(fov_x, fov_y, color='green', alpha=0.08)
    
    def _draw_lidar_scan_light(self, ax):
        """Light LiDAR scan for fusion view"""
        point_cloud = self.lidar.get_point_cloud()
        x, y = self.test_uav.current_position.x, self.test_uav.current_position.y
        
        for point in point_cloud[::3]:  # Every 3rd point
            ax.plot([x, point['x']], [y, point['y']], 'r-', alpha=0.15, linewidth=0.5)
    
    def test_fusion_detection(self):
        """Test fusion properly combines data"""
        print("\n" + "="*60)
        print("FUSION VALIDATION")
        print("="*60)
        
        camera_uav, camera_ra = self.camera.get_data(self.test_uav)
        lidar_uav, lidar_ra = self.lidar.get_data(self.test_uav)
        fused_uav, fused_ra = self.fusion_sensor.get_data(self.test_uav)
        
        print(f"Camera: {len(camera_uav)} UAVs, {len(camera_ra)} RAs")
        print(f"LiDAR:  {len(lidar_uav)} UAVs, {len(lidar_ra)} RAs")
        print(f"Fusion: {len(fused_uav)} UAVs, {len(fused_ra)} RAs")
        
        # Validate no duplicates
        fused_ids = [d['other_uav_id'] for d in fused_uav]
        assert len(fused_ids) == len(set(fused_ids)), "Duplicate UAV IDs!"
        print("✓ No duplicate IDs")
        
        # Validate completeness
        camera_ids = set(d['other_uav_id'] for d in camera_uav)
        lidar_ids = set(d['other_uav_id'] for d in lidar_uav)
        expected = camera_ids | lidar_ids
        actual = set(fused_ids)
        
        assert actual == expected, f"Missing IDs: expected {expected}, got {actual}"
        print("✓ All unique UAVs included")
        
        print("\nFusion test PASSED")
        return True


def main():
    """Main test function"""
    import sys
    
    enable_plotting = '--plot' in sys.argv
    save_plots = '--save-plots' in sys.argv or (not enable_plotting)
    test_fusion = '--fusion' in sys.argv
    
    print("\n" + "="*70)
    print("SENSOR VISUALIZATION TEST (FINAL VERSION)")
    print("="*70)
    print(f"Mode: {'Interactive' if enable_plotting else 'Save-only'}")
    print(f"Fusion: {'Yes' if test_fusion else 'No'}")
    print("="*70 + "\n")
    
    if test_fusion:
        tester = FusionTester(enable_plotting=enable_plotting, save_plots=save_plots)
        
        tester.test_fusion_detection()
        
        # KEEP USER'S POSITION 6 MODIFICATIONS
        test_positions = [
            (100, 300, 0.0),           # Position 1
            (320, 320, math.pi/4),     # Position 2
            (360, 280, math.pi/2),     # Position 3
            (180, 480, 0.0),           # Position 4
            (450, 400, math.pi/2),     # Position 5
            (475, 350, 11*math.pi/6),  # Position 6: USER'S FUSION VALIDATION
        ]
        
        position_descriptions = [
            "General test position 1",
            "General test position 2",
            "General test position 3",
            "General test position 4",
            "General test position 5",
            "FUSION VALIDATION: Camera-only, LiDAR-only, and Both detections"
        ]
        
        for i, (x, y, heading) in enumerate(test_positions):
            print(f"\n▶ Fusion Position {i+1}: ({x}, {y}) - {position_descriptions[i]}")
            tester.test_uav.move_to(x, y, heading)
            tester.visualize_fusion_environment(i+1)
            
            stats = tester.fusion_sensor.get_fusion_statistics(tester.test_uav)
            print(f"  Fusion: {stats['fused_uav_detections']} UAVs, {stats['fused_ra_detections']} RAs")
            print(f"  Types: {stats['uav_fusion_types']}")
            
            # Special validation for position 6
            if i == 5:  # Position 6 (index 5)
                print("\n  === DETAILED FUSION VALIDATION ===")
                camera_uav, _ = tester.camera.get_data(tester.test_uav)
                lidar_uav, _ = tester.lidar.get_data(tester.test_uav)
                fused_uav, _ = tester.fusion_sensor.get_data(tester.test_uav)
                
                camera_ids = set(d['other_uav_id'] for d in camera_uav)
                lidar_ids = set(d['other_uav_id'] for d in lidar_uav)
                
                camera_only = camera_ids - lidar_ids
                lidar_only = lidar_ids - camera_ids
                both_detect = camera_ids & lidar_ids
                
                print(f"  Camera-only detections: {camera_only if camera_only else 'None'}")
                print(f"  LiDAR-only detections:  {lidar_only if lidar_only else 'None'}")
                print(f"  Both sensors detect:    {both_detect if both_detect else 'None'}")
                
                # Verify fusion types match expectations
                for det in fused_uav:
                    uav_id = det['other_uav_id']
                    sensor_type = det.get('sensor_type', 'unknown')
                    
                    if uav_id in camera_only:
                        assert sensor_type == 'camera_only', f"Expected camera_only for {uav_id}, got {sensor_type}"
                        print(f"  ✓ {uav_id}: Correctly marked as camera_only")
                    elif uav_id in lidar_only:
                        assert sensor_type == 'lidar_only', f"Expected lidar_only for {uav_id}, got {sensor_type}"
                        print(f"  ✓ {uav_id}: Correctly marked as lidar_only")
                    elif uav_id in both_detect:
                        assert sensor_type == 'fused', f"Expected fused for {uav_id}, got {sensor_type}"
                        print(f"  ✓ {uav_id}: Correctly marked as fused")
                
                # Check that we actually have all three cases
                has_camera_only = any(d.get('sensor_type') == 'camera_only' for d in fused_uav)
                has_lidar_only = any(d.get('sensor_type') == 'lidar_only' for d in fused_uav)
                has_fused = any(d.get('sensor_type') == 'fused' for d in fused_uav)
                
                if has_camera_only and has_lidar_only and has_fused:
                    print("\n  ✓✓✓ FUSION FULLY VALIDATED - All three detection types present!")
                else:
                    print(f"\n  ⚠ Partial validation - camera_only:{has_camera_only}, lidar_only:{has_lidar_only}, fused:{has_fused}")
                
                print("  ==================================")
            
            time.sleep(0.3)
    else:
        tester = SensorTester(enable_plotting=enable_plotting, save_plots=save_plots)
        
        test_positions = [
            (100, 300, 0.0),
            (250, 350, math.pi/4),
            (450, 400, math.pi/2),
            (600, 300, math.pi),
        ]
        
        pause_time = 1.0 if enable_plotting else 0.3
        tester.test_sensor_movement(test_positions, pause_time=pause_time)
        tester.test_collision_detection()
    
    print("\n" + "="*70)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*70)
    
    if enable_plotting:
        print("\nClose plot window to exit...")
        plt.show()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()