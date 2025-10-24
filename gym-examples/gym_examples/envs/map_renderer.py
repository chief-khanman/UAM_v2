import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import math
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from uav_v2 import UAV_v2
from auto_uav_v2 import Auto_UAV_v2
#from map_env_revised import MapEnv

class MapRenderer:
    """A dedicated class for handling all rendering functionality in MapEnv."""
    
    def __init__(self, env, render_mode=None, sleep_time=0):
        """
        Initialize the renderer with a reference to the environment.
        
        Args:
            env: The MapEnv instance this renderer is associated with
            render_mode: The rendering mode ('human', 'rgb_array', etc.)
            sleep_time: Time to pause between renders in interactive mode
        """
        self.env = env
        self.render_mode = render_mode
        self.sleep_time = sleep_time
        
        # Rendering attributes
        self.fig = None
        self.ax = None
        self.trajectory_by_id = {}
        
        # Animation data
        self.df = pd.DataFrame({
            "current_time_step": [],
            "uav_id": [],
            "uav": [],
            "current_position": [],
            "current_heading": [],
            "final_heading": [],
        })

    def render(self):
        """
        Render the environment with matplotlib.
        This method is used to render the current state of the env. 
        This method is used for online rendering of the env.
        """
        if self.render_mode is None:
            return
        
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            plt.ion()  # Interactive mode on
            
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Draw the airspace and restricted areas
        self.render_static_assets(self.ax)
        
        # Draw the sound heat map
        sound_map = np.random.rand(464,361)
        extent = (602201.377178908, 638378.8759272858, 3330295.7490083342, 3376699.839849744)
        im = self.ax.imshow(
                        sound_map,
                        extent=extent,
                        origin='lower',
                        alpha=0.3,     # transparency so the map is visible
                        interpolation='nearest'  # or 'bilinear' if you prefer smoothing
                    )
        
        # Store current positions for trajectory history
        current_positions = []
        current_uav_ids = []
        
        # Draw vertiports with larger markers
        for vertiport in self.env.airspace.get_vertiport_list():
            self.ax.plot(vertiport.x, vertiport.y, 'gs', markersize=12)
        
        # Draw learning agent (with dark blue color like in uam_uav.py)
        agent_pos = self.env.agent.current_position
        current_positions.append((agent_pos.x, agent_pos.y))
        current_uav_ids.append(self.env.agent.id)
        
        # Agent detection radius
        agent_detection = Circle((agent_pos.x, agent_pos.y),
                            self.env.agent.detection_radius,
                            fill=False, color='#0278c2', alpha=0.3, linewidth=2)
        self.ax.add_patch(agent_detection)
        
        # Agent NMAC radius
        agent_nmac = Circle((agent_pos.x, agent_pos.y),
                        self.env.agent.nmac_radius,
                        fill=False, color='#FF7F50', alpha=0.4, linewidth=2)
        self.ax.add_patch(agent_nmac)
        
        # Agent body
        agent_body = Circle((agent_pos.x, agent_pos.y),
                        self.env.agent.radius,
                        fill=True, color='#0000A0', alpha=0.9)
        self.ax.add_patch(agent_body)
        
        # Agent heading indicator with thicker line
        heading_length = self.env.agent.radius * 5  # Make longer for visibility
        dx = heading_length * np.cos(self.env.agent.current_heading)
        dy = heading_length * np.sin(self.env.agent.current_heading)
        agent_arrow = FancyArrowPatch((agent_pos.x, agent_pos.y),
                                (agent_pos.x + dx, agent_pos.y + dy),
                                color='black',
                                arrowstyle='->',
                                mutation_scale=10,
                                linewidth=2.5)
        self.ax.add_patch(agent_arrow)
        
        # Agent start-end connection with thicker line
        self.ax.plot([self.env.agent.start.x, self.env.agent.end.x],
                    [self.env.agent.start.y, self.env.agent.end.y],
                    'b--', alpha=0.6, linewidth=2.0)
        
        # Draw non-learning UAVs
        for uav in self.env.atc.get_uav_list():
            if isinstance(uav, Auto_UAV_v2):
                continue
                
            pos = uav.current_position
            current_positions.append((pos.x, pos.y))
            current_uav_ids.append(uav.id)
            
            # UAV detection radius
            detection = Circle((pos.x, pos.y), uav.detection_radius, 
                        fill=False, color='green', alpha=0.3, linewidth=2)
            self.ax.add_patch(detection)
            
            # UAV NMAC radius
            nmac = Circle((pos.x, pos.y), uav.nmac_radius, 
                    fill=False, color='orange', alpha=0.4, linewidth=2)
            self.ax.add_patch(nmac)
            
            # UAV body
            body = Circle((pos.x, pos.y), uav.radius, 
                    fill=True, color='blue', alpha=0.7)
            self.ax.add_patch(body)
            
            # UAV heading indicator with thicker line
            heading_length = uav.radius * 5  # Make longer for visibility
            dx = heading_length * np.cos(uav.current_heading)
            dy = heading_length * np.sin(uav.current_heading)
            arrow = FancyArrowPatch((pos.x, pos.y),
                                (pos.x + dx, pos.y + dy),
                                color='black',
                                arrowstyle='->',
                                mutation_scale=10,
                                linewidth=2.5)
            self.ax.add_patch(arrow)
            
            # UAV start-end connection with thicker line
            self.ax.plot([uav.start.x, uav.end.x],
                        [uav.start.y, uav.end.y],
                        'g--', alpha=0.6, linewidth=2.0)
        
        # Update trajectory history - maintaining correct UAV ID mapping
        # Create a dictionary of ID to position if it doesn't exist
        if not hasattr(self, 'trajectory_by_id'):
            self.trajectory_by_id = {}
            
        # Add current positions to trajectories
        for uav_id, pos in zip(current_uav_ids, current_positions):
            if uav_id not in self.trajectory_by_id:
                self.trajectory_by_id[uav_id] = []
            self.trajectory_by_id[uav_id].append(pos)
        
        # Draw trajectory lines - only for UAVs that still exist
        for uav_id in current_uav_ids:
            if uav_id in self.trajectory_by_id and len(self.trajectory_by_id[uav_id]) > 1:
                xs, ys = zip(*self.trajectory_by_id[uav_id])
                # Use thicker lines for trajectories
                line_color = '#0000A0' if uav_id == self.env.agent.id else 'blue'
                self.ax.plot(xs, ys, '-', linewidth=2.5, alpha=0.6, color=line_color)
        
        # Calculate proper plot limits to see the whole map
        # First get vertiport boundaries
        vp_x_coords = [v.x for v in self.env.airspace.get_vertiport_list()]
        vp_y_coords = [v.y for v in self.env.airspace.get_vertiport_list()]
        
        # Then get UAV positions
        uav_x_coords = [pos[0] for pos in current_positions]
        uav_y_coords = [pos[1] for pos in current_positions]
        
        # Combine to get full area
        all_x_coords = vp_x_coords + uav_x_coords
        all_y_coords = vp_y_coords + uav_y_coords
        
        # Add restricted areas dimensions
        if hasattr(self.env.airspace, 'location_tags'):
            for tag_value in self.env.airspace.location_tags.keys():
                # Get bounds of restricted areas
                restricted_bounds = self.env.airspace.location_utm[tag_value].bounds
                if len(restricted_bounds) > 0:
                    for bound in restricted_bounds.values:
                        if len(bound) >= 4:  # minx, miny, maxx, maxy
                            all_x_coords.extend([bound[0], bound[2]])
                            all_y_coords.extend([bound[1], bound[3]])
        
        #! limited map rendering has been deactivated - for clearer sound heat map visualization 
        # Set limits with margin
        # if all_x_coords and all_y_coords:
        #     x_min, x_max = min(all_x_coords), max(all_x_coords)
        #     y_min, y_max = min(all_y_coords), max(all_y_coords)
            
        #     # Add margin to ensure all elements are visible
        #     margin = max(500, (x_max - x_min) * 0.1)
        #     self.ax.set_xlim(x_min - margin, x_max + margin)
        #     self.ax.set_ylim(y_min - margin, y_max + margin)
        
        self.ax.set_title(f'UAM Simulation - {self.env.location_name} - Step {self.env.current_time_step}')
        self.ax.set_aspect('equal')
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        plt.draw()
        plt.pause(self.sleep_time) #! why use this instead of time.sleep()

    def create_animation(self, env_time_step):
        """Create an animation of the environment."""
        # Ensure data exists
        if len(self.df) == 0:
            print("No animation data available")
            return None
        
        try:
            from matplotlib.animation import FuncAnimation
            
            # Pre-process data for animation to avoid df lookups during animation
            self.animation_data = {}
            self.animation_time_steps = sorted(self.df['current_time_step'].unique())
            
            # Group data by time step
            for time_step in self.animation_time_steps:
                step_data = self.df[self.df['current_time_step'] == time_step]
                self.animation_data[time_step] = []
                
                # Extract and store UAV data for this time step
                for _, row in step_data.iterrows():
                    uav_data = {
                        'id': row['uav_id'],
                        'position': row['current_position'],
                        'heading': row['current_heading'],
                        'final_heading': row['final_heading'],
                        'is_auto': isinstance(row['uav'], Auto_UAV_v2),
                        'uav': row['uav']
                    }
                    self.animation_data[time_step].append(uav_data)
            
            # Create trajectories by UAV ID
            self.animation_trajectories = {}
            for time_step in self.animation_time_steps:
                for uav_data in self.animation_data[time_step]:
                    uav_id = uav_data['id']
                    position = uav_data['position']
                    
                    if uav_id not in self.animation_trajectories:
                        self.animation_trajectories[uav_id] = []
                    
                    # Store time step and position
                    self.animation_trajectories[uav_id].append((time_step, position))
                    
            # Create a new figure specifically for animation
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Function to update animation frame
            def animate(frame_index):
                if frame_index >= len(self.animation_time_steps):
                    return []
                    
                time_step = self.animation_time_steps[frame_index]
                ax.clear()
                
                # Draw static assets
                self.render_static_assets(ax)
                
                # Draw vertiports with larger markers
                for vertiport in self.env.airspace.get_vertiport_list():
                    ax.plot(vertiport.x, vertiport.y, 'gs', markersize=12)
                
                # Draw UAVs at this time step
                current_uav_ids = []
                for uav_data in self.animation_data[time_step]:
                    pos = uav_data['position']
                    heading = uav_data['heading']
                    final_heading = uav_data['final_heading']
                    is_auto = uav_data['is_auto']
                    uav_id = uav_data['id']
                    uav_obj = uav_data['uav']
                    
                    current_uav_ids.append(uav_id)
                    
                    # Set colors based on UAV type
                    detection_color = '#0278c2' if is_auto else 'green'
                    nmac_color = '#FF7F50' if is_auto else 'orange'
                    body_color = '#0000A0' if is_auto else 'blue'
                    
                    # Draw detection radius
                    detection_circle = Circle((pos.x, pos.y),
                                        uav_obj.detection_radius,
                                        fill=False, color=detection_color, alpha=0.3, linewidth=2)
                    ax.add_patch(detection_circle)
                    
                    # Draw NMAC radius
                    nmac_circle = Circle((pos.x, pos.y),
                                    uav_obj.nmac_radius,
                                    fill=False, color=nmac_color, alpha=0.4, linewidth=2)
                    ax.add_patch(nmac_circle)
                    
                    # Draw UAV body
                    body_circle = Circle((pos.x, pos.y),
                                    uav_obj.radius,
                                    fill=True, color=body_color, alpha=0.7)
                    ax.add_patch(body_circle)
                    
                    # Draw current heading arrow
                    heading_length = uav_obj.radius * 5  # Make longer for visibility
                    dx = heading_length * np.cos(heading)
                    dy = heading_length * np.sin(heading)
                    arrow = FancyArrowPatch((pos.x, pos.y),
                                    (pos.x + dx, pos.y + dy),
                                    color='black',
                                    arrowstyle='->',
                                    mutation_scale=10,
                                    linewidth=2.5)
                    ax.add_patch(arrow)
                    
                    # Draw final heading (reference direction) with thicker line
                    ref_length = uav_obj.radius * 4
                    dx_ref = ref_length * np.cos(final_heading)
                    dy_ref = ref_length * np.sin(final_heading)
                    ref_color = 'purple' if is_auto else 'red'
                    ref_arrow = FancyArrowPatch((pos.x, pos.y),
                                        (pos.x + dx_ref, pos.y + dy_ref),
                                        color=ref_color,
                                        arrowstyle='->',
                                        mutation_scale=7.5,
                                        linewidth=2,
                                        alpha=0.6)
                    ax.add_patch(ref_arrow)
                    
                    # Draw start-end connection with thicker line
                    if hasattr(uav_obj, 'start') and hasattr(uav_obj, 'end'):
                        line_color = 'blue' if is_auto else 'green'
                        ax.plot([uav_obj.start.x, uav_obj.end.x],
                            [uav_obj.start.y, uav_obj.end.y],
                            '--', color=line_color, alpha=0.6, linewidth=2.0)
                
                # Draw trajectories up to this time step
                # This ensures we don't show trajectories for removed UAVs
                for uav_id in current_uav_ids:
                    if uav_id in self.animation_trajectories:
                        # Get trajectory points up to current time step
                        traj_points = [(t, p) for t, p in self.animation_trajectories[uav_id] if t <= time_step]
                        
                        if len(traj_points) > 1:
                            # Extract positions
                            positions = [p for _, p in traj_points]
                            xs = [p.x for p in positions]
                            ys = [p.y for p in positions]
                            
                            # Determine color based on UAV type
                            is_auto = any(d['is_auto'] for d in self.animation_data[time_step] if d['id'] == uav_id)
                            line_color = '#0000A0' if is_auto else 'blue'
                            
                            # Draw trajectory line with thicker width
                            ax.plot(xs, ys, '-', linewidth=2.5, alpha=0.6, color=line_color)
                
                # Calculate proper plot limits to see the whole map
                vp_x_coords = [v.x for v in self.env.airspace.get_vertiport_list()]
                vp_y_coords = [v.y for v in self.env.airspace.get_vertiport_list()]
                
                # Add UAV positions
                uav_x_coords = [uav_data['position'].x for uav_data in self.animation_data[time_step]]
                uav_y_coords = [uav_data['position'].y for uav_data in self.animation_data[time_step]]
                
                # Combine for full area
                all_x_coords = vp_x_coords + uav_x_coords
                all_y_coords = vp_y_coords + uav_y_coords
                
                # Add restricted areas dimensions
                if hasattr(self.env.airspace, 'location_tags'):
                    for tag_value in self.env.airspace.location_tags.keys():
                        # Get bounds of restricted areas
                        restricted_bounds = self.env.airspace.location_utm[tag_value].bounds
                        if len(restricted_bounds) > 0:
                            for bound in restricted_bounds.values:
                                if len(bound) >= 4:  # minx, miny, maxx, maxy
                                    all_x_coords.extend([bound[0], bound[2]])
                                    all_y_coords.extend([bound[1], bound[3]])
                
                # Set limits with margin
                if all_x_coords and all_y_coords:
                    x_min, x_max = min(all_x_coords), max(all_x_coords)
                    y_min, y_max = min(all_y_coords), max(all_y_coords)
                    
                    # Add margin to ensure all elements are visible
                    margin = max(500, (x_max - x_min) * 0.1)
                    ax.set_xlim(x_min - margin, x_max + margin)
                    ax.set_ylim(y_min - margin, y_max + margin)
                
                ax.set_title(f'UAM Simulation - Step {time_step}')
                ax.set_aspect('equal')
                
                return []
            
            # Create animation object
            frames = min(env_time_step, len(self.animation_time_steps))
            if frames == 0:
                print("No frames to animate")
                return None
                
            ani = FuncAnimation(
                fig, 
                animate, 
                frames=frames,
                interval=200,
                blit=False
            )
            
            return ani
            
        except Exception as e:
            print(f"Error creating animation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_animation(self, animation_obj, file_name, mp4_only=True):
        """Save animation to a file with optimized quality and compatibility.
    
        Args:
            animation_obj: The animation object to save
            file_name: Base file name for the animation (without extension)
            mp4_only: If True, only save MP4 and skip GIF creation
        """
        if animation_obj is None:
            print("No animation to save")
            return
            
        try:
            # Save as MP4 first with optimized settings
            try:
                print(f"Saving animation to {file_name}.mp4...")
                from matplotlib.animation import FFMpegWriter
                
                # Try to use a writer with optimized settings
                writer = FFMpegWriter(
                    fps=10,  # Higher fps for smoother playback
                    metadata=dict(title='UAM Simulation'),
                    bitrate=5000,  # Higher bitrate for better quality
                    extra_args=['-vcodec', 'mpeg4', 
                            '-pix_fmt', 'yuv420p',
                            '-q:v', '3']  # Quality value - lower is better quality (1-31)
                )
                
                # First attempt with quality settings
                try:
                    animation_obj.save(
                        f"{file_name}.mp4",
                        writer=writer,
                        dpi=200  # Higher DPI for better quality
                    )
                    print("MP4 saved successfully with high quality settings!")
                except Exception as e:
                    print(f"High quality MP4 save failed: {e}")
                    
                    # Fallback to simpler settings
                    try:
                        print("Trying with simpler MP4 settings...")
                        animation_obj.save(
                            f"{file_name}.mp4",
                            writer='ffmpeg',
                            fps=8,
                            dpi=150
                        )
                        print("MP4 saved successfully with basic settings!")
                    except Exception as e:
                        print(f"Basic MP4 save failed: {e}")
                        
                        # Try with minimal settings
                        try:
                            print("Trying with minimal MP4 settings...")
                            animation_obj.save(
                                f"{file_name}.mp4",
                                writer='ffmpeg',
                                fps=5,
                                dpi=100
                            )
                            print("MP4 saved successfully with minimal settings!")
                        except Exception as e:
                            print(f"Minimal MP4 save failed: {e}")
            except Exception as mp4_error:
                print(f"MP4 saving failed completely: {mp4_error}")

            # If mp4_only is False, also save as GIF
            if not mp4_only: 
                # Save as GIF (as backup)
                try:
                    print(f"Saving animation to {file_name}.gif...")
                    from matplotlib.animation import PillowWriter
                    
                    # Use higher quality settings for GIF
                    animation_obj.save(
                        f"{file_name}.gif",
                        writer=PillowWriter(fps=8),
                        dpi=150  # Higher DPI for better quality
                    )
                    print("GIF saved successfully!")
                except Exception as gif_error:
                    print(f"GIF save failed: {gif_error}")
                    
                    # Try with minimal settings
                    try:
                        print("Trying with minimal GIF settings...")
                        animation_obj.save(
                            f"{file_name}.gif",
                            writer=PillowWriter(fps=5),
                            dpi=100
                        )
                        print("GIF saved successfully with minimal settings!")
                    except Exception as e:
                        print(f"Minimal GIF save failed: {e}")
            else:
                print("Skipping GIF creation as mp4_only=True.")
                    
        except Exception as e:
            print(f"Error in animation saving: {e}")
            import traceback
            traceback.print_exc()

    def add_data(self, uav):
        """Add UAV data to animation dataframe."""
        self.df = self.df._append(
            {
                "current_time_step": self.env.current_time_step,
                "uav_id": uav.id,
                "uav": uav,
                "current_position": uav.current_position,
                "current_heading": uav.current_heading,
                "final_heading": math.atan2(uav.end.y - uav.current_position.y, 
                                        uav.end.x - uav.current_position.x),
            },
            ignore_index=True,
        )

    def render_static_assets(self, ax):
        """Render the static assets of the environment (map, restricted areas)."""
        # Draw map boundaries
        self.env.airspace.location_utm_gdf.plot(ax=ax, color="gray", linewidth=0.6, alpha=0.5)
        

        
        # Grid
        
        
        
        # Draw restricted areas
        if hasattr(self.env.airspace, 'location_tags'):
            for tag_value in self.env.airspace.location_tags.keys():
                # Draw actual restricted areas
                self.env.airspace.location_utm[tag_value].plot(ax=ax, color="red", alpha=0.7)
                # Draw buffer zones
                self.env.airspace.location_utm_buffer[tag_value].plot(ax=ax, color="orange", alpha=0.3)
        
        # # Draw vertiports
        # vertiport_points = [v for v in self.env.airspace.get_vertiport_list()]
        # if vertiport_points:
        #     gpd.GeoSeries(vertiport_points).plot(ax=ax, color="black", markersize=10)

        # Draw vertiports - extract just the Point geometries instead of using the Vertiport objects
        vertiport_points = self.env.airspace.get_vertiport_list()
        if vertiport_points:
            # Plot vertiports directly using matplotlib instead of GeoSeries
            for vp in vertiport_points:
                ax.plot(vp.x, vp.y, 'o', markersize=10, color="black")

    def reset(self):
        """Reset the renderer state."""
        # Reset trajectory tracking
        self.trajectory_by_id = {}
        
        # Reset animation data
        self.df = pd.DataFrame({
            "current_time_step": [],
            "uav_id": [],
            "uav": [],
            "current_position": [],
            "current_heading": [],
            "final_heading": [],
        })
        
        # Clean up animation data if present
        if hasattr(self, 'animation_data'):
            del self.animation_data
        if hasattr(self, 'animation_time_steps'):
            del self.animation_time_steps
        if hasattr(self, 'animation_trajectories'):
            del self.animation_trajectories
    
    def close(self):
        """Close the renderer and clean up resources."""
        plt.close('all')
        
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
        self.reset()
