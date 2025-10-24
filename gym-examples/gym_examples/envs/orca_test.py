import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
from map_env_revised import MapEnv
import os
import math
from matplotlib.patches import Circle, FancyArrowPatch
from shapely import Point
from copy import deepcopy

def test_orca_collision_avoidance():
    """
    Create renders showing two ORCA UAVs avoiding a head-on collision.
    Captures three critical moments:
    1. Before collision (approaching)
    2. During avoidance (changing headings)
    3. After collision (safely past)
    """
    print("Creating ORCA collision avoidance test environment...")
    seed = 42
    
    # Create environment with 2 ORCA UAVs only
    env = MapEnv(
        number_of_uav=0,  # No basic UAVs
        num_ORCA_uav=2,   # 2 ORCA UAVs
        number_of_vertiport=2,  # Only 2 vertiports for head-on setup
        location_name="Austin, Texas, USA",
        airspace_tag_list=[],  # No restricted areas for cleaner view
        max_episode_steps=1000,
        seed=seed,
        obs_space_str="UAM_UAV",
        sorting_criteria='closest first',
        render_mode=None,
        max_uavs=5,
        max_vertiports=5,
    )
    
    # Reset environment
    obs, info = env.reset(seed=seed)
    
    # Manually override positions to guarantee head-on collision course
    from shapely import Point
    
    # Override vertiport positions for guaranteed head-on setup
    vertiport_1 = Point(615000, 3346000)
    vertiport_2 = Point(625000, 3346000)  # 10km apart, same y-coordinate
    
    env.airspace.vertiport_list = [vertiport_1, vertiport_2]
    
    # Get the two ORCA UAVs
    orca_uav_1 = env.ORCA_agent_list[0]
    orca_uav_2 = env.ORCA_agent_list[1]
    
    # Assign them opposite start/end points for head-on collision
    # UAV 1: left to right
    orca_uav_1.start = vertiport_1
    orca_uav_1.end = vertiport_2
    orca_uav_1.current_position = vertiport_1
    
    # UAV 2: right to left
    orca_uav_2.start = vertiport_2
    orca_uav_2.end = vertiport_1
    orca_uav_2.current_position = vertiport_2
    
    # Initialize their headings toward each other
    orca_uav_1.current_heading = 0.0  # Heading east
    orca_uav_2.current_heading = math.pi  # Heading west
    
    # Reset RVO2 sim with new positions
    env.rvo2_sim.reset()
    
    print("\n" + "="*60)
    print("SETUP: Two ORCA UAVs on head-on collision course")
    print("="*60)
    print(f"UAV 1: Start={orca_uav_1.start}, End={orca_uav_1.end}")
    print(f"UAV 2: Start={orca_uav_2.start}, End={orca_uav_2.end}")
    print(f"Initial separation: {orca_uav_1.current_position.distance(orca_uav_2.current_position):.1f}m")
    
    # Initialize trajectory tracking
    env.renderer.trajectory_by_id[orca_uav_1.id] = []
    env.renderer.trajectory_by_id[orca_uav_2.id] = []
    
    # Store snapshots at each moment
    snapshots = {
        'before': None,
        'during': None,
        'after': None
    }
    
    min_distance = float('inf')
    min_distance_step = 0
    heading_change_detected = False
    passed_each_other = False
    
    # Store initial headings for comparison
    initial_heading_1 = 0.0
    initial_heading_2 = math.pi
    
    print("\n" + "="*60)
    print("PHASE: Simulating collision avoidance...")
    print("="*60)
    
    # Run simulation
    for step in range(800):
        # Sample random action for learning agent (not used, but required by step())
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track trajectories
        pos1 = orca_uav_1.current_position
        pos2 = orca_uav_2.current_position
        env.renderer.trajectory_by_id[orca_uav_1.id].append((pos1.x, pos1.y))
        env.renderer.trajectory_by_id[orca_uav_2.id].append((pos2.x, pos2.y))
        
        # Track minimum distance
        current_distance = pos1.distance(pos2)
        if current_distance < min_distance:
            min_distance = current_distance
            min_distance_step = step
        
        # Capture "before" moment - when distance is around 2000m
        if snapshots['before'] is None and current_distance < 650:
            snapshots['before'] = {
                'step': step,
                'uav1_pos': Point(pos1.x, pos1.y),
                'uav1_heading': orca_uav_1.current_heading,
                'uav1_speed': orca_uav_1.current_speed,
                'uav2_pos': Point(pos2.x, pos2.y),
                'uav2_heading': orca_uav_2.current_heading,
                'uav2_speed': orca_uav_2.current_speed,
                'trajectories': {
                    orca_uav_1.id: list(env.renderer.trajectory_by_id[orca_uav_1.id]),
                    orca_uav_2.id: list(env.renderer.trajectory_by_id[orca_uav_2.id])
                },
                'distance': current_distance
            }
            print(f"Step {step}: BEFORE captured - Distance: {current_distance:.1f}m")
        
        # Detect heading changes (collision avoidance activation)
        if not heading_change_detected and step > 30:
            # Check if either UAV has deviated significantly from straight path
            heading_deviation_1 = abs(orca_uav_1.current_heading - initial_heading_1)
            heading_deviation_2 = abs(abs(orca_uav_2.current_heading) - math.pi)
            
            if heading_deviation_1 > 0.15 or heading_deviation_2 > 0.15:
                heading_change_detected = True
                snapshots['during'] = {
                    'step': step,
                    'uav1_pos': Point(pos1.x, pos1.y),
                    'uav1_heading': orca_uav_1.current_heading,
                    'uav1_speed': orca_uav_1.current_speed,
                    'uav2_pos': Point(pos2.x, pos2.y),
                    'uav2_heading': orca_uav_2.current_heading,
                    'uav2_speed': orca_uav_2.current_speed,
                    'trajectories': {
                        orca_uav_1.id: list(env.renderer.trajectory_by_id[orca_uav_1.id]),
                        orca_uav_2.id: list(env.renderer.trajectory_by_id[orca_uav_2.id])
                    },
                    'distance': current_distance
                }
                print(f"Step {step}: DURING captured - Avoidance activated! Distance: {current_distance:.1f}m")
                print(f"  UAV1 heading deviation: {math.degrees(heading_deviation_1):.1f}°")
                print(f"  UAV2 heading deviation: {math.degrees(heading_deviation_2):.1f}°")
        
        # Detect when they've passed each other
        if not passed_each_other and heading_change_detected:
            # Check if they're moving away from each other
            if step > snapshots['during']['step'] + 30 and current_distance > min_distance + 200:
                passed_each_other = True
                snapshots['after'] = {
                    'step': step,
                    'uav1_pos': Point(pos1.x, pos1.y),
                    'uav1_heading': orca_uav_1.current_heading,
                    'uav1_speed': orca_uav_1.current_speed,
                    'uav2_pos': Point(pos2.x, pos2.y),
                    'uav2_heading': orca_uav_2.current_heading,
                    'uav2_speed': orca_uav_2.current_speed,
                    'trajectories': {
                        orca_uav_1.id: list(env.renderer.trajectory_by_id[orca_uav_1.id]),
                        orca_uav_2.id: list(env.renderer.trajectory_by_id[orca_uav_2.id])
                    },
                    'distance': current_distance
                }
                print(f"Step {step}: AFTER captured - UAVs safely passed! Distance: {current_distance:.1f}m")
        
        # Progress update
        if step % 100 == 0:
            print(f"Step {step}: Distance={current_distance:.1f}m, "
                  f"UAV1 heading={math.degrees(orca_uav_1.current_heading):.1f}°, "
                  f"UAV2 heading={math.degrees(orca_uav_2.current_heading):.1f}°")
        
        # Stop if all moments captured
        if all(snapshots.values()):
            break
    
    print(f"\nMinimum distance achieved: {min_distance:.1f}m at step {min_distance_step}")
    
    # Create output directory
    os.makedirs("test_results/orca_collision", exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating visualization renders...")
    print("="*60)
    
    # Create renders for each moment using snapshots
    moments = ['before', 'during', 'after']
    for moment in moments:
        snapshot = snapshots[moment]
        if snapshot is None:
            print(f"Warning: Could not capture '{moment}' moment")
            continue
        
        print(f"\nRendering '{moment}' phase at step {snapshot['step']}...")
        
        # Create dual UAV zoomed view using snapshot data
        create_dual_orca_render_from_snapshot(
            env, 
            orca_uav_1, 
            orca_uav_2,
            snapshot,
            f"test_results/orca_collision/collision_{moment}.png",
            f"ORCA Collision Avoidance - {moment.title()} (Step {snapshot['step']})"
        )

    # Create combined subplot figure
    print("\nCreating combined subplot figure...")
    create_combined_orca_renders(
        env,
        orca_uav_1,
        orca_uav_2,
        snapshots,
        "test_results/orca_collision/collision_combined.png"
    )
    
    env.close()
    
    print("\n" + "="*60)
    print("ORCA Collision Avoidance Test Complete!")
    print("="*60)
    print(f"\nGenerated files in test_results/orca_collision/:")
    print("  - collision_before.png")
    print("  - collision_during.png")
    print("  - collision_after.png")

def create_dual_orca_render_from_snapshot(env, uav1, uav2, snapshot, save_path, title):
    """
    Create a zoomed render showing both ORCA UAVs using snapshot data.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Draw static assets (gray background)
    env.renderer.render_static_assets(ax)
    
    # Extract snapshot data
    pos1 = snapshot['uav1_pos']
    heading1 = snapshot['uav1_heading']
    speed1 = snapshot['uav1_speed']
    
    pos2 = snapshot['uav2_pos']
    heading2 = snapshot['uav2_heading']
    speed2 = snapshot['uav2_speed']
    
    distance = snapshot['distance']
    
    # Calculate midpoint for zoom center
    center_x = (pos1.x + pos2.x) / 2
    center_y = (pos1.y + pos2.y) / 2
    
    # Draw vertiports
    for vertiport in env.airspace.get_vertiport_list():
        ax.plot(vertiport.x, vertiport.y, 'gs', markersize=15)
    
    # Draw both UAVs
    uav_data = [
        (uav1, pos1, heading1, speed1, 'uav1'),
        (uav2, pos2, heading2, speed2, 'uav2')
    ]
    
    for uav, pos, heading, speed, color_scheme in uav_data:
        # Color schemes for differentiation
        if color_scheme == 'uav1':
            detection_color = '#0278c2'
            nmac_color = '#FF7F50'
            body_color = '#0000A0'
            traj_color = '#0000A0'
            ref_color = 'purple'
        else:
            detection_color = '#00C278'
            nmac_color = '#FFA500'
            body_color = '#008000'
            traj_color = '#008000'
            ref_color = 'red'
        
        # Detection radius
        detection = Circle((pos.x, pos.y), uav.detection_radius,
                          fill=False, color=detection_color, alpha=0.3, linewidth=2)
        ax.add_patch(detection)
        
        # NMAC radius
        nmac = Circle((pos.x, pos.y), uav.nmac_radius,
                     fill=False, color=nmac_color, alpha=0.5, linewidth=3)
        ax.add_patch(nmac)
        
        # UAV body
        body = Circle((pos.x, pos.y), uav.radius,
                     fill=True, color=body_color, alpha=0.9)
        ax.add_patch(body)
        
        # Current heading arrow
        heading_length = uav.radius * 8
        dx = heading_length * np.cos(heading)
        dy = heading_length * np.sin(heading)
        arrow = FancyArrowPatch((pos.x, pos.y),
                               (pos.x + dx, pos.y + dy),
                               color='black',
                               arrowstyle='->',
                               mutation_scale=15,
                               linewidth=3.5)
        ax.add_patch(arrow)
        
        # Reference heading arrow
        final_heading = math.atan2(uav.end.y - pos.y, uav.end.x - pos.x)
        ref_length = uav.radius * 6
        dx_ref = ref_length * np.cos(final_heading)
        dy_ref = ref_length * np.sin(final_heading)
        ref_arrow = FancyArrowPatch((pos.x, pos.y),
                                   (pos.x + dx_ref, pos.y + dy_ref),
                                   color=ref_color,
                                   arrowstyle='->',
                                   mutation_scale=12,
                                   linewidth=2.5,
                                   alpha=0.7)
        ax.add_patch(ref_arrow)
        
        # Start-end line
        ax.plot([uav.start.x, uav.end.x],
               [uav.start.y, uav.end.y],
               '--', color=traj_color, alpha=0.4, linewidth=2.0)
        
        # Trajectory from snapshot
        if uav.id in snapshot['trajectories']:
            traj = snapshot['trajectories'][uav.id]
            if len(traj) > 1:
                xs, ys = zip(*traj)
                ax.plot(xs, ys, '-', linewidth=3, alpha=0.7, color=traj_color)
    
    # Draw distance line between UAVs
    ax.plot([pos1.x, pos2.x], [pos1.y, pos2.y], 
           'r--', linewidth=2, alpha=0.5, label=f'Separation: {distance:.1f}m')
    
    # Set zoom centered on midpoint
    separation = distance
    zoom_margin = max(separation * 2, 400)
    ax.set_xlim(center_x - zoom_margin, center_x + zoom_margin)
    ax.set_ylim(center_y - zoom_margin, center_y + zoom_margin)
    
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add text annotations with snapshot data
    info_text = (
        f"UAV 1 (Blue): Speed={speed1:.1f} m/s, Heading={math.degrees(heading1):.1f}°\n"
        f"UAV 2 (Green): Speed={speed2:.1f} m/s, Heading={math.degrees(heading2):.1f}°\n"
        f"Separation: {distance:.1f}m | NMAC Radius: {uav1.nmac_radius}m"
    )
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes, fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='#0000A0', linewidth=3, label='UAV 1 Trajectory (Blue)'),
        plt.Line2D([0], [0], color='#008000', linewidth=3, label='UAV 2 Trajectory (Green)'),
        plt.Line2D([0], [0], color='#0278c2', linewidth=3, alpha=0.3, label='UAV 1 Detection Radius (500m)'),
        plt.Line2D([0], [0], color='#00C278', linewidth=3, alpha=0.3, label='UAV 2 Detection Radius (500m)'),
        plt.Line2D([0], [0], color='#FF7F50', linewidth=3, label='UAV 1 NMAC Radius (200m)'),
        plt.Line2D([0], [0], color='#FFA500', linewidth=3, label='UAV 2 NMAC Radius (200m)'),
        plt.Line2D([0], [0], color='black', linewidth=3, label='Current Heading'),
        plt.Line2D([0], [0], color='purple', linewidth=2, alpha=0.7, label='UAV 1 Reference Heading'),
        plt.Line2D([0], [0], color='red', linewidth=2, alpha=0.7, label='UAV 2 Reference Heading'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved render to: {save_path}")

def create_combined_orca_renders(env, uav1, uav2, snapshots, save_path):
    """
    Create a combined figure with all three collision avoidance phases.
    Styling matches individual renders exactly.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    moments = ['before', 'during', 'after']
    titles = ['Before - Step {step}', 'During - Step {step}', 'After - Step {step}']
    
    for idx, (ax, moment, title_template) in enumerate(zip(axes, moments, titles)):
        snapshot = snapshots[moment]
        if snapshot is None:
            continue

        env.renderer.render_static_assets(ax)
            
        # Extract snapshot data
        pos1 = snapshot['uav1_pos']
        heading1 = snapshot['uav1_heading']
        speed1 = snapshot['uav1_speed']
        pos2 = snapshot['uav2_pos']
        heading2 = snapshot['uav2_heading']
        speed2 = snapshot['uav2_speed']
        distance = snapshot['distance']
        
        # Calculate midpoint
        center_x = (pos1.x + pos2.x) / 2
        center_y = (pos1.y + pos2.y) / 2
        
        # Draw vertiports - MATCH individual render markersize
        for vertiport in env.airspace.get_vertiport_list():
            ax.plot(vertiport.x, vertiport.y, 'gs', markersize=15)
        
        # Draw both UAVs
        uav_data = [
            (uav1, pos1, heading1, speed1, 'uav1'),
            (uav2, pos2, heading2, speed2, 'uav2')
        ]
        
        for uav, pos, heading, speed, color_scheme in uav_data:
            if color_scheme == 'uav1':
                detection_color = '#0278c2'
                nmac_color = '#FF7F50'
                body_color = '#0000A0'
                traj_color = '#0000A0'
                ref_color = 'purple'
            else:
                detection_color = '#00C278'
                nmac_color = '#FFA500'
                body_color = '#008000'
                traj_color = '#008000'
                ref_color = 'red'
            
            # Detection radius
            detection = Circle((pos.x, pos.y), uav.detection_radius,
                              fill=False, color=detection_color, alpha=0.3, linewidth=2)
            ax.add_patch(detection)
            
            # NMAC radius
            nmac = Circle((pos.x, pos.y), uav.nmac_radius,
                         fill=False, color=nmac_color, alpha=0.5, linewidth=3)
            ax.add_patch(nmac)
            
            # UAV body
            body = Circle((pos.x, pos.y), uav.radius,
                         fill=True, color=body_color, alpha=0.9)
            ax.add_patch(body)
            
            # Current heading arrow
            heading_length = uav.radius * 8
            dx = heading_length * np.cos(heading)
            dy = heading_length * np.sin(heading)
            arrow = FancyArrowPatch((pos.x, pos.y),
                                   (pos.x + dx, pos.y + dy),
                                   color='black',
                                   arrowstyle='->',
                                   mutation_scale=15,
                                   linewidth=3.5)
            ax.add_patch(arrow)
            
            # Reference heading arrow
            final_heading = math.atan2(uav.end.y - pos.y, uav.end.x - pos.x)
            ref_length = uav.radius * 6
            dx_ref = ref_length * np.cos(final_heading)
            dy_ref = ref_length * np.sin(final_heading)
            ref_arrow = FancyArrowPatch((pos.x, pos.y),
                                       (pos.x + dx_ref, pos.y + dy_ref),
                                       color=ref_color,
                                       arrowstyle='->',
                                       mutation_scale=12,
                                       linewidth=2.5,
                                       alpha=0.7)
            ax.add_patch(ref_arrow)
            
            # Start-end line
            ax.plot([uav.start.x, uav.end.x],
                   [uav.start.y, uav.end.y],
                   '--', color=traj_color, alpha=0.4, linewidth=2.0)
            
            # Trajectory
            if uav.id in snapshot['trajectories']:
                traj = snapshot['trajectories'][uav.id]
                if len(traj) > 1:
                    xs, ys = zip(*traj)
                    ax.plot(xs, ys, '-', linewidth=3, alpha=0.7, color=traj_color)
        
        # Draw distance line
        ax.plot([pos1.x, pos2.x], [pos1.y, pos2.y], 
               'r--', linewidth=2, alpha=0.5)
        
        # Set zoom with same margins as individual renders
        separation = distance
        zoom_margin = max(separation * 2, 400)
            
        ax.set_xlim(center_x - zoom_margin, center_x + zoom_margin)
        ax.set_ylim(center_y - zoom_margin, center_y + zoom_margin)
        
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_aspect('equal')
        
        # Title - MATCH individual render fontsize
        ax.set_title(title_template.format(step=snapshot['step']), 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Info text - MATCH individual render fontsize and alpha
        info_text = (
            f"UAV 1 (Blue): Speed={speed1:.1f} m/s, Heading={math.degrees(heading1):.1f}°\n"
            f"UAV 2 (Green): Speed={speed2:.1f} m/s, Heading={math.degrees(heading2):.1f}°\n"
            f"Separation: {distance:.1f}m | NMAC Radius: {uav1.nmac_radius}m"
        )
        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add full legend to rightmost subplot only - MATCH individual render exactly
    legend_elements = [
        plt.Line2D([0], [0], color='#0000A0', linewidth=3, label='UAV 1 Trajectory (Blue)'),
        plt.Line2D([0], [0], color='#008000', linewidth=3, label='UAV 2 Trajectory (Green)'),
        plt.Line2D([0], [0], color='#0278c2', linewidth=3, alpha=0.3, label='UAV 1 Detection Radius (500m)'),
        plt.Line2D([0], [0], color='#00C278', linewidth=3, alpha=0.3, label='UAV 2 Detection Radius (500m)'),
        plt.Line2D([0], [0], color='#FF7F50', linewidth=3, label='UAV 1 NMAC Radius (200m)'),
        plt.Line2D([0], [0], color='#FFA500', linewidth=3, label='UAV 2 NMAC Radius (200m)'),
        plt.Line2D([0], [0], color='black', linewidth=3, label='Current Heading'),
        plt.Line2D([0], [0], color='purple', linewidth=2, alpha=0.7, label='UAV 1 Reference Heading'),
        plt.Line2D([0], [0], color='red', linewidth=2, alpha=0.7, label='UAV 2 Reference Heading'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined render to: {save_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("ORCA Head-On Collision Avoidance Test")
    print("=" * 60)
    print("\nThis test demonstrates ORCA collision avoidance with:")
    print("- Two ORCA UAVs on a guaranteed head-on collision course")
    print("- Visualization of three critical moments:")
    print("  1. Before: UAVs approaching each other")
    print("  2. During: UAVs detecting collision and changing headings")
    print("  3. After: UAVs safely passing each other")
    print("=" * 60)
    print()
    
    test_orca_collision_avoidance()