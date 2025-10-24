import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
from map_env_revised import MapEnv
import os
import math
from matplotlib.patches import Circle, FancyArrowPatch

def test_uav_rendering():
    """
    Create zoomed renders for each UAV type at initialization and after movement.
    Also creates map renders at both stages.
    """
    print("Creating test environment for UAV rendering verification...")
    seed = 6
    
    # Create environment with 1 basic UAV and 1 ORCA UAV
    env = MapEnv(
        number_of_uav=1,  # 1 basic non-learning UAV
        num_ORCA_uav=1,   # 1 ORCA UAV
        number_of_vertiport=5,
        location_name="Austin, Texas, USA",
        airspace_tag_list= [], # [("amenity", "hospital"), ("aeroway", "aerodrome")],  # No restricted areas for cleaner view
        max_episode_steps=500,
        seed=seed,
        obs_space_str="UAM_UAV",
        sorting_criteria='closest first',
        render_mode=None,
        max_uavs=30, # this is maximum number of UAVs allowed in env
        max_vertiports=50, # this is maximum number of vertiports allowed in env
    )
    
    # Reset environment
    obs, info = env.reset(seed=seed)
    
    from auto_uav_v2 import Auto_UAV_v2
    
    # Identify UAV types
    agent_uav = env.agent
    basic_uav = None
    orca_uav = None
    
    for uav in env.atc.get_uav_list():
        if not isinstance(uav, Auto_UAV_v2):
            # Check if it's an ORCA UAV (exists in ORCA_agent_list)
            if uav in env.ORCA_agent_list:
                orca_uav = uav
            else:
                basic_uav = uav
    
    print("\n" + "="*60)
    print("PHASE 1: Initial State (at reset)")
    print("="*60)
    
    # Create initial renders
    create_map_render(env, "test_results/map_render_initial.png", "Initial State")
    
    if agent_uav:
        create_zoomed_uav_render(env, agent_uav, "agent", 
                                "test_results/agent_uav_initial.png")
    
    if basic_uav:
        create_zoomed_uav_render(env, basic_uav, "basic", 
                                "test_results/basic_uav_initial.png")
    
    if orca_uav:
        create_zoomed_uav_render(env, orca_uav, "orca", 
                                "test_results/orca_uav_initial.png")
    
    print("\n" + "="*60)
    print("PHASE 2: Taking steps to build trajectories...")
    print("="*60)
    
    # Take steps to build up trajectory
    for step in range(300):  # More steps to ensure ORCA UAV moves
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Manually update trajectory tracking (since render_mode=None, render() isn't called)
        # Track agent trajectory
        agent_pos = env.agent.current_position
        if env.agent.id not in env.renderer.trajectory_by_id:
            env.renderer.trajectory_by_id[env.agent.id] = []
        env.renderer.trajectory_by_id[env.agent.id].append((agent_pos.x, agent_pos.y))
        
        # Track non-learning UAV trajectories
        for uav in env.atc.get_uav_list():
            if not isinstance(uav, Auto_UAV_v2):
                pos = uav.current_position
                if uav.id not in env.renderer.trajectory_by_id:
                    env.renderer.trajectory_by_id[uav.id] = []
                env.renderer.trajectory_by_id[uav.id].append((pos.x, pos.y))
        
        if step % 5 == 0:
            print(f"Step {step}:")
            print(f"  Agent: pos=({agent_uav.current_position.x:.1f}, {agent_uav.current_position.y:.1f}), speed={agent_uav.current_speed:.2f}")
            if basic_uav:
                print(f"  Basic: pos=({basic_uav.current_position.x:.1f}, {basic_uav.current_position.y:.1f}), speed={basic_uav.current_speed:.2f}")
            if orca_uav:
                print(f"  ORCA:  pos=({orca_uav.current_position.x:.1f}, {orca_uav.current_position.y:.1f}), speed={orca_uav.current_speed:.2f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    print("\n" + "="*60)
    print("PHASE 3: After Movement (with trajectories)")
    print("="*60)
    
    # Print trajectory statistics
    print("\nTrajectory Statistics:")
    for uav_id, traj in env.renderer.trajectory_by_id.items():
        print(f"  UAV {uav_id}: {len(traj)} trajectory points")
    
    # Create renders after movement
    create_map_render(env, "test_results/map_render_after_movement.png", f"After {env.current_time_step} Steps")
    print(f"\nSteps completed: {env.current_time_step}")
    
    if agent_uav:
        create_zoomed_uav_render(env, agent_uav, "agent", 
                                "test_results/agent_uav_after_movement.png")
    
    if basic_uav:
        create_zoomed_uav_render(env, basic_uav, "basic", 
                                "test_results/basic_uav_after_movement.png")
    
    if orca_uav:
        create_zoomed_uav_render(env, orca_uav, "orca", 
                                "test_results/orca_uav_after_movement.png")
    
    env.close()
    
    print("\n" + "="*60)
    print("Rendering test completed!")
    print("="*60)
    print("\nGenerated files:")
    print("\nInitial State:")
    print("  - test_results/map_render_initial.png")
    if agent_uav:
        print("  - test_results/agent_uav_initial.png")
    if basic_uav:
        print("  - test_results/basic_uav_initial.png")
    if orca_uav:
        print("  - test_results/orca_uav_initial.png")
    print("\nAfter Movement:")
    print("  - test_results/map_render_after_movement.png")
    if agent_uav:
        print("  - test_results/agent_uav_after_movement.png")
    if basic_uav:
        print("  - test_results/basic_uav_after_movement.png")
    if orca_uav:
        print("  - test_results/orca_uav_after_movement.png")

def create_map_render(env, save_path, title_suffix):
    """
    Create a full map render showing all UAVs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Draw static assets (gray background)
    env.renderer.render_static_assets(ax)
    
    from auto_uav_v2 import Auto_UAV_v2
    
    # Collect positions for calculating view limits
    current_positions = []
    current_uav_ids = []
    
    # Draw vertiports (MATCH MapRenderer)
    for vertiport in env.airspace.get_vertiport_list():
        ax.plot(vertiport.x, vertiport.y, 'gs', markersize=12)
    
    # Draw learning agent
    agent_pos = env.agent.current_position
    current_positions.append((agent_pos.x, agent_pos.y))
    current_uav_ids.append(env.agent.id)
    
    # Agent detection radius
    agent_detection = Circle((agent_pos.x, agent_pos.y),
                        env.agent.detection_radius,
                        fill=False, color='#0278c2', alpha=0.3, linewidth=2)
    ax.add_patch(agent_detection)
    
    # Agent NMAC radius
    agent_nmac = Circle((agent_pos.x, agent_pos.y),
                    env.agent.nmac_radius,
                    fill=False, color='#FF7F50', alpha=0.4, linewidth=2)
    ax.add_patch(agent_nmac)
    
    # Agent body
    agent_body = Circle((agent_pos.x, agent_pos.y),
                    env.agent.radius,
                    fill=True, color='#0000A0', alpha=0.9)
    ax.add_patch(agent_body)
    
    # Agent heading
    heading_length = env.agent.radius * 5
    dx = heading_length * np.cos(env.agent.current_heading)
    dy = heading_length * np.sin(env.agent.current_heading)
    agent_arrow = FancyArrowPatch((agent_pos.x, agent_pos.y),
                            (agent_pos.x + dx, agent_pos.y + dy),
                            color='black',
                            arrowstyle='->',
                            mutation_scale=10,
                            linewidth=2.5)
    ax.add_patch(agent_arrow)
    
    # Agent start-end line
    ax.plot([env.agent.start.x, env.agent.end.x],
            [env.agent.start.y, env.agent.end.y],
            'b--', alpha=0.6, linewidth=2.0)
    
    # Draw non-learning UAVs
    for uav in env.atc.get_uav_list():
        if isinstance(uav, Auto_UAV_v2):
            continue
            
        pos = uav.current_position
        current_positions.append((pos.x, pos.y))
        current_uav_ids.append(uav.id)
        
        # UAV detection radius
        detection = Circle((pos.x, pos.y), uav.detection_radius, 
                    fill=False, color='green', alpha=0.3, linewidth=2)
        ax.add_patch(detection)
        
        # UAV NMAC radius
        nmac = Circle((pos.x, pos.y), uav.nmac_radius, 
                fill=False, color='orange', alpha=0.4, linewidth=2)
        ax.add_patch(nmac)
        
        # UAV body
        body = Circle((pos.x, pos.y), uav.radius, 
                fill=True, color='blue', alpha=0.7)
        ax.add_patch(body)
        
        # UAV heading
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
        
        # UAV start-end line
        ax.plot([uav.start.x, uav.end.x],
                [uav.start.y, uav.end.y],
                'g--', alpha=0.6, linewidth=2.0)
    
    # Draw trajectories (MATCH MapRenderer)
    if hasattr(env.renderer, 'trajectory_by_id'):
        for uav_id in current_uav_ids:
            if uav_id in env.renderer.trajectory_by_id and len(env.renderer.trajectory_by_id[uav_id]) > 1:
                xs, ys = zip(*env.renderer.trajectory_by_id[uav_id])
                line_color = '#0000A0' if uav_id == env.agent.id else 'blue'
                ax.plot(xs, ys, '-', linewidth=2.5, alpha=0.6, color=line_color)
    
    # Set limits to show full environment
    if current_positions:
        all_x = [pos[0] for pos in current_positions]
        all_y = [pos[1] for pos in current_positions]
        
        # Add vertiport positions
        vp_x = [v.x for v in env.airspace.get_vertiport_list()]
        vp_y = [v.y for v in env.airspace.get_vertiport_list()]
        
        all_x.extend(vp_x)
        all_y.extend(vp_y)
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        margin = max(500, (x_max - x_min) * 0.1)
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
    
    ax.set_title(f'UrbanNav UAM Operation in Austin, Texas - {title_suffix}', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved map render to: {save_path}")

def create_zoomed_uav_render(env, target_uav, uav_type, save_path):
    """
    Create a zoomed-in render focusing on a specific UAV.
    Colors and styling match MapRenderer exactly.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Add gray background (matches MapRenderer)
    env.renderer.render_static_assets(ax)
    
    target_pos = target_uav.current_position
    
    from auto_uav_v2 import Auto_UAV_v2
    is_agent = isinstance(target_uav, Auto_UAV_v2)
    
    # Set colors based on UAV type (MATCH MapRenderer exactly)
    if is_agent:
        detection_color = '#0278c2'
        detection_alpha = 0.3
        nmac_color = '#FF7F50'
        nmac_alpha = 0.4
        body_color = '#0000A0'
        body_alpha = 0.9
        start_end_color = 'b'
        trajectory_color = '#0000A0'
    else:
        detection_color = 'green'
        detection_alpha = 0.3
        nmac_color = 'orange'
        nmac_alpha = 0.4
        body_color = 'blue'
        body_alpha = 0.7
        start_end_color = 'g'
        trajectory_color = 'blue'
    
    # 1. Detection radius
    detection = Circle((target_pos.x, target_pos.y),
                      target_uav.detection_radius,
                      fill=False, color=detection_color, alpha=detection_alpha, linewidth=2)
    ax.add_patch(detection)
    
    # 2. NMAC radius
    nmac = Circle((target_pos.x, target_pos.y),
                  target_uav.nmac_radius,
                  fill=False, color=nmac_color, alpha=nmac_alpha, linewidth=2)
    ax.add_patch(nmac)
    
    # 3. UAV body
    body = Circle((target_pos.x, target_pos.y),
                  target_uav.radius,
                  fill=True, color=body_color, alpha=body_alpha)
    ax.add_patch(body)
    
    # 4. Current heading - BLACK arrow
    heading_length = target_uav.radius * 5
    dx = heading_length * np.cos(target_uav.current_heading)
    dy = heading_length * np.sin(target_uav.current_heading)
    current_heading_arrow = FancyArrowPatch((target_pos.x, target_pos.y),
                            (target_pos.x + dx, target_pos.y + dy),
                            color='black',
                            arrowstyle='->',
                            mutation_scale=10,
                            linewidth=2.5)
    ax.add_patch(current_heading_arrow)
    
    # 5. Reference heading (purple for agent, red for others)
    final_heading = math.atan2(target_uav.end.y - target_pos.y, 
                              target_uav.end.x - target_pos.x)
    ref_length = target_uav.radius * 4
    dx_ref = ref_length * np.cos(final_heading)
    dy_ref = ref_length * np.sin(final_heading)
    ref_color = 'purple' if is_agent else 'red'
    ref_heading_arrow = FancyArrowPatch((target_pos.x, target_pos.y),
                        (target_pos.x + dx_ref, target_pos.y + dy_ref),
                        color=ref_color,
                        arrowstyle='->',
                        mutation_scale=7.5,
                        linewidth=2,
                        alpha=0.6)
    ax.add_patch(ref_heading_arrow)
    
    # 6. Start-end connection line
    ax.plot([target_uav.start.x, target_uav.end.x],
            [target_uav.start.y, target_uav.end.y],
            f'{start_end_color}--', alpha=0.6, linewidth=2.0)
    
    # 7. Trajectory
    if hasattr(env.renderer, 'trajectory_by_id') and target_uav.id in env.renderer.trajectory_by_id:
        traj = env.renderer.trajectory_by_id[target_uav.id]
        if len(traj) > 1:
            xs, ys = zip(*traj)
            ax.plot(xs, ys, '-', linewidth=2.5, alpha=0.6, color=trajectory_color)
    
    # Draw vertiports
    for vertiport in env.airspace.get_vertiport_list():
        ax.plot(vertiport.x, vertiport.y, 'gs', markersize=12)
    
    # Mark start and end points
    ax.plot(target_uav.start.x, target_uav.start.y, 'go', markersize=10)
    ax.plot(target_uav.end.x, target_uav.end.y, 'ro', markersize=10)
    
    # Draw other UAVs for context (faded)
    for uav in [env.agent] + env.atc.get_uav_list():
        if uav.id == target_uav.id:
            continue
            
        pos = uav.current_position
        is_other_agent = isinstance(uav, Auto_UAV_v2)
        
        if is_other_agent:
            det_col, nmac_col, body_col = '#0278c2', '#FF7F50', '#0000A0'
        else:
            det_col, nmac_col, body_col = 'green', 'orange', 'blue'
        
        detection = Circle((pos.x, pos.y), uav.detection_radius, 
                    fill=False, color=det_col, alpha=0.15, linewidth=1)
        ax.add_patch(detection)
        
        nmac = Circle((pos.x, pos.y), uav.nmac_radius, 
                fill=False, color=nmac_col, alpha=0.2, linewidth=1)
        ax.add_patch(nmac)
        
        body = Circle((pos.x, pos.y), uav.radius, 
                fill=True, color=body_col, alpha=0.4)
        ax.add_patch(body)
    
    # Set zoom
    zoom_margin = max(target_uav.detection_radius * 2.6, 1000)
    ax.set_xlim(target_pos.x - zoom_margin, target_pos.x + zoom_margin)
    ax.set_ylim(target_pos.y - zoom_margin, target_pos.y + zoom_margin)
    
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_aspect('equal')
    
    # Title
    uav_type_name = {
        'agent': 'Learning Agent (Auto_UAV_v2)',
        'basic': 'Non-Learning UAV',
        'orca': 'ORCA UAV'
    }
    ax.set_title(f'{uav_type_name[uav_type]} - Zoomed View\nPosition: ({target_pos.x:.1f}, {target_pos.y:.1f})', 
                fontsize=16, fontweight='bold')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=detection_color, linewidth=3, alpha=detection_alpha, 
                   label=f'Detection Radius ({target_uav.detection_radius}m)'),
        plt.Line2D([0], [0], color=nmac_color, linewidth=3, alpha=nmac_alpha, 
                   label=f'NMAC Radius ({target_uav.nmac_radius}m)'),
        plt.Line2D([0], [0], color='black', linewidth=2.5, label='Current Heading (Black)'),
        plt.Line2D([0], [0], color=ref_color, linewidth=2, alpha=0.6, label='Reference Heading'),
        plt.Line2D([0], [0], color=trajectory_color, linewidth=2.5, alpha=0.6, label='Trajectory'),
        plt.Line2D([0], [0], color=start_end_color, linestyle='--', alpha=0.6, linewidth=2, label='Start-End Line'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Annotations
    ax.text(0.02, 0.98, f'UAV ID: {target_uav.id}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.text(0.02, 0.94, f'Speed: {target_uav.current_speed:.2f} m/s', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.text(0.02, 0.90, f'Heading: {math.degrees(target_uav.current_heading):.1f}°', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.text(0.02, 0.86, f'Distance to Goal: {target_pos.distance(target_uav.end):.1f}m', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Show trajectory length if exists
    if hasattr(env.renderer, 'trajectory_by_id') and target_uav.id in env.renderer.trajectory_by_id:
        traj_len = len(env.renderer.trajectory_by_id[target_uav.id])
        ax.text(0.02, 0.82, f'Trajectory Points: {traj_len}', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {uav_type} UAV render to: {save_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("UAV Rendering Verification Test")
    print("=" * 60)
    print("\nThis test creates renders at two stages:")
    print("1. Initial state (at reset)")
    print("2. After movement (with trajectories)")
    print("\nFor each stage, creates:")
    print("- Full map view showing all UAVs")
    print("- Zoomed view of Learning Agent")
    print("- Zoomed view of Basic Non-Learning UAV")
    print("- Zoomed view of ORCA UAV")
    print("=" * 60)
    print()
    
    test_uav_rendering()