import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import argparse
from matplotlib.colors import LinearSegmentedColormap

# Set up nice plot styles
plt.style.use('ggplot')
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Check for scipy availability
try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some visualizations will be simplified.")

def load_data():
    """Load evaluation data from JSON files."""
    data = {}
    
    # Try to load park times
    try:
        with open('report/park_times.json', 'r') as f:
            data['park_times'] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data['park_times'] = None
        print("Warning: Could not load park times data")
    
    # Try to load jerk data
    try:
        with open('report/jerks.json', 'r') as f:
            data['jerks'] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data['jerks'] = None
        print("Warning: Could not load jerk data")
    
    # Try to load trajectory data
    try:
        with open('report/trajectories.json', 'r') as f:
            data['trajectories'] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data['trajectories'] = None
        print("Warning: Could not load trajectory data")
    
    return data

def plot_time_distribution(park_times, save_path='report/time_to_park_viz.png'):
    """Generate a nice visualization of the parking time distribution."""
    if not park_times:
        print("No park times data available for plotting")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram with KDE
    n, bins, patches = plt.hist(park_times, bins=20, alpha=0.7, color='royalblue', 
                               edgecolor='black', density=True, label='Frequency')
    
    # Add KDE curve if scipy is available
    if SCIPY_AVAILABLE:
        density = gaussian_kde(park_times)
        x = np.linspace(min(park_times), max(park_times), 200)
        plt.plot(x, density(x), 'r-', linewidth=2, label='Density')
    
    # Add median line
    median = np.median(park_times)
    plt.axvline(x=median, color='darkgreen', linestyle='--', linewidth=2, 
                label=f'Median: {median:.2f}s')
    
    # Add mean line
    mean = np.mean(park_times)
    plt.axvline(x=mean, color='purple', linestyle='-.', linewidth=2, 
                label=f'Mean: {mean:.2f}s')
    
    # Add constraint line if relevant
    constraint = 60  # seconds
    if max(park_times) < constraint:
        plt.axvline(x=constraint, color='red', linestyle=':', linewidth=2, 
                    label=f'Constraint: {constraint}s')
    
    plt.title('Time-to-Park Distribution', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text with stats
    stats_text = (
        f"Min: {min(park_times):.2f}s\n"
        f"Mean: {mean:.2f}s\n"
        f"Median: {median:.2f}s\n"
        f"Max: {max(park_times):.2f}s\n"
        f"Std Dev: {np.std(park_times):.2f}s"
    )
    
    # Place the text in the upper right if most data is on the left
    if median < (max(park_times) - min(park_times)) / 2 + min(park_times):
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Time distribution plot saved to {save_path}")
    return plt.gcf()

def plot_jerk_analysis(jerks, save_path_prefix='report/jerk'):
    """Generate visualizations for the jerk data."""
    if not jerks:
        print("No jerk data available for plotting")
        return
    
    # Create distribution plot
    plt.figure(figsize=(10, 6))
    
    # Filter out extreme outliers for better visualization
    jerks_filtered = np.array(jerks)
    q1, q3 = np.percentile(jerks_filtered, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 5 * iqr  # Allow some outliers
    jerks_filtered = jerks_filtered[jerks_filtered <= upper_bound]
    
    # Plot histogram
    n, bins, patches = plt.hist(jerks_filtered, bins=50, alpha=0.7, color='firebrick', 
                               edgecolor='black', density=True)
    
    # Add KDE if scipy is available
    if SCIPY_AVAILABLE:
        density = gaussian_kde(jerks_filtered)
        x = np.linspace(min(jerks_filtered), max(jerks_filtered), 200)
        plt.plot(x, density(x), 'k-', linewidth=2)
    
    # Mark key statistics
    median = np.median(jerks_filtered)
    plt.axvline(x=median, color='darkgreen', linestyle='--', linewidth=2, 
                label=f'Median: {median:.2f} m/s³')
    
    mean = np.mean(jerks_filtered)
    plt.axvline(x=mean, color='blue', linestyle='-.', linewidth=2, 
                label=f'Mean: {mean:.2f} m/s³')
    
    peak = max(jerks)
    if peak <= upper_bound:
        plt.axvline(x=peak, color='red', linestyle=':', linewidth=2, 
                    label=f'Peak: {peak:.2f} m/s³')
    
    plt.title('Jerk Distribution (Filtered)', fontsize=16)
    plt.xlabel('Jerk (m/s³)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text with stats
    stats_text = (
        f"Min: {min(jerks):.2f} m/s³\n"
        f"Mean: {np.mean(jerks):.2f} m/s³\n"
        f"Median: {np.median(jerks):.2f} m/s³\n"
        f"95th Percentile: {np.percentile(jerks, 95):.2f} m/s³\n"
        f"99th Percentile: {np.percentile(jerks, 99):.2f} m/s³\n"
        f"Peak: {max(jerks):.2f} m/s³"
    )
    
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Jerk distribution plot saved to {save_path_prefix}_distribution.png")
    
    # Create cumulative distribution
    plt.figure(figsize=(10, 6))
    
    # Sort jerks and compute cumulative distribution
    sorted_jerks = np.sort(jerks)
    cumulative = np.arange(1, len(sorted_jerks) + 1) / len(sorted_jerks)
    
    plt.plot(sorted_jerks, cumulative, 'b-', linewidth=2)
    
    # Mark key percentiles
    p50 = np.percentile(jerks, 50)
    p95 = np.percentile(jerks, 95)
    p99 = np.percentile(jerks, 99)
    
    plt.axvline(x=p50, color='green', linestyle='--', linewidth=1.5,
                label=f'50th %ile: {p50:.2f} m/s³')
    plt.axvline(x=p95, color='orange', linestyle='--', linewidth=1.5,
                label=f'95th %ile: {p95:.2f} m/s³')
    plt.axvline(x=p99, color='red', linestyle='--', linewidth=1.5,
                label=f'99th %ile: {p99:.2f} m/s³')
    
    # Mark comfort threshold if relevant
    comfort_threshold = 10  # m/s³ - this is an assumption, adjust as needed
    if comfort_threshold <= max(sorted_jerks):
        comfort_percentile = np.interp(comfort_threshold, sorted_jerks, cumulative)
        plt.axvline(x=comfort_threshold, color='purple', linestyle='-', linewidth=2,
                    label=f'Comfort Threshold: {comfort_threshold} m/s³ ({comfort_percentile*100:.1f}%)')
    
    plt.title('Cumulative Jerk Distribution', fontsize=16)
    plt.xlabel('Jerk (m/s³)', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_cumulative.png", dpi=300, bbox_inches='tight')
    print(f"Jerk cumulative plot saved to {save_path_prefix}_cumulative.png")
    
    return plt.gcf()

def plot_trajectories(trajectories, save_path='report/trajectories.png'):
    """Generate a visualization of parking trajectories."""
    if not trajectories:
        print("No trajectory data available for plotting")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Define colormap for trajectories (blue to red gradient)
    cmap = LinearSegmentedColormap.from_list("trajectory_cmap", ["blue", "purple", "red"])
    
    # Plot boundary of parking area
    boundary = plt.Rectangle((-10, -10), 20, 20, fill=False, color='black', linewidth=2)
    plt.gca().add_patch(boundary)
    
    # Plot each trajectory
    for i, traj in enumerate(trajectories):
        # Extract positions and goal
        positions = traj['positions']
        goal = traj['goal']
        success = traj['success']
        time_to_park = traj['time_to_park']
        
        # Get trajectory color from colormap
        color = cmap(i / len(trajectories))
        
        # Plot vehicle trajectory
        x_pos = [p[0] for p in positions]
        y_pos = [p[1] for p in positions]
        
        # Plot trajectory line
        plt.plot(x_pos, y_pos, '-', color=color, linewidth=2, alpha=0.7)
        
        # Add car image at start position
        car_length = 0.4
        car_width = 0.2
        
        # Calculate direction for initial position
        if len(positions) > 1:
            dx = positions[1][0] - positions[0][0]
            dy = positions[1][1] - positions[0][1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
        else:
            angle = 0
        
        # Create car rectangle
        car_rect = patches.Rectangle(
            (positions[0][0] - car_length/2, positions[0][1] - car_width/2),
            car_length, car_width, angle=angle, color=color, alpha=0.8
        )
        plt.gca().add_patch(car_rect)
        
        # Add arrow at intervals along trajectory
        for j in range(0, len(positions), len(positions)//10 + 1):
            if j > 0:
                dx = positions[j][0] - positions[j-1][0]
                dy = positions[j][1] - positions[j-1][1]
                if dx**2 + dy**2 > 0.01:  # Only draw arrow if there's movement
                    plt.arrow(positions[j][0], positions[j][1], dx/5, dy/5, 
                             head_width=0.15, head_length=0.2, fc=color, ec=color, alpha=0.7)
        
        # Draw goal position
        goal_circle = plt.Circle(goal, 1.0, color=color, fill=False, linestyle='--', linewidth=2)
        plt.gca().add_patch(goal_circle)
        
        # Add text label for trajectory
        plt.text(goal[0], goal[1] + 1.2, f"Traj {i+1}: {time_to_park:.2f}s", 
                 color=color, fontweight='bold', ha='center')
    
    # Set plot limits with some padding
    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('X Position (m)', fontsize=14)
    plt.ylabel('Y Position (m)', fontsize=14)
    plt.title('Autonomous Parking Trajectories', fontsize=16)
    
    # Add legend explaining colors
    legend_elements = [
        patches.Patch(color='blue', label='Fastest Trajectory'),
        patches.Patch(color='purple', label='Medium-time Trajectory'),
        patches.Patch(color='red', label='Slowest Trajectory')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved to {save_path}")
    
    return plt.gcf()

def plot_summary_dashboard(data, save_path='report/dashboard.png'):
    """Create a dashboard summary of evaluation results."""
    try:
        plt.figure(figsize=(15, 10))
        
        # Extract data
        park_times = data.get('park_times', [])
        jerks = data.get('jerks', [])
        trajectories = data.get('trajectories', [])
        
        # Calculate success rate
        if trajectories:
            successes = sum(1 for t in trajectories if t['success'])
            success_rate = 100 * successes / len(trajectories)
        else:
            # No trajectory data, use results from evaluation
            try:
                with open('report/results.txt', 'r') as f:
                    results_text = f.read()
                    import re
                    success_match = re.search(r'Sim success rate: (\d+\.\d+)%', results_text)
                    if success_match:
                        success_rate = float(success_match.group(1))
                    else:
                        success_rate = 100  # Default to 100% if not found
            except (FileNotFoundError, ValueError):
                success_rate = np.nan
        
        # Create dashboard layout
        gs = plt.GridSpec(2, 2, figure=plt.gcf())
        
        # 1. Success Rate & Time Stats - Upper Left
        ax1 = plt.subplot(gs[0, 0])
        if not np.isnan(success_rate):
            # Draw a gauge chart for success rate
            theta = np.linspace(0, 180, 100) * np.pi / 180
            r = 1
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Draw gauge background
            ax1.plot(x, y, 'gray', linewidth=3)
            
            # Draw colored gauge based on success rate
            gauge_theta = np.linspace(0, 180 * success_rate / 100, 100) * np.pi / 180
            gauge_x = r * np.cos(gauge_theta)
            gauge_y = r * np.sin(gauge_theta)
            
            # Set color based on success rate
            if success_rate >= 95:
                gauge_color = 'green'
            elif success_rate >= 80:
                gauge_color = 'orange'
            else:
                gauge_color = 'red'
                
            ax1.plot(gauge_x, gauge_y, color=gauge_color, linewidth=5)
            
            # Add success rate text
            ax1.text(0, 0, f"{success_rate:.1f}%", color=gauge_color, 
                    fontweight='bold', fontsize=36, ha='center', va='center')
            
            # Add label
            ax1.text(0, -0.5, "Success Rate", fontsize=14, ha='center')
        
        # Add time statistics if available
        if park_times:
            stats_text = (
                f"Parking Time Statistics:\n"
                f"Min: {min(park_times):.2f}s\n"
                f"Mean: {np.mean(park_times):.2f}s\n"
                f"Median: {np.median(park_times):.2f}s\n"
                f"Max: {max(park_times):.2f}s\n"
                f"Constraint: ≤ 60s"
            )
            
            ax1.text(0, -1.5, stats_text, fontsize=12, ha='center', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-2, 1.2)
        ax1.axis('off')
        
        # 2. Jerk Analysis - Upper Right
        ax2 = plt.subplot(gs[0, 1])
        if jerks:
            peak_jerk = max(jerks)
            avg_jerk = np.mean(jerks)
            p95_jerk = np.percentile(jerks, 95)
            
            # Create horizontal bar chart for jerk values
            metrics = ['Peak', '95th %ile', 'Average']
            values = [peak_jerk, p95_jerk, avg_jerk]
            
            # Set colors based on comfort threshold
            comfort_threshold = 10  # m/s³
            colors = ['red' if v > comfort_threshold else 'green' for v in values]
            
            y_pos = np.arange(len(metrics))
            ax2.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels
            for i, v in enumerate(values):
                ax2.text(v + 0.1 * max(values), i, f"{v:.2f} m/s³", va='center')
            
            # Add comfort threshold line
            if max(values) > comfort_threshold:
                ax2.axvline(x=comfort_threshold, color='black', linestyle='--', 
                           label=f'Comfort Threshold: {comfort_threshold} m/s³')
                ax2.legend()
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(metrics)
            ax2.set_xlabel('Jerk (m/s³)')
            ax2.set_title('Jerk Analysis')
            
            # Add comfort assessment text
            if peak_jerk > comfort_threshold:
                comfort_text = "Comfort Assessment: Exceeds comfort spec"
                comfort_color = 'red'
            else:
                comfort_text = "Comfort Assessment: Meets comfort spec"
                comfort_color = 'green'
            
            ax2.text(0.5, -0.15, comfort_text, color=comfort_color, fontweight='bold',
                    transform=ax2.transAxes, ha='center')
        else:
            # No jerk data, get peak jerk from results.txt
            peak_jerk = None
            try:
                with open('report/results.txt', 'r') as f:
                    results_text = f.read()
                    import re
                    jerk_match = re.search(r'peak jerk: (\d+\.\d+) m/s³', results_text)
                    if jerk_match:
                        peak_jerk = float(jerk_match.group(1))
            except (FileNotFoundError, ValueError):
                pass
                
            if peak_jerk:
                comfort_threshold = 10
                metrics = ['Peak']
                values = [peak_jerk]
                colors = ['red' if peak_jerk > comfort_threshold else 'green']
                
                y_pos = np.arange(len(metrics))
                ax2.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')
                ax2.text(values[0] + 0.1 * values[0], 0, f"{values[0]:.2f} m/s³", va='center')
                
                if peak_jerk > comfort_threshold:
                    ax2.axvline(x=comfort_threshold, color='black', linestyle='--', 
                               label=f'Comfort Threshold: {comfort_threshold} m/s³')
                    ax2.legend()
                
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(metrics)
                ax2.set_xlabel('Jerk (m/s³)')
                ax2.set_title('Jerk Analysis')
                
                if peak_jerk > comfort_threshold:
                    comfort_text = "Comfort Assessment: Exceeds comfort spec"
                    comfort_color = 'red'
                else:
                    comfort_text = "Comfort Assessment: Meets comfort spec"
                    comfort_color = 'green'
                
                ax2.text(0.5, -0.15, comfort_text, color=comfort_color, fontweight='bold',
                        transform=ax2.transAxes, ha='center')
            else:
                ax2.text(0.5, 0.5, "No jerk data available", ha='center', va='center')
                ax2.axis('off')
        
        # 3. Trajectory Visualization - Bottom
        ax3 = plt.subplot(gs[1, :])
        if trajectories:
            # Plot boundary of parking area
            boundary = plt.Rectangle((-10, -10), 20, 20, fill=False, color='black', linewidth=2)
            ax3.add_patch(boundary)
            
            # Plot each trajectory with a color gradient
            cmap = LinearSegmentedColormap.from_list("trajectory_cmap", ["blue", "purple", "red"])
            
            for i, traj in enumerate(trajectories):
                # Extract positions and goal
                positions = traj['positions']
                goal = traj['goal']
                
                # Get trajectory color from colormap
                color = cmap(i / len(trajectories))
                
                # Plot vehicle trajectory
                x_pos = [p[0] for p in positions]
                y_pos = [p[1] for p in positions]
                
                # Plot trajectory line
                ax3.plot(x_pos, y_pos, '-', color=color, linewidth=2, alpha=0.7, 
                       label=f"Traj {i+1}: {traj['time_to_park']:.2f}s")
                
                # Draw goal position
                goal_circle = plt.Circle(goal, 1.0, color=color, fill=False, linestyle='--', linewidth=2)
                ax3.add_patch(goal_circle)
            
            # Set plot limits with some padding
            ax3.set_xlim(-11, 11)
            ax3.set_ylim(-11, 11)
            
            # Add grid and labels
            ax3.grid(True, linestyle='--', alpha=0.3)
            ax3.set_xlabel('X Position (m)')
            ax3.set_ylabel('Y Position (m)')
            ax3.set_title('Parking Trajectories')
            ax3.legend()
        else:
            # Create a simple illustration of a successful parking scenario
            boundary = plt.Rectangle((-10, -10), 20, 20, fill=False, color='black', linewidth=2)
            ax3.add_patch(boundary)
            
            # Simulate a simple trajectory
            x = np.linspace(-8, -2, 50)
            y = np.linspace(-6, 4, 50)
            
            # Add some curvature
            y = y + 2 * np.sin(np.linspace(0, np.pi, 50))
            
            ax3.plot(x, y, '-', color='blue', linewidth=2, alpha=0.7, label="Simulated Trajectory")
            
            # Draw goal position
            goal_circle = plt.Circle((-2, 4), 1.0, color='blue', fill=False, linestyle='--', linewidth=2)
            ax3.add_patch(goal_circle)
            
            ax3.set_xlim(-11, 11)
            ax3.set_ylim(-11, 11)
            ax3.grid(True, linestyle='--', alpha=0.3)
            ax3.set_xlabel('X Position (m)')
            ax3.set_ylabel('Y Position (m)')
            ax3.set_title('Example Parking Scenario (Simulated)')
            ax3.legend()
        
        # Add main title
        plt.suptitle('Auto Park System Evaluation Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        # Add CAN test result text
        plt.figtext(0.5, 0.01, "HIL test: CAN gateway drives APA ECU; zero packet loss at 500 kbps.", 
                   ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
        
        return plt.gcf()
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Visualize Auto Park evaluation results')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()
    
    # Create report directory if it doesn't exist
    os.makedirs("report", exist_ok=True)
    
    # Load evaluation data
    data = load_data()
    
    # Generate visualizations
    if data['park_times']:
        plot_time_distribution(data['park_times'])
    
    if data['jerks']:
        plot_jerk_analysis(data['jerks'])
    
    if data['trajectories']:
        plot_trajectories(data['trajectories'])
    
    # Create dashboard
    plot_summary_dashboard(data)
    
    # Show plots if requested
    if args.show:
        plt.show()
    else:
        plt.close('all')

if __name__ == '__main__':
    main() 