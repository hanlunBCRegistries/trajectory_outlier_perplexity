# Python
import folium
import h3
import ast
import os
import glob
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm
import numpy as np
import json

# Create a folium map centered on Porto
m = folium.Map(location=[41.163, -8.619], zoom_start=13)

# Limit the number of trajectories to visualize
max_trajectories = 30

# Create feature groups for normal and outlier trajectories
normal_group = folium.FeatureGroup(name="Normal Trajectories")
outlier_group = folium.FeatureGroup(name="Outlier Trajectories")
m.add_child(normal_group)
m.add_child(outlier_group)

# Function to get the geo boundaries of an H3 cell
def get_hexagon_coordinates(h3_index):
    """Convert H3 index to polygon coordinates"""
    try:
        # For H3 4.x
        boundaries = h3.cell_to_boundary(h3_index)
        return [(lat, lng) for lat, lng in boundaries]
    except Exception as e:
        try:
            # For H3 3.x
            boundaries = h3.h3_to_geo_boundary(h3_index)
            return [(lat, lng) for lat, lng in boundaries]
        except:
            # Fallback
            lat, lng = h3.cell_to_latlng(h3_index)
            return [(lat+0.0001, lng+0.0001), (lat+0.0001, lng-0.0001), 
                    (lat-0.0001, lng-0.0001), (lat-0.0001, lng+0.0001)]

# Load all outlier trajectories into a set for comparison
print("Loading outlier trajectories...")
outlier_trajectories = set()
outlier_files = glob.glob("./data/porto/outliers/*.csv")
for file_path in outlier_files:
    outlier_type = os.path.basename(file_path).split('_')[0]  # 'route_switch' or 'detour'
    with open(file_path) as f:
        for line in f:
            try:
                traj = tuple(ast.literal_eval(line.strip()))
                outlier_trajectories.add(traj)
            except Exception as e:
                print(f"Error parsing outlier in {file_path}: {e}")
                continue

print(f"Loaded {len(outlier_trajectories)} outlier trajectories")

# Read the processed trajectories from the csv file
print("Loading normal trajectories...")
trajectories = []
with open("./data/porto/porto_processed.csv") as f:
    for i, line in enumerate(f):
        if i >= max_trajectories:
            break
        try:
            traj = ast.literal_eval(line.strip())
            trajectories.append(traj)
        except Exception as e:
            print(f"Error parsing line {i}: {e}")
            continue

print(f"Loaded {len(trajectories)} normal trajectories")

# Create a feature group for each trajectory
for i, traj in enumerate(trajectories):
    # Check if this trajectory is an outlier
    is_outlier = tuple(traj) in outlier_trajectories
    
    # Choose color based on outlier status
    if is_outlier:
        color = "#ff0000"  # Red for outliers
        group = outlier_group
        status = "OUTLIER"
    else:
        color = "#3388ff"  # Blue for normal trajectories
        group = normal_group
        status = "Normal"
        
    points = [h3.cell_to_latlng(hex_token) for hex_token in traj]
    
    # Create a unique ID for this trajectory
    traj_id = f"traj_{i}"
    
    # Draw the trajectory line
    polyline = folium.PolyLine(
        locations=points,
        color=color,
        weight=3,
        opacity=0.7,
        tooltip=f"{status} Trajectory {i+1}",
        popup=f"{status} Trajectory {i+1}",
    )
    
    # Add a unique identifier to the polyline
    polyline._id = traj_id
    group.add_child(polyline)
    
    # Draw hexagons for each point
    for j, hex_index in enumerate(traj):
        # Get the hexagon coordinates
        hex_coords = get_hexagon_coordinates(hex_index)
        
        # Draw hexagon with appropriate fill color but lower opacity
        folium.Polygon(
            locations=hex_coords,
            color=color,
            weight=1,
            opacity=0.7 if j == 0 or j == len(traj)-1 else 0.4,  # More visible for start/end
            fill=True,
            fill_color=color,
            fill_opacity=0.3 if j == 0 or j == len(traj)-1 else 0.1,  # More visible for start/end
            tooltip=f"{status} Trajectory {i+1}, Point {j+1}",
        ).add_to(group)
    
    # Add start and end markers
    folium.Marker(
        location=points[0],
        icon=folium.Icon(color='green', icon='play'),
        tooltip=f"Start of {status} trajectory {i+1}"
    ).add_to(group)
    
    folium.Marker(
        location=points[-1],
        icon=folium.Icon(color='red', icon='stop'),
        tooltip=f"End of {status} trajectory {i+1}"
    ).add_to(group)

# Add JavaScript for click interaction
click_script = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Add a reset button to restore all trajectories
    var resetButton = document.createElement('button');
    resetButton.innerHTML = 'Reset View';
    resetButton.style.position = 'absolute';
    resetButton.style.top = '10px';
    resetButton.style.right = '10px';
    resetButton.style.zIndex = '1000';
    resetButton.style.padding = '8px 12px';
    resetButton.style.backgroundColor = '#fff';
    resetButton.style.border = '2px solid rgba(0,0,0,0.2)';
    resetButton.style.borderRadius = '4px';
    resetButton.style.display = 'none';
    document.body.appendChild(resetButton);
    
    var polylines = document.querySelectorAll('path.leaflet-interactive');
    var markers = document.querySelectorAll('.leaflet-marker-icon');
    var activeTrajectory = null;
    var highlightMode = false;
    
    function resetAllElements() {
        // Reset all polylines to original state
        polylines.forEach(function(p) {
            p.setAttribute('stroke-width', '3');
            p.setAttribute('stroke-opacity', '0.7');
        });
        
        // Show all markers
        markers.forEach(function(m) {
            m.style.opacity = '1';
        });
        
        highlightMode = false;
        resetButton.style.display = 'none';
    }
    
    resetButton.addEventListener('click', resetAllElements);
    
    polylines.forEach(function(polyline, index) {
        polyline.setAttribute('data-index', index);
        polyline.addEventListener('click', function(e) {
            
            if (highlightMode && activeTrajectory === this) {
                // If clicking the already selected trajectory, reset view
                resetAllElements();
            } else {
                // Enter highlight mode
                highlightMode = true;
                activeTrajectory = this;
                resetButton.style.display = 'block';
                
                // Get the index of the current trajectory
                var currentIndex = this.getAttribute('data-index');
                
                // Dim all polylines
                polylines.forEach(function(p) {
                    p.setAttribute('stroke-width', '1');
                    p.setAttribute('stroke-opacity', '0.15');
                });
                
                // Hide all markers except for the ones of this trajectory
                markers.forEach(function(m, i) {
                    // If marker is the start or end of current trajectory (markers appear in pairs)
                    if (Math.floor(i/2) === parseInt(currentIndex)) {
                        m.style.opacity = '1';
                        m.style.zIndex = '1000';
                    } else {
                        m.style.opacity = '0.15';
                        m.style.zIndex = '500';
                    }
                });
                
                // Highlight clicked polyline
                this.setAttribute('stroke-width', '6');
                this.setAttribute('stroke-opacity', '1');
            }
            
            // Prevent the click from propagating to the map
            L.DomEvent.stopPropagation(e);
        });
    });
    
    // Add click on map to reset
    document.querySelector('.leaflet-map-pane').addEventListener('click', function() {
        if (highlightMode) {
            resetAllElements();
        }
    });
});
</script>
"""

# Add the JavaScript to the map
m.get_root().html.add_child(folium.Element(click_script))

# Add a legend
legend_html = '''
<div style="position: fixed; 
    bottom: 50px; left: 50px; width: 150px; height: 90px; 
    border:2px solid grey; z-index:9999; font-size:14px;
    background-color:white; padding: 10px">
    <p><i style="background:#3388ff;width:10px;height:10px;display:inline-block"></i> Normal Trajectory</p>
    <p><i style="background:#ff0000;width:10px;height:10px;display:inline-block"></i> Outlier Trajectory</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Add layer control
folium.LayerControl().add_to(m)

# Save the map as an HTML file
m.save("trajectory_map.html")
print("Created trajectory_map.html")