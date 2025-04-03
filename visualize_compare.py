import folium
import h3
import ast
import os
import glob
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm
import numpy as np

# Create a folium map centered on Porto
m = folium.Map(location=[41.163, -8.619], zoom_start=13)

# Limit the number of trajectories to visualize
max_trajectories = 30

# Create feature groups for normal and outlier trajectories
normal_group = folium.FeatureGroup(name="Normal Trajectories")
outlier_group = folium.FeatureGroup(name="Outlier Trajectories")
m.add_child(normal_group)
m.add_child(outlier_group)

# Load outlier trajectories
print("Loading outlier trajectories...")
outlier_trajectories = set()
outlier_files = glob.glob("./data/porto/outliers/*.csv")
for file_path in outlier_files:
    try:
        with open(file_path) as f:
            for line in f:
                try:
                    # Convert the line to a list of strings, handling numpy string objects
                    traj = eval(line.strip(), {'np': np})
                    # Convert all elements to regular Python strings
                    traj = [str(x) for x in traj]
                    outlier_trajectories.add(tuple(traj))
                except Exception as e:
                    print(f"Error parsing outlier in {file_path}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

print(f"Loaded {len(outlier_trajectories)} outlier trajectories")

# Read normal trajectories
print("Loading normal trajectories...")
normal_trajectories = []
try:
    with open("./data/porto/porto_processed.csv") as f:
        for i, line in enumerate(f):
            if i >= max_trajectories:
                break
            try:
                traj = eval(line.strip(), {'np': np})
                traj = [str(x) for x in traj]
                normal_trajectories.append(traj)
            except Exception as e:
                print(f"Error parsing normal trajectory line {i}: {e}")
                continue
except Exception as e:
    print(f"Error reading normal trajectories: {e}")

print(f"Loaded {len(normal_trajectories)} normal trajectories")

# Add interactive features with JavaScript
click_script = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Add a reset button
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
    resetButton.style.cursor = 'pointer';
    resetButton.style.display = 'none';
    document.body.appendChild(resetButton);
    
    // Track all interactive elements
    var polylines = document.querySelectorAll('path.leaflet-interactive');
    var markers = document.querySelectorAll('.leaflet-marker-icon');
    var activeTrajectory = null;
    var highlightMode = false;
    
    // Function to reset view
    function resetView() {
        polylines.forEach(function(p) {
            p.style.strokeWidth = '2';
            p.style.strokeOpacity = '0.8';
            p.style.zIndex = '1';
        });
        
        markers.forEach(function(m) {
            m.style.opacity = '1';
            m.style.zIndex = '1000';
        });
        
        highlightMode = false;
        activeTrajectory = null;
        resetButton.style.display = 'none';
    }
    
    // Add click handler to reset button
    resetButton.addEventListener('click', resetView);
    
    // Add click handlers to all trajectories
    polylines.forEach(function(polyline) {
        polyline.addEventListener('click', function(e) {
            if (highlightMode && activeTrajectory === this) {
                // Reset if clicking active trajectory
                resetView();
            } else {
                // Highlight clicked trajectory
                highlightMode = true;
                activeTrajectory = this;
                resetButton.style.display = 'block';
                
                // Dim all trajectories
                polylines.forEach(function(p) {
                    p.style.strokeWidth = '1';
                    p.style.strokeOpacity = '0.2';
                    p.style.zIndex = '1';
                });
                
                // Dim all markers
                markers.forEach(function(m) {
                    m.style.opacity = '0.2';
                    m.style.zIndex = '999';
                });
                
                // Highlight selected trajectory
                this.style.strokeWidth = '4';
                this.style.strokeOpacity = '1.0';
                this.style.zIndex = '1001';
                
                // Find and highlight corresponding markers
                var trajectoryId = this.getAttribute('data-trajectory-id');
                markers.forEach(function(m) {
                    if (m.getAttribute('data-trajectory-id') === trajectoryId) {
                        m.style.opacity = '1';
                        m.style.zIndex = '1002';
                    }
                });
            }
            
            L.DomEvent.stopPropagation(e);
        });
        
        // Add hover effect
        polyline.addEventListener('mouseover', function() {
            if (!highlightMode) {
                this.style.strokeWidth = '3';
            }
        });
        
        polyline.addEventListener('mouseout', function() {
            if (!highlightMode) {
                this.style.strokeWidth = '2';
            }
        });
    });
    
    // Click on map to reset
    document.querySelector('#map').addEventListener('click', function() {
        if (highlightMode) {
            resetView();
        }
    });
});
</script>
"""

# Process trajectories with unique IDs
for i, traj in enumerate(normal_trajectories):
    try:
        points = [h3.cell_to_latlng(hex_token) for hex_token in traj]
        
        # Create unique ID for this trajectory
        traj_id = f"normal_{i}"
        
        # Add trajectory line with data attribute
        line = folium.PolyLine(
            locations=points,
            color='blue',
            weight=2,
            opacity=0.8,
            popup=f"Normal Trajectory {i+1}",
            tooltip=f"Click to highlight Normal Trajectory {i+1}"
        )
        line.add_to(normal_group)
        
        # Add data-trajectory-id attribute
        line._name = f'path{i}'
        line._parent.script.add_child(folium.Element(
            f"""
            document.querySelector("path[class='{line._name}']").setAttribute('data-trajectory-id', '{traj_id}');
            """
        ))
        
        # Add markers with same ID
        start_marker = folium.Marker(
            location=points[0],
            icon=folium.Icon(color='green', icon='info-sign'),
            popup=f"Start {i+1}"
        )
        start_marker.add_to(normal_group)
        start_marker._name = f'marker_start_{i}'
        start_marker._parent.script.add_child(folium.Element(
            f"""
            document.querySelector("img[class='{start_marker._name}']").setAttribute('data-trajectory-id', '{traj_id}');
            """
        ))
        
        end_marker = folium.Marker(
            location=points[-1],
            icon=folium.Icon(color='red', icon='info-sign'),
            popup=f"End {i+1}"
        )
        end_marker.add_to(normal_group)
        end_marker._name = f'marker_end_{i}'
        end_marker._parent.script.add_child(folium.Element(
            f"""
            document.querySelector("img[class='{end_marker._name}']").setAttribute('data-trajectory-id', '{traj_id}');
            """
        ))
        
    except Exception as e:
        print(f"Error processing normal trajectory {i}: {e}")
        continue

# Process outlier trajectories similarly
for i, traj in enumerate(list(outlier_trajectories)[:max_trajectories]):
    try:
        points = [h3.cell_to_latlng(hex_token) for hex_token in traj]
        
        traj_id = f"outlier_{i}"
        
        line = folium.PolyLine(
            locations=points,
            color='red',
            weight=2,
            opacity=0.8,
            popup=f"Outlier Trajectory {i+1}",
            tooltip=f"Click to highlight Outlier Trajectory {i+1}"
        )
        line.add_to(outlier_group)
        line._name = f'path_outlier_{i}'
        line._parent.script.add_child(folium.Element(
            f"""
            document.querySelector("path[class='{line._name}']").setAttribute('data-trajectory-id', '{traj_id}');
            """
        ))
        
        # Add markers with same ID
        start_marker = folium.Marker(
            location=points[0],
            icon=folium.Icon(color='green', icon='info-sign'),
            popup=f"Start {i+1}"
        )
        start_marker.add_to(outlier_group)
        start_marker._name = f'marker_outlier_start_{i}'
        start_marker._parent.script.add_child(folium.Element(
            f"""
            document.querySelector("img[class='{start_marker._name}']").setAttribute('data-trajectory-id', '{traj_id}');
            """
        ))
        
        end_marker = folium.Marker(
            location=points[-1],
            icon=folium.Icon(color='red', icon='info-sign'),
            popup=f"End {i+1}"
        )
        end_marker.add_to(outlier_group)
        end_marker._name = f'marker_outlier_end_{i}'
        end_marker._parent.script.add_child(folium.Element(
            f"""
            document.querySelector("img[class='{end_marker._name}']").setAttribute('data-trajectory-id', '{traj_id}');
            """
        ))
        
    except Exception as e:
        print(f"Error processing outlier trajectory {i}: {e}")
        continue

# Add legend
legend_html = '''
<div style="position: fixed; 
    bottom: 50px; left: 50px; width: 180px; height: 100px; 
    border:2px solid grey; z-index:9999; font-size:14px;
    background-color:white; padding: 10px">
    <p><i style="background:#3388ff;width:10px;height:10px;display:inline-block"></i> Normal Trajectory</p>
    <p><i style="background:#ff0000;width:10px;height:10px;display:inline-block"></i> Outlier Trajectory</p>
    <p><small>Click trajectory to highlight</small></p>
</div>
'''

# Add the JavaScript and legend to the map
m.get_root().html.add_child(folium.Element(click_script))
m.get_root().html.add_child(folium.Element(legend_html))

# Add layer control
folium.LayerControl().add_to(m)

# Save the map
m.save("interactive_trajectory_visualization.html")
print("Created interactive_trajectory_visualization.html")