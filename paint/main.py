import dearpygui.dearpygui as dpg
import numpy as np
import threading
import queue
import colorsys
import random
import time

# Scientific data structures
mea_variables = {
    "electrode_grid": [8, 8],  # Number of electrodes in a grid (rows, columns)
    "electrode_pitch": 0.5,  # Distance between electrodes (mm)
    "layout_type": "grid"  # Layout type: "grid" or other configurations
}

stimulation_protocol = {
    "pattern": "Poisson",  # Stimulation pattern type (e.g., "Poisson", "regular")
    "intensity": [1.5, 0.01],  # Stimulation intensity [Voltage (V), Current (A)]
    "region": [[0, 0], [1, 1], [2, 2]],  # Electrode coordinates for stimulation
    "pulse": {  # Pulse parameters
        "frequency": 50,  # Stimulation frequency in Hz (pulses per second)
        "duty_cycle": 0.5,  # Fraction of time active (range: 0.0 to 1.0)
        "burst": {  # Burst mode parameters
            "on": 100,  # Burst ON duration in milliseconds (ms)
            "off": 200,  # Burst OFF duration in milliseconds (ms)
            "align": "start"  # Alignment mode: "start" or "end"
        }
    },
    "protocol": {  # Experimental protocol settings
        "timing": [5, 10, 5],  # Time phases: [baseline (s), stim_on (s), stim_off (s)]
        "repeats": 5  # Number of times the protocol repeats
    },
    "feedback": True  # Enable real-time feedback mechanism
}

neuronal_level_variables = {
    "neuron_model": "leaky integrate and fire",  # Model type for neurons
    "plasticity_rule": "STDP",  # Synaptic plasticity rule (e.g., STDP)
    "network_topology": "random",  # Network connectivity pattern
    "firing_params": {  # Parameters governing neuron firing behavior
        "threshold": -55.0,  # Action potential threshold (mV)
        "refractory_period": 2.0,  # Refractory period duration (ms)
        "homeostatic_regulation": True  # Enable homeostatic regulation
    }
}

interaction_ca_variables = {  # Cellular Automata (CA) interaction rules
    "ca_rules": {
        "neighborhood": "Moore",  # Neighborhood type (e.g., Moore, von Neumann)
        "time_step": 0.1  # Time step for CA updates (seconds)
    },
    "heatmap_enabled": True,  # Enable heatmap visualization
    "pattern_evolution": "spread"  # Evolutionary behavior of activity patterns
}

# Constants
GRID_DISPLAY_SIZE = 400  # Size of grid display in pixels
SIMULATION_SPEED = 0.2   # Simulation speed in seconds

# Application state
class AppState:
    def __init__(self):
        self.grid_data = np.zeros((mea_variables["electrode_grid"][0], mea_variables["electrode_grid"][1], 4), dtype=np.float32)
        self.node_positions = {}
        self.edges = []
        self.metrics = {
            "MI": {},  # Multi-information
            "SI": {},  # Synergistic information
            "GI": {},  # Geometric integrated information
            "IF": {},  # Information flow
            "KC": {}   # Kolmogorov complexity
        }
        self.ui_state = {
            "interaction_mode": "stim",
            "grid_size": mea_variables["electrode_grid"][0],  # it is a square
            "selected_metrics": [
                "Multi information",
                "Synergistic information",
                "Total information flow",
                "Geometric integrated information",
                "Free energy principle"
            ],
            "color": [1.0, 0.0, 0.0, 1.0],  # RGBA format (0-1)
            "simulation_running": False,
            "paused": True,
            "mouse_down": False,
            "last_cell": None,
            "update_grid": False,      # Flag to track grid updates
            "update_network": False    # Flag to track network updates
        }
        # Queue for thread-safe communication
        self.sim_queue = queue.Queue()

# Create application state
app = AppState()

# Functions
def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB values (0-1 range)"""
    return colorsys.hsv_to_rgb(h, s, v)

def initialize_grid(state):
    """Initialize the grid with some random cells"""
    state.grid_data = np.zeros((state.ui_state["grid_size"], state.ui_state["grid_size"], 4), dtype=np.float32)
    
    # Add some random colored cells
    for i in range(state.ui_state["grid_size"]):
        for j in range(state.ui_state["grid_size"]):
            if random.random() < 0.3:  # 30% chance of a cell
                hue = random.random()
                r, g, b = hsv_to_rgb(hue, 0.8, 0.9)
                state.grid_data[i, j] = [r, g, b, 1.0]
    
    # Make sure to update the display after initializing
    update_grid_display(state)

def update_grid_display(state):
    """Update the grid display with current grid_data"""
    dpg.delete_item("grid_drawlist", children_only=True)
    
    cell_size = GRID_DISPLAY_SIZE // state.ui_state["grid_size"]
    
    # Draw grid cells
    for i in range(state.ui_state["grid_size"]):
        for j in range(state.ui_state["grid_size"]):
            # Calculate pixel coordinates (i is row/y, j is column/x)
            x = j * cell_size
            y = i * cell_size
            
            # Only draw cells with non-zero alpha
            if state.grid_data[i, j, 3] > 0:
                # Convert from 0-1 to 0-255 for DPG, ensuring values are in range
                r = max(0, min(1, state.grid_data[i, j, 0]))
                g = max(0, min(1, state.grid_data[i, j, 1]))
                b = max(0, min(1, state.grid_data[i, j, 2]))
                a = max(0, min(1, state.grid_data[i, j, 3]))
                
                color = [int(r*255), int(g*255), int(b*255), int(a*255)]
                
                dpg.draw_rectangle(
                    [x, y], 
                    [x + cell_size, y + cell_size],
                    fill=color,
                    parent="grid_drawlist"
                )
            else:
                # Draw empty cell border
                dpg.draw_rectangle(
                    [x, y], 
                    [x + cell_size, y + cell_size],
                    color=[50, 50, 50, 255],
                    fill=[0, 0, 0, 255],
                    parent="grid_drawlist"
                )

def update_network_diagram(state):
    """Update the network diagram based on calculated metrics"""
    dpg.delete_item("network_drawlist", children_only=True)
    
    # Draw legend
    legend_x = 300
    legend_y = 20
    
    dpg.draw_text([legend_x, legend_y], "Legend", color=[255, 255, 255, 255], parent="network_drawlist")
    
    # Draw legend items
    legend_items = [
        {"text": "high MI", "color": [255, 200, 0, 255], "shape": "circle"},
        {"text": "low MI", "color": [100, 0, 255, 255], "shape": "circle"},
        {"text": "high GI", "color": [255, 255, 255, 255], "shape": "dot", "size": 8},
        {"text": "low GI", "color": [255, 255, 255, 255], "shape": "dot", "size": 3},
        {"text": "low SI", "color": [0, 0, 0, 255], "shape": "dot", "size": 5},
        {"text": "high SI", "color": [0, 0, 0, 255], "shape": "dot", "size": 8},
        {"text": "low IF", "color": [255, 255, 255, 255], "shape": "circle", "outline": True, "size": 10},
        {"text": "high IF", "color": [255, 255, 255, 255], "shape": "circle", "outline": True, "size": 15}
    ]
    
    for i, item in enumerate(legend_items):
        y_offset = legend_y + 20 + i * 20
        
        if item["shape"] == "circle":
            if item.get("outline", False):
                dpg.draw_circle([legend_x + 10, y_offset], item.get("size", 10), 
                               color=item["color"], parent="network_drawlist", fill=[0, 0, 0, 0])
            else:
                dpg.draw_circle([legend_x + 10, y_offset], item.get("size", 10), 
                               fill=item["color"], parent="network_drawlist")
        elif item["shape"] == "dot":
            dpg.draw_circle([legend_x + 10, y_offset], item.get("size", 5), 
                           fill=item["color"], parent="network_drawlist")
        
        dpg.draw_text([legend_x + 25, y_offset - 7], item["text"], 
                     color=[255, 255, 255, 255], parent="network_drawlist")
    
    # Generate some random nodes for demonstration
    nodes = []
    for i in range(15):
        x = random.randint(50, 300)
        y = random.randint(100, 350)
        size = random.randint(8, 15)
        # Randomly choose between yellow and purple for MI
        color = [255, 200, 0, 255] if random.random() > 0.7 else [100, 0, 255, 255]
        nodes.append({"pos": [x, y], "size": size, "color": color})
    
    # Draw edges between some nodes
    for i in range(20):
        if len(nodes) > 1:
            n1 = random.randint(0, len(nodes)-1)
            n2 = random.randint(0, len(nodes)-1)
            if n1 != n2:
                dpg.draw_line(nodes[n1]["pos"], nodes[n2]["pos"], 
                             color=[255, 255, 255, 150], parent="network_drawlist")
    
    # Draw nodes
    for node in nodes:
        dpg.draw_circle(node["pos"], node["size"], fill=node["color"], parent="network_drawlist")

def cell_interaction(state, sender, app_data):
    """Handle mouse interaction with the grid"""
    if not state.ui_state["mouse_down"]:
        return
    
    # Get mouse position relative to the drawlist
    mouse_pos = dpg.get_mouse_pos(local=False)
    drawlist_pos = dpg.get_item_rect_min("grid_drawlist")
    
    # Calculate position relative to the drawlist
    x = mouse_pos[0] - drawlist_pos[0]
    y = mouse_pos[1] - drawlist_pos[1]
    
    # Convert to grid coordinates
    cell_size = GRID_DISPLAY_SIZE // state.ui_state["grid_size"]
    
    # In DearPyGui, (0,0) is at the top-left corner
    # So grid_x increases from left to right, grid_y increases from top to bottom
    grid_x = int(x // cell_size)
    grid_y = int(y // cell_size)
    
    # Ensure we're within bounds
    if 0 <= grid_x < state.ui_state["grid_size"] and 0 <= grid_y < state.ui_state["grid_size"]:
        # Avoid redundant updates for the same cell
        current_cell = (grid_x, grid_y)
        if state.ui_state["last_cell"] == current_cell:
            return
        
        state.ui_state["last_cell"] = current_cell
        
        # Apply interaction based on mode
        if state.ui_state["interaction_mode"] == "stim":
            # Stimulate: add white color only.
            state.grid_data[grid_y, grid_x] = [1.0, 1.0, 1.0, 1.0]
            
        elif state.ui_state["interaction_mode"] == "replenish neurons":
            # Replenish: Add a cell with current selected color
            base_color = state.ui_state["color"]
            
            # Ensure color values are in valid range
            r = max(0.0, min(1.0, base_color[0]))
            g = max(0.0, min(1.0, base_color[1]))
            b = max(0.0, min(1.0, base_color[2]))
            a = max(0.0, min(1.0, base_color[3]))
            
            # Convert RGB to HSV for easier color manipulation
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            
            # Stimulate the center cell and its neighbors with variations of the selected color
            for i in range(max(0, grid_y-1), min(state.ui_state["grid_size"], grid_y+2)):
                for j in range(max(0, grid_x-1), min(state.ui_state["grid_size"], grid_x+2)):
                    # Vary the hue slightly for each cell
                    hue_variation = h + random.uniform(-0.1, 0.1)
                    # Keep hue in valid range [0, 1]
                    hue_variation = hue_variation % 1.0
                    
                    # Convert back to RGB
                    r_new, g_new, b_new = colorsys.hsv_to_rgb(hue_variation, s, v)
                    
                    # Set the cell color with full alpha
                    state.grid_data[i, j] = [r_new, g_new, b_new, 1.0]
            
            # Set the center cell to the exact selected color
            state.grid_data[grid_y, grid_x] = [r, g, b, a]
            
        elif state.ui_state["interaction_mode"] == "video":
            # Video mode: placeholder for video reconstruction
            pass
        
        # Update the display after modifying the grid
        update_grid_display(state)

def update_cellular_automata(state):
    """Update the cellular automata grid based on rules"""
    new_grid = np.copy(state.grid_data)
    
    for i in range(state.ui_state["grid_size"]):
        for j in range(state.ui_state["grid_size"]):
            # Count live neighbors (cells with alpha > 0)
            live_neighbors = 0
            neighbor_colors = []
            
            for ni in range(max(0, i-1), min(state.ui_state["grid_size"], i+2)):
                for nj in range(max(0, j-1), min(state.ui_state["grid_size"], j+2)):
                    if (ni, nj) != (i, j) and state.grid_data[ni, nj, 3] > 0:
                        live_neighbors += 1
                        neighbor_colors.append(state.grid_data[ni, nj, :3])
            
            # Apply rules
            if state.grid_data[i, j, 3] > 0:  # Cell is alive
                if live_neighbors < 2 or live_neighbors > 3:
                    # Die from loneliness or overcrowding
                    new_grid[i, j, 3] = 0
            else:  # Cell is dead
                if live_neighbors == 3:
                    # Reproduction - average the colors of neighbors
                    if neighbor_colors:
                        avg_color = np.mean(neighbor_colors, axis=0)
                        new_grid[i, j, :3] = avg_color
                        new_grid[i, j, 3] = 1.0
    
    return new_grid

def run_simulation(state):
    """Simulation logic in a background thread"""
    while True:
        try:
            # Check for commands from the main thread
            command = state.sim_queue.get(block=False)
            if command == "stop":
                break
        except queue.Empty:
            pass
        
        # Skip simulation steps if paused
        if state.ui_state["paused"]:
            time.sleep(0.1)
            continue
        
        # Update the cellular automata based on rules
        state.grid_data = update_cellular_automata(state)
        
        # Signal the main thread to update the display
        state.ui_state["update_grid"] = True
        state.ui_state["update_network"] = True
        
        # Calculate network metrics and update the network diagram
        calculate_network_data(state)
        
        time.sleep(SIMULATION_SPEED)

def calculate_network_data(state):
    """Calculate network metrics for visualization"""
    # This would be where your actual metrics calculation happens
    # For now, we'll just use random values for demonstration
    
    # Update metrics dictionary with random values
    cells = []
    for i in range(state.ui_state["grid_size"]):
        for j in range(state.ui_state["grid_size"]):
            if state.grid_data[i, j, 3] > 0:  # If cell is alive
                cell_id = f"cell_{i}_{j}"
                cells.append(cell_id)
                
                # Generate random metrics
                state.metrics["MI"][cell_id] = random.random()
                state.metrics["SI"][cell_id] = random.random()
                state.metrics["GI"][cell_id] = random.random()
                state.metrics["IF"][cell_id] = random.random()
                state.metrics["KC"][cell_id] = random.random()
    
    # Generate random connections between cells
    state.edges = []
    for _ in range(len(cells) * 2):
        if len(cells) > 1:
            c1 = random.choice(cells)
            c2 = random.choice(cells)
            if c1 != c2:
                state.edges.append((c1, c2, random.random()))

def on_mouse_down():
    """Handle mouse down event"""
    app.ui_state["mouse_down"] = True
    app.ui_state["last_cell"] = None
    cell_interaction(app, None, None)

def on_mouse_up():
    """Handle mouse up event"""
    app.ui_state["mouse_down"] = False
    app.ui_state["last_cell"] = None

def on_color_change(sender, app_data):
    """Handle color picker change"""
    # Check if app_data is already in 0-1 range or 0-255 range
    r, g, b, a = app_data
    
    # If values are already in 0-1 range (DearPyGui sometimes returns values in this range)
    if all(0 <= val <= 1 for val in [r, g, b, a]):
        # Use values directly
        normalized_color = [r, g, b, a]
        # For display in the color indicator, convert to 0-255
        display_color = [int(r*255), int(g*255), int(b*255), int(a*255)]
    else:
        # Ensure values are within range [0, 255] before normalizing
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        a = max(0, min(255, a))
        
        # Normalize to 0-1 range
        normalized_color = [r/255, g/255, b/255, a/255]
        display_color = app_data
    
    # Print color values for debugging
    print(f"Original color values: {app_data}")
    print(f"Normalized color: {normalized_color}")
    
    # Update the color in the UI state
    app.ui_state["color"] = normalized_color
    
    # Update the color indicator if it exists
    try:
        dpg.configure_item("current_color_indicator", fill=display_color)
    except:
        # The color indicator might not be created yet
        pass
    
    # Update the grid display to reflect the new color
    # This is useful when cells are added with the new color
    app.ui_state["update_grid"] = True

def on_mode_change(sender, app_data):
    """Handle interaction mode change"""
    modes = ["stim", "replenish neurons", "video"]
    app.ui_state["interaction_mode"] = app_data

def on_metric_change(sender, app_data):
    """Handle metric checkbox change"""
    metric = dpg.get_item_label(sender)
    if app_data:  # Checked
        if metric not in app.ui_state["selected_metrics"]:
            app.ui_state["selected_metrics"].append(metric)
    else:  # Unchecked
        if metric in app.ui_state["selected_metrics"]:
            app.ui_state["selected_metrics"].remove(metric)

def on_play_pause():
    """Toggle simulation play/pause"""
    app.ui_state["paused"] = not app.ui_state["paused"]
    if app.ui_state["paused"]:
        dpg.set_item_label("play_pause_button", "Play")
    else:
        dpg.set_item_label("play_pause_button", "Pause")

def on_reset():
    """Reset the simulation"""
    initialize_grid(app)

def check_updates():
    """Check if updates are needed and update the grid display"""
    if app.ui_state["update_grid"]:
        update_grid_display(app)
        app.ui_state["update_grid"] = False
    
    if app.ui_state["update_network"]:
        update_network_diagram(app)
        app.ui_state["update_network"] = False

# Dear PyGui setup
dpg.create_context()
dpg.create_viewport(title="Paint MEA!", width=1200, height=800)
dpg.setup_dearpygui()

# Add mouse handlers
with dpg.handler_registry():
    dpg.add_mouse_down_handler(callback=on_mouse_down)
    dpg.add_mouse_release_handler(callback=on_mouse_up)

# Create a primary window
with dpg.window(label="Paint MEA!", width=1000, height=1000, tag="primary_window"):
    # Create a horizontal group to contain both panels
    with dpg.group(horizontal=True):
        # Left Panel (Cellular Automata)
        with dpg.child_window(width=500, height=1000):
            with dpg.group(horizontal=True):
                dpg.add_text("Neural cellular automata")
                
                # Add control buttons on the right side
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Play", callback=on_play_pause, id="play_pause_button", width=60, height=30)
                    dpg.add_button(label="Reset", callback=on_reset, width=60, height=30)
            
            # Grid display
            with dpg.drawlist(width=GRID_DISPLAY_SIZE, height=GRID_DISPLAY_SIZE, tag="grid_drawlist"):
                # Will be populated by update_grid_display
                pass
            
            # Help text
            with dpg.group():
                dpg.add_text("click and hold over the grid to influence neurons", color=[255, 200, 0, 255], wrap=400)
                
                # Information box
                with dpg.child_window(width=400, height=80, border=False):
                    dpg.add_text("MI quantifies the shared information among multiple variables in the", wrap=400)
                    dpg.add_text("system, capturing dependencies between cells.", wrap=400)

        # Right Panel (Controls and Network Diagram)
        with dpg.child_window(width=450, height=1000):
            # Controls section
            with dpg.group(horizontal=True):
                # Left side - Interaction Mode
                with dpg.group():
                    dpg.add_text("Interaction Mode", color=[255, 255, 100, 255], wrap=200)
                    
                    # Add radio buttons
                    dpg.add_radio_button(
                        items=["stim", "replenish neurons", "video"], 
                        horizontal=False,
                        callback=on_mode_change,
                        default_value="stim",
                        tag="interaction_mode_radio"
                    )
                    
                
                # Right side - Color Picker
                with dpg.group():
                    
                    dpg.add_text("Cell Color", color=[255, 255, 100, 255], wrap=200)
                    dpg.add_color_picker(
                        default_value=[255, 0, 0, 255],
                        callback=on_color_change,
                        width=180,
                        height=150
                    )
        
            
            dpg.add_spacer(height=5)
            dpg.add_text(f"Grid Size: {app.ui_state['grid_size']}×{app.ui_state['grid_size']}", wrap=430)
            dpg.add_slider_int(
                label="Simulation speed",
                default_value=3,
                min_value=1,
                max_value=10,
                width=200
            )
            
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # Metrics section
            dpg.add_text("Metrics", color=[255, 255, 100, 255], wrap=430)
            dpg.add_checkbox(label="Komolgorov complexity", callback=on_metric_change)
            dpg.add_checkbox(label="Multi information", default_value=True, callback=on_metric_change)
            dpg.add_checkbox(label="Synergistic information", default_value=True, callback=on_metric_change)
            dpg.add_checkbox(label="Total information flow", default_value=True, callback=on_metric_change)
            dpg.add_checkbox(label="Geometric integrated information", default_value=True, callback=on_metric_change)
            dpg.add_checkbox(label="Free Energy Principle", callback=on_metric_change)
            
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            # Network diagram
            dpg.add_text("Network Diagram", color=[255, 255, 100, 255], wrap=400)
            with dpg.drawlist(width=400, height=400, tag="network_drawlist"):
                # Will be populated by update_network_diagram
                pass

# Initialize grid and update displays
initialize_grid(app)
update_grid_display(app)
update_network_diagram(app)

# Set the primary window
dpg.set_primary_window("primary_window", True)

# Start simulation thread
def start_simulation(state):
    """Start the simulation thread"""
    if not state.ui_state["simulation_running"]:
        state.ui_state["simulation_running"] = True
        sim_thread = threading.Thread(target=run_simulation, args=(state,), daemon=True)
        sim_thread.start()

start_simulation(app)

dpg.show_viewport()

# Main loop with manual update checks
while dpg.is_dearpygui_running():
    # Check for updates
    check_updates()
    # Render frame
    dpg.render_dearpygui_frame()

dpg.destroy_context()