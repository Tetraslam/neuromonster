# NeuroMonster 2025

A neuronal computing framework that lets you "paint" living neural networks and experiment with free energy principles through an accessible interface.

## What is this?

A PyTorch-inspired toolkit for neuronal computing with a focus on:
1. Interactive neuron painting (simulates Multi-Electrode Arrays)
2. Real-time visualization of neural activity
3. Testing active inference and free energy principles
4. Video reconstruction tasks using neuromorphic approaches

## Install

### Prerequisites
- Python 3.11 or higher
- For GPU acceleration: CUDA 12.6 (download from https://developer.nvidia.com/cuda-12-6-3-download-archive)

### Installation Steps

1. Create and activate a virtual environment:
```bash
uv venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

2. Install dependencies with PyTorch:

For CPU-only version:
```bash
uv pip install -r pyproject.toml --extra cpu
```

For CUDA 12.6 version (requires NVIDIA GPU with CUDA 12.6 installed):
```bash
uv pip install -r pyproject.toml --extra cu126
```

That's it! The correct version of PyTorch will be installed automatically based on your choice.

## Tech Stack

- **Python**: Core library and simulation engine
- **PyTorch**: For hardware acceleration
- **NumPy**: Numerical operations
- **DearPyGui**: Desktop UI for neuron painting interface and visualizations

## Core Components

### NeuroMonster
```python
from neuromonster import NeuroPaint

canvas = NeuroMonster(width=100, height=100)
# Paint a cluster of Izhikevich neurons
canvas.paint(x=50, y=50, radius=10, model="izhikevich")
# Run simulation
canvas.simulate(timesteps=1000)
# Visualize activity
canvas.visualize()
```

### Active Inference Module
```python
from neuromonster.models import FreeEnergyAgent

agent = FreeEnergyAgent()
agent.learn_from_damaged_video("input.mp4", damage_rate=0.99)
agent.reconstruct("output.mp4")
```

## Usage

### Information Theory Metrics
```python
from neuromonster.metrics.information_theory import MetricsConfig, calculate_mi, calculate_si, calculate_if, calculate_gi, calculate_kc

# Using default configuration
grid_data = canvas.get_grid()  # Your 3D grid data (height, width, RGBA)
mi_values = calculate_mi(grid_data)  # Returns {cell_id: value} dict

# Custom configuration
config = MetricsConfig(
    # Neighborhood settings
    neighborhood_type="von_neumann",  # "moore", "von_neumann", or "custom"
    neighborhood_size=2,  # Radius of neighborhood
    
    # Channel settings
    mi_use_channels=[0, 1],  # Only use R,G channels
    mi_histogram_bins=30,  # More bins for MI calculation
    
    # Information Flow settings
    if_time_window=10,  # Consider last 10 states
    if_decay_factor=0.9,  # Slower decay of old information
    
    # Geometric Integration settings
    gi_clustering_method="hierarchical",
    gi_min_cluster_size=4,
    
    # General settings
    normalize_output=True,  # Scale outputs to 0-1
    active_threshold=0.1  # Ignore cells with alpha < 0.1
)

# Calculate all metrics with custom config
metrics = {
    "mi": calculate_mi(grid_data, config),
    "si": calculate_si(grid_data, config),
    "kc": calculate_kc(grid_data, config),
    "gi": calculate_gi(grid_data, config),
    "if": calculate_if(grid_data, previous_states, config)
}

# Access metric values for specific cells
cell_id = "cell_5_7"  # Format: cell_row_col
mi_value = metrics["mi"][cell_id]  # Multi-information
si_value = metrics["si"][cell_id]  # Synergistic information
kc_value = metrics["kc"][cell_id]  # Kolmogorov complexity
gi_value = metrics["gi"][cell_id]  # Geometric integration
if_value = metrics["if"][cell_id]  # Information flow
```

Each metric quantifies different aspects of the system:
- **MI**: Shared information between cells and their neighborhoods
- **SI**: Information that emerges only from collective interactions
- **KC**: Complexity/randomness of cell patterns
- **GI**: System's capacity to integrate information as clusters
- **IF**: How information propagates through the grid over time

All metrics are normalized to [0,1] by default for easy visualization and comparison.

## Project Structure

```
neuromonster/
├── README.md
├── core/            # Core spiking neuron models
├── paint/           # MEA painting interface
├── models/          # FEP and morphological computing
├── metrics/         # Information theory benchmarks
├── examples/        # Demo applications
└── viz/             # Visualization toolkit
```

## Roadmap

1. Core painting interface with basic neuron models
2. FEP implementation
3. Video processing pipeline
4. Information theory benchmarking suite