# NeuroMonster 2025

A neuronal computing framework that lets you "paint" living neural networks and experiment with free energy principles through an accessible interface.

## What is this?

A PyTorch-inspired toolkit for neuronal computing with a focus on:
1. Interactive neuron painting (simulates Multi-Electrode Arrays)
2. Real-time visualization of neural activity
3. Testing active inference and free energy principles
4. Video reconstruction tasks using neuromorphic approaches

## Install

```bash
uv venv
uv pip install -e .
```

## Tech Stack

- **Python**: Core library and simulation engine
- **JAX**: For hardware acceleration (way faster than raw PyTorch for our use case)
- **NumPy**: Numerical operations
- **DearPyGui**: Desktop UI for neuron painting interface
- **Plotly/Dash**: Interactive visualizations 

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