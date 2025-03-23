import zlib  # For Kolmogorov complexity approximation
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform


@dataclass
class MetricsConfig:
    """Configuration for information theory metrics calculations.
    
    All parameters have sensible defaults but can be customized as needed.
    """
    # Neighborhood settings
    neighborhood_type: str = "moore"  # One of: "moore", "von_neumann", "custom"
    neighborhood_size: int = 1  # Radius of neighborhood
    custom_neighborhood: Optional[List[Tuple[int, int]]] = None  # For custom neighborhood patterns
    
    # Mutual Information settings
    mi_histogram_bins: int = 20  # Number of bins for MI calculation
    mi_use_channels: List[int] = field(default_factory=lambda: [0, 1, 2])  # RGB by default
    
    # Kolmogorov Complexity settings
    kc_compression_algo: str = "zlib"  # One of: "zlib", "custom"
    kc_custom_compressor: Optional[callable] = None  # Custom compression function
    
    # Information Flow settings
    if_time_window: int = 5  # Number of previous states to consider
    if_decay_factor: float = 0.8  # Weight decay for older states
    
    # General settings
    normalize_output: bool = True  # Whether to normalize outputs to 0-1 range
    active_threshold: float = 0.0  # Threshold for considering a cell active

def normalize_values(values: Dict[str, float]) -> Dict[str, float]:
    """Normalize dictionary values to 0-1 range."""
    if not values:
        return values
        
    min_val = min(values.values())
    max_val = max(values.values())
    
    if max_val == min_val:
        return {k: 0.0 for k in values}
        
    return {k: (v - min_val) / (max_val - min_val) for k, v in values.items()}

def get_neighborhood(grid: np.ndarray, row: int, col: int, config: MetricsConfig) -> np.ndarray:
    """Get neighborhood for a given cell based on config.
    
    Args:
        grid: 3D numpy array of shape (height, width, channels)
        row: Row index of center cell
        col: Column index of center cell
        config: MetricsConfig object
        
    Returns:
        Array of neighboring cell values
    """
    height, width = grid.shape[:2]
    neighbors = []
    
    if config.neighborhood_type == "custom" and config.custom_neighborhood:
        for dr, dc in config.custom_neighborhood:
            r, c = row + dr, col + dc
            if 0 <= r < height and 0 <= c < width:
                neighbors.append(grid[r, c])
    
    elif config.neighborhood_type == "von_neumann":
        # Von Neumann neighborhood (4 adjacent cells)
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            r, c = row + dr, col + dc
            if 0 <= r < height and 0 <= c < width:
                neighbors.append(grid[r, c])
    
    else:  # Default to Moore
        size = config.neighborhood_size
        for i in range(max(0, row-size), min(height, row+size+1)):
            for j in range(max(0, col-size), min(width, col+size+1)):
                if (i, j) != (row, col):
                    neighbors.append(grid[i, j])
                
    return np.array(neighbors)

def calculate_mutual_information(X: np.ndarray, Y: np.ndarray, config: MetricsConfig) -> float:
    """Calculate mutual information between two variables.
    
    Args:
        X: First variable's values
        Y: Second variable's values
        config: MetricsConfig object
        
    Returns:
        Mutual information value
    """
    # Use only specified channels and active cells
    X = X[X[:, 3] > config.active_threshold][:, config.mi_use_channels]
    Y = Y[Y[:, 3] > config.active_threshold][:, config.mi_use_channels]
    
    if len(X) == 0 or len(Y) == 0:
        return 0.0
        
    # Calculate joint histogram
    hist_2d, x_edges, y_edges = np.histogram2d(
        X.ravel(), Y.ravel(), 
        bins=config.mi_histogram_bins,
        density=True
    )
    
    # Calculate marginal distributions
    px = np.sum(hist_2d, axis=1)
    py = np.sum(hist_2d, axis=0)
    
    # Calculate mutual information
    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if hist_2d[i,j] > 0 and px[i] > 0 and py[j] > 0:
                mi += hist_2d[i,j] * np.log2(hist_2d[i,j] / (px[i] * py[j]))
                
    return max(0.0, mi)

def calculate_kc(grid_data: np.ndarray, config: MetricsConfig = MetricsConfig()) -> Dict[str, float]:
    """Calculate Kolmogorov complexity approximation.
    
    Args:
        grid_data: 3D numpy array of shape (height, width, channels)
        config: Optional MetricsConfig object
        
    Returns:
        Dictionary mapping cell_ids to their KC values
    """
    active_cells = get_active_cells(grid_data, config.active_threshold)
    kc_values = {}
    
    for row, col in active_cells:
        # Get neighborhood including the cell itself
        neighborhood = get_neighborhood(grid_data, row, col, config)
        
        if config.kc_compression_algo == "custom" and config.kc_custom_compressor:
            ratio = config.kc_custom_compressor(neighborhood)
        else:
            # Default zlib compression
            data = neighborhood.tobytes()
            compressed = zlib.compress(data)
            ratio = len(compressed) / len(data)
        
        kc_values[f"cell_{row}_{col}"] = ratio
    
    return normalize_values(kc_values) if config.normalize_output else kc_values

def calculate_mi(grid_data: np.ndarray, config: MetricsConfig = MetricsConfig()) -> Dict[str, float]:
    """Calculate Multi-information between cells.
    
    Args:
        grid_data: 3D numpy array of shape (height, width, channels)
        config: Optional MetricsConfig object
        
    Returns:
        Dictionary mapping cell_ids to their MI values
    """
    active_cells = get_active_cells(grid_data, config.active_threshold)
    mi_values = {}
    
    for row, col in active_cells:
        # Get cell's neighborhood
        neighborhood = get_neighborhood(grid_data, row, col, config)
        cell_value = grid_data[row, col]
        
        if len(neighborhood) == 0:
            mi_values[f"cell_{row}_{col}"] = 0.0
            continue
            
        # Calculate MI between cell and its neighborhood
        mi = calculate_mutual_information(
            np.array([cell_value] * len(neighborhood)),
            neighborhood,
            config
        )
        
        mi_values[f"cell_{row}_{col}"] = mi
    
    return normalize_values(mi_values) if config.normalize_output else mi_values

def get_active_cells(grid_data: np.ndarray, threshold: float = 0.0) -> List[Tuple[int, int]]:
    """Get coordinates of active cells.
    
    Args:
        grid_data: 3D numpy array of shape (height, width, channels)
        threshold: Activity threshold on alpha channel
        
    Returns:
        List of (row, col) tuples for active cells
    """
    active = np.where(grid_data[:, :, 3] > threshold)
    return list(zip(active[0], active[1]))

def calculate_si(grid_data: np.ndarray, config: MetricsConfig = MetricsConfig()) -> Dict[str, float]:
    """Calculate Synergistic information between cells.
    
    SI measures information that arises only from the collective interaction of multiple variables,
    beyond what individual variables provide. Uses partial information decomposition approach.
    
    Args:
        grid_data: 3D numpy array of shape (height, width, channels)
        config: Optional MetricsConfig object
        
    Returns:
        Dictionary mapping cell_ids to their SI values
    """
    active_cells = get_active_cells(grid_data, config.active_threshold)
    si_values = {}
    
    for row, col in active_cells:
        # Get cell's neighborhood
        neighborhood = get_neighborhood(grid_data, row, col, config)
        cell_value = grid_data[row, col]
        
        if len(neighborhood) < 2:  # Need at least 2 neighbors for synergy
            si_values[f"cell_{row}_{col}"] = 0.0
            continue
        
        # Calculate total mutual information between cell and all neighbors together
        total_mi = calculate_mutual_information(
            np.array([cell_value] * len(neighborhood)),
            neighborhood,
            config
        )
        
        # Calculate individual mutual information contributions
        individual_mis = []
        for neighbor in neighborhood:
            mi = calculate_mutual_information(
                np.array([cell_value]),
                np.array([neighbor]),
                config
            )
            individual_mis.append(mi)
        
        # Synergistic information is the information that can only be obtained
        # by considering the neighbors together, beyond their individual contributions
        si = total_mi - sum(individual_mis)
        si_values[f"cell_{row}_{col}"] = max(0.0, si)  # Ensure non-negative
    
    return normalize_values(si_values) if config.normalize_output else si_values

def calculate_if(grid_data: np.ndarray, prev_states: Optional[List[np.ndarray]] = None,
                config: MetricsConfig = MetricsConfig()) -> Dict[str, float]:
    """Calculate Total Information Flow between cells over time.
    
    IF measures how information propagates through the grid over time by calculating
    the weighted mutual information between current cell states and their past states.
    The weights decay exponentially with time to give more importance to recent states.
    
    Args:
        grid_data: Current 3D numpy array of shape (height, width, channels)
        prev_states: List of previous grid states, ordered from most to least recent
        config: Optional MetricsConfig object
        
    Returns:
        Dictionary mapping cell_ids to their IF values
    """
    if not prev_states:
        # No previous states to calculate flow from
        return {f"cell_{i}_{j}": 0.0 for i, j in get_active_cells(grid_data, config.active_threshold)}
    
    # Limit number of previous states to time window
    prev_states = prev_states[:config.if_time_window]
    
    # Calculate weights for each time step
    weights = [config.if_decay_factor ** t for t in range(len(prev_states))]
    # Normalize weights to sum to 1
    weights = np.array(weights) / sum(weights)
    
    if_values = {}
    active_cells = get_active_cells(grid_data, config.active_threshold)
    
    for row, col in active_cells:
        cell_value = grid_data[row, col]
        total_flow = 0.0
        
        # For each previous state
        for t, (prev_state, weight) in enumerate(zip(prev_states, weights)):
            # Get the neighborhood from the previous state
            prev_neighborhood = get_neighborhood(prev_state, row, col, config)
            if len(prev_neighborhood) == 0:
                continue
            
            # Calculate mutual information between current cell and previous neighborhood
            flow = calculate_mutual_information(
                np.array([cell_value] * len(prev_neighborhood)),
                prev_neighborhood,
                config
            )
            
            # Weight the flow by temporal distance
            total_flow += flow * weight
        
        if_values[f"cell_{row}_{col}"] = total_flow
    
    return normalize_values(if_values) if config.normalize_output else if_values

def calculate_gi(grid_data: np.ndarray, config: MetricsConfig = MetricsConfig()) -> Dict[str, float]:
    """Calculate Geometric Integrated Information."""
    raise NotImplementedError() 