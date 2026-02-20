# API Reference

## Core Functions

### `propose_design(history)`

Proposes next design parameters based on exploration history.

**Location**: `llm/llm_agent.py`

**Parameters**:
- `history` (List[Tuple[Dict, Dict]]): List of (params, metrics) tuples

**Returns**:
- `Dict[str, int]`: Dictionary with "PAR" and "BUFFER_DEPTH" keys

**Example**:
```python
from llm.llm_agent import propose_design

history = [
    ({"PAR": 2, "BUFFER_DEPTH": 512}, {"total_cells": 295, ...}),
    ...
]
params = propose_design(history)
# Returns: {"PAR": 4, "BUFFER_DEPTH": 1024}
```

---

### `synthesize(verilog_file, debug=False)`

Synthesizes Verilog design and extracts hardware metrics.

**Location**: `tools/run_yosys.py`

**Parameters**:
- `verilog_file` (str): Path to Verilog file
- `debug` (bool): Print detailed Yosys output

**Returns**:
- `Tuple[Optional[int], str, Dict]`: (area, log, metrics)
  - `area`: Total cell count (None if synthesis failed)
  - `log`: Yosys output log
  - `metrics`: Dictionary with hardware metrics

**Metrics Dictionary**:
```python
{
    "total_cells": int,
    "flip_flops": int,
    "logic_cells": int,
    "wires": int
}
```

**Example**:
```python
from tools.run_yosys import synthesize

area, log, metrics = synthesize("rtl/tmp.v")
if area is not None:
    print(f"Total cells: {area}")
    print(f"Flip-flops: {metrics['flip_flops']}")
```

---

### `simulate(rtl_file, params, simulator='auto')`

Runs functional simulation of design.

**Location**: `tools/simulate.py`

**Parameters**:
- `rtl_file` (str): Path to RTL file
- `params` (Dict): Design parameters {"PAR": int, "BUFFER_DEPTH": int}
- `simulator` (str): 'auto', 'icarus', or 'verilator'

**Returns**:
- `Tuple[bool, Dict, str]`: (success, metrics, log)
  - `success`: Boolean indicating simulation success
  - `metrics`: Dictionary with simulation results
  - `log`: Simulation output log

**Metrics Dictionary**:
```python
{
    "cycle_count": int,
    "throughput": float,
    "max_frequency_mhz": float
}
```

**Example**:
```python
from tools.simulate import simulate

params = {"PAR": 4, "BUFFER_DEPTH": 1024}
success, metrics, log = simulate("rtl/tmp.v", params)
if success:
    print(f"Throughput: {metrics['throughput']} ops/cycle")
```

---

### `calculate_objective(params, metrics)`

Calculates optimization objective function.

**Location**: `main.py`

**Parameters**:
- `params` (Dict): Design parameters
- `metrics` (Dict): Hardware metrics

**Returns**:
- `float`: Objective value (lower is better)

**Formula**:
```
AEP = total_cells + 0.5 × (total_cells / throughput)
+ constraint_penalties
```

**Example**:
```python
params = {"PAR": 2, "BUFFER_DEPTH": 512}
metrics = {"total_cells": 295, "flip_flops": 75, ...}
objective = calculate_objective(params, metrics)
# Returns: 368.8
```

---

### `generate_all_reports(history, best_design)`

Generates all reports and visualizations.

**Location**: `tools/results_reporter.py`

**Parameters**:
- `history` (List[Tuple[Dict, Dict]]): Full exploration history
- `best_design` (Tuple[Dict, Dict]): Best design (params, metrics)

**Returns**:
- `List[str]`: List of generated file paths

**Example**:
```python
from tools.results_reporter import generate_all_reports

history = [...]
best_design = (best_params, best_metrics)
files = generate_all_reports(history, best_design)
print(f"Generated {len(files)} files")
```

---

## Classes

### `DesignAgent`

LLM-powered design space exploration agent.

**Location**: `llm/llm_agent.py`

#### Constructor

```python
DesignAgent(mode='auto')
```

**Parameters**:
- `mode` (str): 'openai', 'anthropic', 'gemini', 'heuristic', or 'auto'

**Example**:
```python
from llm.llm_agent import DesignAgent

agent = DesignAgent(mode='openai')
```

#### Methods

##### `propose_design(history)`

Proposes next design parameters.

**Parameters**:
- `history` (List[Tuple[Dict, Dict]]): Exploration history

**Returns**:
- `Dict[str, int]`: Proposed parameters

**Example**:
```python
agent = DesignAgent()
params = agent.propose_design(history)
```

---

## Utility Functions

### `generate_comprehensive_dashboard(history, best_design, filename)`

Generates comprehensive all-in-one dashboard.

**Location**: `tools/dashboard.py`

**Parameters**:
- `history` (List[Tuple[Dict, Dict]]): Exploration history
- `best_design` (Tuple[Dict, Dict]): Best design
- `filename` (str): Output file path (default: "results/comprehensive_dashboard.png")

**Returns**:
- `str`: Path to generated dashboard file

**Example**:
```python
from tools.dashboard import generate_comprehensive_dashboard

dashboard_path = generate_comprehensive_dashboard(history, best_design)
```

---

### `find_pareto_optimal(history, metric1, metric2, ...)`

Finds Pareto-optimal designs.

**Location**: `tools/pareto_analysis.py`

**Parameters**:
- `history` (List[Tuple[Dict, Dict]]): Exploration history
- `metric1` (str): First metric name
- `metric2` (str): Second metric name
- Additional keyword arguments for optimization direction

**Returns**:
- `List[Tuple[Dict, Dict]]`: Pareto-optimal designs

**Example**:
```python
from tools.pareto_analysis import find_pareto_optimal

pareto = find_pareto_optimal(
    history,
    'total_cells',
    'throughput',
    minimize1=True,
    maximize2=True
)
```

---

### `calculate_statistics(history)`

Calculates statistical metrics.

**Location**: `tools/statistical_analysis.py`

**Parameters**:
- `history` (List[Tuple[Dict, Dict]]): Exploration history

**Returns**:
- `Dict`: Dictionary with statistical metrics

**Metrics**:
```python
{
    "objective": {
        "min": float,
        "max": float,
        "mean": float,
        "std": float,
        "median": float
    },
    "area": {...},
    "improvement_rate": float,
    "stability": float,
    "coverage": float
}
```

**Example**:
```python
from tools.statistical_analysis import calculate_statistics

stats = calculate_statistics(history)
print(f"Mean objective: {stats['objective']['mean']}")
```

---

## Data Structures

### History Format

```python
history: List[Tuple[Dict, Dict]]

# Each entry:
(params: Dict, metrics: Dict)

# params:
{
    "PAR": int,           # Parallelism: {1, 2, 4, 8, 16, 32}
    "BUFFER_DEPTH": int   # Buffer depth: {256, 512, 1024, 2048}
}

# metrics:
{
    "total_cells": int,
    "flip_flops": int,
    "logic_cells": int,
    "wires": int,
    "throughput": float,
    "area_per_throughput": float,
    "objective": float,
    "constraints_violated": bool,
    "max_frequency_mhz": float,
    "critical_path_delay_ns": float
}
```

### Best Design Format

```python
best_design: Tuple[Dict, Dict]
# Same format as history entry
# (best_params, best_metrics)
```

---

## Constants

### Design Constraints

**Location**: `main.py`

```python
MAX_AREA_CELLS = 1500      # Maximum total cells
MIN_THROUGHPUT = 2         # Minimum ops/cycle
MAX_FLIP_FLOPS = 400       # Maximum flip-flops
CONSTRAINT_PENALTY = 10000 # Penalty for violations
```

### Search Space

**Location**: `llm/llm_agent.py`

```python
PAR_OPTIONS = [1, 2, 4, 8, 16, 32]
BUFFER_DEPTH_OPTIONS = [256, 512, 1024, 2048]
```

---

## Error Handling

### Common Exceptions

1. **LLM API Errors**: Fallback to heuristic search
2. **Synthesis Failures**: Return None for area, continue
3. **Simulation Failures**: Return success=False, continue
4. **File I/O Errors**: Print warning, skip operation

### Error Handling Pattern

```python
try:
    result = operation()
except Exception as e:
    print(f"Warning: {operation} failed: {e}")
    # Fallback or continue
```

---

## Type Hints

All functions include type hints for better IDE support:

```python
from typing import List, Tuple, Dict, Optional

def function(
    param1: Dict[str, int],
    param2: List[Tuple[Dict, Dict]]
) -> Tuple[Optional[int], str, Dict]:
    ...
```

---

## Examples

### Complete Optimization Loop

```python
from llm.llm_agent import propose_design
from tools.run_yosys import synthesize
from tools.results_reporter import generate_all_reports
import math

history = []
ITERATIONS = 5

for i in range(ITERATIONS):
    # Propose design
    params = propose_design(history)
    
    # Generate RTL
    par = params["PAR"]
    buffer_depth = params["BUFFER_DEPTH"]
    addr_width = int(math.ceil(math.log2(buffer_depth)))
    
    rtl = f"""
    module reduce_sum #(
        parameter PAR = {par},
        parameter BUFFER_DEPTH = {buffer_depth}
    ) (
        // ... implementation
    );
    endmodule
    """
    
    with open("rtl/tmp.v", "w") as f:
        f.write(rtl)
    
    # Synthesize
    area, log, metrics = synthesize("rtl/tmp.v")
    
    if area is not None:
        # Calculate objective
        from main import calculate_objective
        objective = calculate_objective(params, metrics)
        metrics['objective'] = objective
        
        # Update history
        history.append((params, metrics))

# Find best
best_design = min(history, key=lambda x: x[1].get('objective', float('inf')))

# Generate reports
generate_all_reports(history, best_design)
```
