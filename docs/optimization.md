# Optimization Process

## Overview

The optimization process in ARCH-AI follows an iterative design space exploration approach, using LLM-powered intelligence to guide the search for optimal microarchitecture configurations.

## Optimization Loop

### Iteration Structure

```
┌─────────────────────────────────────┐
│  1. LLM Agent Proposes Design      │
│     - Analyzes history              │
│     - Proposes PAR, BUFFER_DEPTH    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. Generate RTL Code                │
│     - Dynamic Verilog generation     │
│     - Parameterized design          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. Synthesize Design               │
│     - Yosys synthesis               │
│     - Extract hardware metrics      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Simulate (Optional)             │
│     - Functional verification       │
│     - Timing analysis               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. Calculate Objective             │
│     - Area-Efficiency Product       │
│     - Constraint penalties          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. Update History                  │
│     - Store params and metrics      │
│     - Track best design             │
└──────────────┬──────────────────────┘
               │
               ▼
         [Repeat N times]
```

## Design Space

### Parameter Space

The optimization explores a 2D discrete space:

- **PAR (Parallelism)**: {1, 2, 4, 8, 16, 32}
- **BUFFER_DEPTH**: {256, 512, 1024, 2048}

**Total Combinations**: 6 × 4 = 24 unique designs

### Search Strategy

The LLM agent uses a hybrid approach:

1. **Exploration**: Try new, untested combinations
2. **Exploitation**: Refine around promising designs
3. **Balance**: 70% exploitation, 30% exploration

## Objective Function

### Area-Efficiency Product (AEP)

The optimization minimizes:

```
AEP = total_cells + 0.5 × (total_cells / throughput)
```

Where:
- `total_cells`: Hardware area (lower is better)
- `throughput`: Operations per cycle = PAR (higher is better)
- `area_efficiency`: Cells per operation (lower is better)

### Interpretation

- **First Term** (`total_cells`): Direct area cost
- **Second Term** (`0.5 × area_efficiency`): Efficiency penalty
- **Weight (0.5)**: Balances area vs efficiency

### Example Calculation

For PAR=4, total_cells=400:
```
throughput = 4 ops/cycle
area_efficiency = 400 / 4 = 100 cells/op
AEP = 400 + 0.5 × 100 = 450
```

## Constraints

### Hard Constraints

Three constraints are enforced:

1. **Max Area**: `total_cells ≤ 1500`
2. **Min Throughput**: `throughput ≥ 2 ops/cycle`
3. **Max Flip-Flops**: `flip_flops ≤ 400`

### Constraint Penalties

Violations are penalized:

```python
CONSTRAINT_PENALTY = 10000

if total_cells > MAX_AREA_CELLS:
    objective += CONSTRAINT_PENALTY
if throughput < MIN_THROUGHPUT:
    objective += CONSTRAINT_PENALTY
if flip_flops > MAX_FLIP_FLOPS:
    objective += CONSTRAINT_PENALTY
```

**Effect**: Constraint-violating designs have very high objectives and are naturally avoided.

## Convergence

### Convergence Metrics

The optimization tracks:

1. **Best Objective**: Minimum AEP found
2. **Improvement Rate**: Rate of improvement over iterations
3. **Stability**: Consistency of results
4. **Coverage**: Percentage of design space explored

### Convergence Criteria

The optimization continues for a fixed number of iterations:

- **Default**: 5 iterations
- **Configurable**: Set `ITERATIONS` in `main.py`
- **Trade-off**: More iterations = better results but longer runtime

### Typical Convergence Pattern

```
Iteration  Objective  Best So Far
    1        10360       10360
    2         368.8       368.8  ← Significant improvement
    3         450.0       368.8
    4         642.8       368.8
    5         368.8       368.8  ← Converged
```

## History Management

### History Structure

```python
history: List[Tuple[Dict, Dict]]

# Each entry:
(params, metrics)

# params:
{
    "PAR": int,
    "BUFFER_DEPTH": int
}

# metrics:
{
    "total_cells": int,
    "flip_flops": int,
    "logic_cells": int,
    "throughput": float,
    "objective": float,
    ...
}
```

### History Usage

1. **LLM Input**: Last 10 designs sent to LLM
2. **Best Tracking**: Maintain best design separately
3. **Reporting**: Full history used for reports

## Best Design Selection

### Selection Criteria

The best design is selected based on:

1. **Lowest Objective**: Minimum AEP value
2. **Constraint Satisfaction**: Must meet all constraints
3. **Tie-Breaking**: If objectives equal, prefer lower area

### Best Design Tracking

```python
best_design = None
best_objective = float('inf')

for params, metrics in history:
    objective = metrics['objective']
    if objective < best_objective:
        best_design = (params, metrics)
        best_objective = objective
```

## Optimization Strategies

### LLM-Guided Search

**Advantages**:
- Learns from exploration history
- Identifies patterns and trends
- Balances exploration/exploitation intelligently

**Process**:
1. LLM analyzes history
2. Identifies promising regions
3. Proposes next design
4. Validates proposal
5. Falls back to heuristic if invalid

### Heuristic Fallback

**Strategy**:
- Initial exploration: Diverse starting points
- Exploitation: Neighbor search around best
- Exploration: Random untried combinations

**Use Cases**:
- LLM unavailable
- LLM returns invalid parameters
- API errors

## Performance Metrics

### Optimization Quality

Key metrics tracked:

1. **Improvement**: `(worst - best) / worst × 100%`
2. **Convergence Speed**: Iterations to find best
3. **Design Space Coverage**: Unique designs explored
4. **Constraint Satisfaction**: Percentage meeting constraints

### Typical Results

- **Improvement**: 90-95% over worst design
- **Convergence**: Best found in 2-3 iterations
- **Coverage**: 20-50% of design space
- **Constraints**: 100% of final designs meet constraints

## Post-Optimization

### Report Generation

After optimization completes:

1. **Find Best Design**: Select optimal configuration
2. **Generate Reports**: JSON, CSV, text reports
3. **Create Visualizations**: Plots and charts
4. **Statistical Analysis**: Convergence metrics
5. **Export RTL**: Best design Verilog code

### Output Files

Generated in `results/` directory:

- `optimization_results.json`: Machine-readable data
- `optimization_results.csv`: Spreadsheet data
- `optimization_plots.png`: Standard visualizations
- `comprehensive_dashboard.png`: All-in-one dashboard
- `best_design.v`: Optimal design RTL
- Additional analysis files

## Tuning Parameters

### Configurable Parameters

In `main.py`:

```python
ITERATIONS = 5              # Number of iterations
MAX_AREA_CELLS = 1500      # Area constraint
MIN_THROUGHPUT = 2         # Throughput constraint
MAX_FLIP_FLOPS = 400       # Flip-flop constraint
CONSTRAINT_PENALTY = 10000  # Penalty weight
```

### Objective Function Weights

In `calculate_objective()`:

```python
objective = total_cells + 0.5 * area_efficiency
#          ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^
#          weight: 1.0    weight: 0.5
```

Adjust weights to prioritize area vs efficiency.

## Future Enhancements

### Potential Improvements

1. **Adaptive Iterations**: Stop when converged
2. **Multi-Start**: Multiple optimization runs
3. **Parallel Exploration**: Test multiple designs simultaneously
4. **Advanced Objectives**: Multi-objective optimization
5. **Constraint Relaxation**: Soft constraints with weights
6. **Design Pruning**: Remove dominated designs early
