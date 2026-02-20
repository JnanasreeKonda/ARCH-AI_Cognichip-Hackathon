# Implementation Details

## Overview

This document describes the detailed implementation of ARCH-AI, including code structure, algorithms, and design decisions.

## Main Optimization Loop

### Entry Point: `main.py`

The main optimization loop orchestrates the entire process:

```python
# 1. Initialize
history = []
best_design = None
best_objective = float('inf')

# 2. Iteration Loop
for i in range(ITERATIONS):
    # Propose design
    params = propose_design(history)
    
    # Generate RTL
    rtl = generate_rtl(params)
    
    # Synthesize
    area, log, metrics = synthesize(rtl)
    
    # Calculate objective
    objective = calculate_objective(params, metrics)
    
    # Update history
    history.append((params, metrics))
    
    # Track best
    if objective < best_objective:
        best_design = (params, metrics)
        best_objective = objective

# 3. Generate Reports
generate_all_reports(history, best_design)
```

## LLM Agent Implementation

### Design Agent Class

The `DesignAgent` class in `llm/llm_agent.py` implements the intelligent design proposal system.

#### Initialization

```python
class DesignAgent:
    def __init__(self, mode='auto'):
        # Auto-detect available LLM
        if mode == 'auto':
            if OPENAI_API_KEY and OPENAI_AVAILABLE:
                self.mode = 'openai'
            elif ANTHROPIC_API_KEY and ANTHROPIC_AVAILABLE:
                self.mode = 'anthropic'
            else:
                self.mode = 'heuristic'
```

#### Design Proposal Process

1. **Format History**: Convert exploration history to text format for LLM
2. **Build Prompt**: Create detailed prompt with context
3. **Call LLM**: Send request to selected LLM provider
4. **Parse Response**: Extract JSON parameters
5. **Validate**: Ensure parameters are in valid search space
6. **Fallback**: Use heuristic if LLM fails

#### Prompt Engineering

The LLM receives a structured prompt:

```
You are a hardware design optimization expert.

SEARCH SPACE:
- PAR: [1, 2, 4, 8, 16, 32]
- BUFFER_DEPTH: [256, 512, 1024, 2048]

OBJECTIVE: Minimize AEP = total_cells + 0.5 × (total_cells / throughput)

EXPLORATION HISTORY:
[Formatted table of previous designs]

INSTRUCTIONS:
1. Analyze patterns
2. Identify promising regions
3. Balance exploration vs exploitation
4. Propose next PAR and BUFFER_DEPTH

Respond with JSON: {"PAR": <value>, "BUFFER_DEPTH": <value>}
```

## RTL Generation

### Dynamic Verilog Generation

RTL is generated on-the-fly based on proposed parameters:

```python
def generate_rtl(par, buffer_depth):
    addr_width = int(math.ceil(math.log2(buffer_depth)))
    
    rtl = f"""
    module reduce_sum #(
        parameter PAR = {par},
        parameter BUFFER_DEPTH = {buffer_depth}
    ) (
        input clk, rst,
        input [31:0] in_data,
        input in_valid,
        output reg [31:0] out_data,
        output reg out_valid
    );
    // ... implementation
    endmodule
    """
    return rtl
```

### Design Template

The `reduce_sum` module implements:
- Parallel accumulation (PAR accumulators)
- Buffer-based processing (BUFFER_DEPTH elements)
- Valid/ready handshaking
- Reset functionality

## Synthesis Integration

### Yosys Synthesis

The `tools/run_yosys.py` module handles hardware synthesis:

```python
def synthesize(verilog_file, debug=False):
    # Check Yosys availability
    if not yosys_available():
        return estimate_metrics(verilog_file)
    
    # Run Yosys synthesis
    yosys_script = """
    read_verilog {file}
    synth -top reduce_sum
    stat
    """
    
    # Parse output for metrics
    metrics = parse_yosys_output(output)
    return metrics
```

### Metrics Extraction

Key metrics extracted:
- **Total Cells**: Total gate count
- **Flip-Flops**: Sequential elements
- **Logic Cells**: Combinational logic
- **Wires**: Interconnect count

### Fallback Estimation

When Yosys is unavailable, metrics are estimated:

```python
def estimate_metrics(par, buffer_depth):
    addr_width = ceil(log2(buffer_depth))
    
    # Estimated based on design parameters
    flip_flops = par * 32 + addr_width + 1
    logic_cells = par * 50 + buffer_depth // 4
    total_cells = flip_flops + logic_cells
    
    return {
        'total_cells': total_cells,
        'flip_flops': flip_flops,
        'logic_cells': logic_cells
    }
```

## Objective Function

### Calculation

```python
def calculate_objective(params, metrics):
    par = params["PAR"]
    total_cells = metrics.get('total_cells', float('inf'))
    throughput = par
    
    # Base objective: Area-Efficiency Product
    area_efficiency = total_cells / throughput
    objective = total_cells + 0.5 * area_efficiency
    
    # Apply constraint penalties
    penalty = 0
    if total_cells > MAX_AREA_CELLS:
        penalty += CONSTRAINT_PENALTY
    if throughput < MIN_THROUGHPUT:
        penalty += CONSTRAINT_PENALTY
    if metrics.get('flip_flops', 0) > MAX_FLIP_FLOPS:
        penalty += CONSTRAINT_PENALTY
    
    return objective + penalty
```

### Optimization Goal

Minimize AEP while satisfying constraints:
- Lower area (total_cells) is better
- Higher throughput (PAR) is better
- Better efficiency (cells/throughput) is better

## Constraint Handling

### Soft Constraints with Penalties

Constraints are enforced using penalty functions:

```python
CONSTRAINT_PENALTY = 10000

if constraint_violated:
    objective += CONSTRAINT_PENALTY
```

This ensures:
- Constraint-violating designs are heavily penalized
- Optimization naturally avoids invalid designs
- Best design always meets all constraints

## Reporting System

### Report Generation Pipeline

```python
def generate_all_reports(history, best_design):
    # 1. Export data
    export_to_json(history, best_design)
    export_to_csv(history)
    
    # 2. Generate visualizations
    generate_visualizations(history, best_design)
    generate_3d_design_space(history)
    generate_pareto_frontier(history)
    
    # 3. Statistical analysis
    generate_statistical_report(history)
    
    # 4. Comprehensive dashboard
    generate_comprehensive_dashboard(history, best_design)
```

### Dashboard Components

The comprehensive dashboard includes:
- Best design metrics
- Top 3 designs comparison
- Statistics summary
- Pareto frontier plot
- Optimization progress
- Design space exploration
- Timing analysis
- Area efficiency
- Hardware resources
- Power estimation
- Key insights

## Error Handling

### Graceful Degradation

The system handles errors gracefully:

1. **LLM Unavailable**: Falls back to heuristic search
2. **Yosys Unavailable**: Uses estimated metrics
3. **Synthesis Failure**: Skips iteration, continues
4. **Simulation Failure**: Continues without simulation data

### Robustness Features

- Try-except blocks around critical operations
- Validation of LLM responses
- Fallback mechanisms for all external tools
- Clear error messages and warnings

## Performance Optimizations

### Efficient History Management

- Only last 10 designs sent to LLM (reduces token usage)
- Efficient data structures for history tracking
- Fast objective calculation

### Parallel Processing Opportunities

Future enhancements could include:
- Parallel synthesis of multiple designs
- Batch LLM requests
- Concurrent simulation runs

## Code Quality

### Standards Compliance

- **PEP 8**: Python style guide compliance
- **Type Hints**: Function signatures with types
- **Docstrings**: Comprehensive documentation
- **Modular Design**: Clean separation of concerns

### Testing Considerations

Key areas for testing:
- LLM response parsing
- Objective function calculation
- Constraint validation
- Report generation
- Error handling paths
