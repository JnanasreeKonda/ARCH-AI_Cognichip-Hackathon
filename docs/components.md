# Component Documentation

## Core Components

### 1. Main Optimization Loop (`main.py`)

**Purpose**: Orchestrates the entire optimization process

**Key Functions**:
- `calculate_objective(params, metrics)`: Computes optimization objective
- `safe_print(text)`: Handles Unicode encoding for Windows console

**Responsibilities**:
- Initialize optimization environment
- Run iteration loop
- Coordinate LLM agent, synthesis, and simulation
- Track best design
- Generate final reports

### 2. LLM Agent (`llm/llm_agent.py`)

**Purpose**: Intelligent design space exploration using LLMs

**Class**: `DesignAgent`

**Key Methods**:
- `__init__(mode='auto')`: Initialize agent with LLM selection
- `propose_design(history)`: Propose next design parameters
- `_propose_openai(history)`: Use OpenAI GPT-4
- `_propose_anthropic(history)`: Use Anthropic Claude
- `_propose_gemini(history)`: Use Google Gemini
- `_propose_heuristic(history)`: Fallback heuristic search
- `_format_history(history)`: Format history for LLM prompt
- `_validate_params(params)`: Validate proposed parameters

**Search Strategy**:
- Initial exploration: Diverse starting points
- Exploitation: Explore neighbors of best design (70%)
- Exploration: Random untried combinations (30%)

### 3. Synthesis Module (`tools/run_yosys.py`)

**Purpose**: Hardware synthesis and metrics extraction

**Key Functions**:
- `synthesize(verilog_file, debug=False)`: Synthesize design and extract metrics

**Returns**:
- `area`: Total cell count (or None if failed)
- `log`: Yosys output log
- `metrics`: Dictionary with hardware metrics

**Metrics Extracted**:
- `total_cells`: Total gate count
- `flip_flops`: Sequential elements
- `logic_cells`: Combinational logic
- `wires`: Interconnect count

**Fallback**: Estimates metrics when Yosys unavailable

### 4. Simulation Module (`tools/simulate.py`)

**Purpose**: Functional verification and timing analysis

**Key Functions**:
- `simulate(rtl_file, params, simulator='auto')`: Run functional simulation
- `generate_testbench(par, buffer_depth, output_path)`: Generate testbench
- `simulate_icarus(rtl_file, tb_file)`: Use Icarus Verilog
- `simulate_verilator(rtl_file, tb_file)`: Use Verilator

**Returns**:
- `success`: Boolean indicating simulation success
- `metrics`: Dictionary with simulation results
- `log`: Simulation output log

**Metrics**:
- `cycle_count`: Number of cycles to complete
- `throughput`: Operations per cycle
- `max_frequency_mhz`: Maximum operating frequency

## Analysis Components

### 5. Results Reporter (`tools/results_reporter.py`)

**Purpose**: Coordinate all report generation

**Key Functions**:
- `generate_all_reports(history, best_design)`: Main entry point
- `export_to_json(history, best_design, filename)`: Export JSON data
- `export_to_csv(history, filename)`: Export CSV data
- `generate_visualizations(history, best_design, filename)`: Create plots
- `generate_report(history, best_design, filename)`: Generate text report

**Outputs**:
- JSON/CSV data files
- Visualization plots
- Text reports
- Best design RTL

### 6. Dashboard Generator (`tools/dashboard.py`)

**Purpose**: Create comprehensive all-in-one dashboard

**Key Functions**:
- `generate_comprehensive_dashboard(history, best_design, filename)`: Main function

**Dashboard Sections**:
1. **Title & Summary**: Project info, timestamp, best objective
2. **Best Design Metrics**: Complete design details
3. **Top 3 Comparison**: Area, Performance, Balanced designs
4. **Statistics Summary**: Min/max/mean/std metrics
5. **Pareto Frontier**: Trade-off visualization
6. **Optimization Progress**: Convergence plot
7. **Design Space**: PAR vs Area exploration
8. **Timing Analysis**: Frequency trends
9. **Area Efficiency**: Cells per throughput
10. **Hardware Resources**: Cells, FFs, logic cells
11. **Buffer Depth Impact**: Buffer depth analysis
12. **Power Estimation**: Static/dynamic/total power
13. **Key Insights**: Summary and recommendations

### 7. Enhanced Visualizations (`tools/enhanced_visualizations.py`)

**Purpose**: Advanced plotting and analysis

**Key Functions**:
- `generate_3d_design_space(history, filename)`: 3D design space plot
- `generate_statistical_analysis(history, filename)`: Statistical plots
- `generate_power_estimation_plot(history, filename)`: Power analysis

**Visualizations**:
- 3D scatter plots (PAR × Buffer Depth × Objective)
- Statistical distributions
- Power consumption trends
- Exploration vs exploitation balance

### 8. Statistical Analysis (`tools/statistical_analysis.py`)

**Purpose**: Statistical metrics and convergence analysis

**Key Functions**:
- `calculate_statistics(history)`: Compute statistical metrics
- `calculate_improvement_rate(objectives)`: Measure convergence
- `calculate_stability(objectives)`: Measure stability
- `calculate_coverage(history)`: Design space coverage
- `generate_statistical_report(history, filename)`: Generate report

**Metrics**:
- Mean, median, std deviation
- Min/max values
- Improvement rate
- Convergence metrics
- Design space coverage

### 9. Pareto Analysis (`tools/pareto_analysis.py`)

**Purpose**: Identify Pareto-optimal designs

**Key Functions**:
- `find_pareto_optimal(history, metric1, metric2, ...)`: Find Pareto front
- `generate_pareto_frontier_plot(history, filename)`: Visualize Pareto front
- `generate_pareto_report(pareto_optimal, filename)`: Generate report

**Pareto Criteria**:
- Minimize area (total_cells)
- Maximize throughput (PAR)
- Minimize objective (AEP)

### 10. Timing Analysis (`tools/timing_analysis.py`)

**Purpose**: Timing and performance analysis

**Key Functions**:
- `estimate_timing(metrics, params)`: Estimate timing metrics
- `add_timing_to_metrics(history)`: Add timing to history
- `generate_timing_analysis_plot(history, filename)`: Create timing plots

**Timing Metrics**:
- Maximum frequency (MHz)
- Critical path delay (ns)
- Setup/hold times
- Clock period

### 11. Comparison Tools (`tools/comparison_tools.py`)

**Purpose**: Compare designs and export best design

**Key Functions**:
- `compare_heuristic_vs_llm(heuristic_history, llm_history)`: Compare methods
- `export_best_design_verilog(best_params, best_metrics, filename)`: Export RTL

**Comparisons**:
- Heuristic vs LLM performance
- Design variants
- Optimization strategies

### 12. Comparison Table (`tools/comparison_table.py`)

**Purpose**: Generate comparison tables and reports

**Key Functions**:
- `generate_comparison_table(history, filename)`: Create comparison table
- `generate_comparison_report(history, filename)`: Generate report

**Table Contents**:
- Top N designs
- Metrics comparison
- Parameter comparison
- Objective comparison

## Data Structures

### History Format

```python
history: List[Tuple[Dict, Dict]]
# Each entry: (params, metrics)

params: Dict[str, int]
{
    "PAR": int,           # Parallelism
    "BUFFER_DEPTH": int    # Buffer depth
}

metrics: Dict[str, Any]
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

## Component Interactions

```
main.py
  ├─> llm_agent.propose_design()
  ├─> run_yosys.synthesize()
  ├─> simulate.simulate() [optional]
  └─> results_reporter.generate_all_reports()
        ├─> dashboard.generate_comprehensive_dashboard()
        ├─> enhanced_visualizations.*()
        ├─> statistical_analysis.*()
        ├─> pareto_analysis.*()
        ├─> timing_analysis.*()
        └─> comparison_tools.*()
```

## Dependencies

### External Tools
- **Yosys**: Hardware synthesis (optional)
- **Icarus Verilog**: Simulation (optional)
- **Verilator**: Alternative simulator (optional)

### Python Libraries
- **openai**: OpenAI API client
- **anthropic**: Anthropic API client
- **google-genai**: Google Gemini API client
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **scipy**: Statistical functions
- **tqdm**: Progress bars
