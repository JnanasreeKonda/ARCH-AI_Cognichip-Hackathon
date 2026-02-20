# Architecture Overview

## System Architecture

ARCH-AI is an AI-powered hardware optimization framework that uses Large Language Models (LLMs) to intelligently explore microarchitecture design spaces and find optimal configurations.

```
┌─────────────────────────────────────────────────────────────┐
│                    ARCH-AI System                            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                       │
        ▼                     ▼                       ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  LLM Agent   │      │   Synthesis  │      │  Simulation  │
│              │      │              │      │              │
│ - OpenAI     │─────▶│ - Yosys      │─────▶│ - Icarus     │
│ - Claude     │      │ - Metrics    │      │ - Verilator  │
│ - Gemini     │      │ - Analysis   │      │ - Timing     │
│ - Heuristic  │      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                       │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Optimization   │
                    │     Loop        │
                    │                 │
                    │ - Objective     │
                    │ - Constraints   │
                    │ - History       │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    Reporting     │
                    │                 │
                    │ - Dashboard     │
                    │ - Visualizations│
                    │ - Statistics    │
                    │ - Pareto        │
                    └─────────────────┘
```

## Design Principles

### 1. Modular Architecture
- **Separation of Concerns**: Each component has a single, well-defined responsibility
- **LLM Agent**: Handles design space exploration
- **Synthesis Tools**: Extract hardware metrics
- **Reporting System**: Generate comprehensive outputs

### 2. Extensibility
- **Plugin-based LLM Support**: Easy to add new LLM providers
- **Flexible Objective Functions**: Customizable optimization goals
- **Modular Reporting**: Add new visualization types easily

### 3. Robustness
- **Graceful Degradation**: Falls back to heuristic search if LLM unavailable
- **Error Handling**: Continues optimization even if individual iterations fail
- **Estimated Metrics**: Works without Yosys using estimated synthesis results

### 4. Industry Standards
- **PEP 8 Compliance**: Clean, readable Python code
- **Type Hints**: Better code documentation and IDE support
- **Comprehensive Documentation**: Well-documented functions and classes

## Data Flow

```
1. Initialize
   └─> Load API keys, set up environment
   
2. Optimization Loop (N iterations)
   ├─> LLM Agent proposes design parameters
   ├─> Generate RTL code dynamically
   ├─> Synthesize with Yosys (or estimate)
   ├─> Simulate (optional)
   ├─> Calculate objective function
   ├─> Check constraints
   └─> Update history
   
3. Post-Processing
   ├─> Find best design
   ├─> Generate reports
   ├─> Create visualizations
   └─> Export results
```

## Key Components

### Core Modules

1. **`main.py`**: Main optimization loop and orchestration
2. **`llm/llm_agent.py`**: LLM-powered design agent
3. **`tools/run_yosys.py`**: Hardware synthesis integration
4. **`tools/simulate.py`**: Functional simulation
5. **`tools/results_reporter.py`**: Report generation coordinator

### Analysis Modules

1. **`tools/dashboard.py`**: Comprehensive dashboard generation
2. **`tools/enhanced_visualizations.py`**: Advanced plotting
3. **`tools/statistical_analysis.py`**: Statistical metrics
4. **`tools/pareto_analysis.py`**: Pareto frontier analysis
5. **`tools/timing_analysis.py`**: Timing analysis
6. **`tools/comparison_tools.py`**: Design comparison utilities

## Design Space

The optimization explores a 2D parameter space:

- **PAR (Parallelism)**: {1, 2, 4, 8, 16, 32}
- **BUFFER_DEPTH**: {256, 512, 1024, 2048}

Total design space: 6 × 4 = 24 unique configurations

## Objective Function

The system minimizes the **Area-Efficiency Product (AEP)**:

```
AEP = total_cells + 0.5 × (total_cells / throughput)
```

Where:
- `total_cells`: Hardware area metric
- `throughput`: Operations per cycle (equal to PAR)

## Constraints

Real-world hardware constraints are enforced:

- **Max Area**: ≤ 1500 cells
- **Min Throughput**: ≥ 2 ops/cycle
- **Max Flip-Flops**: ≤ 400

Violations are penalized with a large penalty (10000) added to the objective.

## LLM Integration

The system supports multiple LLM providers:

1. **OpenAI GPT-4**: Primary LLM for intelligent design proposals
2. **Anthropic Claude**: Alternative LLM provider
3. **Google Gemini**: Additional LLM option
4. **Heuristic Fallback**: Rule-based search when LLMs unavailable

The LLM receives:
- Exploration history (last 10 designs)
- Search space constraints
- Objective function definition
- Current best design

And proposes the next design parameters to explore.
