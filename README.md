# ARCH-AI: AI-Powered Hardware Optimization

An intelligent microarchitecture optimization framework that uses Reinforcement Learning (DQN) and Large Language Models (LLMs) to explore design spaces and find optimal hardware configurations.

## Features

- **Hybrid AI Optimization**:
  - **DQN Agent**: Deep Reinforcement Learning for efficient design space exploration
  - **LLM Agent**: GPT-4/Claude/Gemini for reasoning-based optimization
  - **Heuristic Fallback**: Robust rule-based search when AI models are unavailable
- **Hardware Synthesis**: Fully integrated with Yosys for accurate gate-level synthesis and real hardware metrics
- **Multi-Objective Optimization**: Balances area, performance, and efficiency with customizable constraints
- **Comprehensive Reporting**: Generates detailed reports, visualizations, and a unified dashboard
- **Pareto Frontier Analysis**: Identifies optimal trade-off designs
- **Automatic Verilog Generation**: Produces ready-to-use RTL for the best discovered design

## Requirements

- Python 3.8+
- Yosys (integrated - automatically detects OSS CAD Suite installation)
- Icarus Verilog (optional, for functional simulation)
- PyTorch (for DQN agent)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JnanasreeKonda/ARCH-AI_Cognichip-Hackathon.git
cd ARCH-AI_Cognichip-Hackathon
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys (optional, for LLM features):
Create a `.env` file in the project root or set environment variables:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
```

## Usage

### 1. Quick Start (Recommended)

Run the main optimization loop. The system automatically selects the best available agent (DQN > LLM > Heuristic).

```bash
python main.py
```

### 2. Train the DQN Agent

To train a custom Reinforcement Learning agent:

```bash
# Quick training (10 episodes)
python reinforcement_learning/training/run_dqn_quick.py

# Full training (50 episodes)
python reinforcement_learning/training/run_dqn_quick.py --full
```

### 3. Generate Verilog for Specific Parameters

If you want to generate Verilog code for specific parameters manually:

```bash
python tools/generate_verilog.py --par 4 --buffer_depth 1024
```

## Project Structure

```
ARCH-AI_Cognichip-Hackathon/
├── main.py                 # Main optimization loop
├── llm/
│   └── unified_agent.py    # Unified agent (DQN + LLM + Heuristic)
├── rl/
│   ├── training/           # DQN training scripts
│   └── checkpoints/        # Trained model checkpoints
├── tools/
│   ├── run_yosys.py        # Yosys synthesis integration
│   ├── simulate.py         # Functional simulation
│   ├── results_reporter.py # Report generation
│   └── generate_verilog.py # Verilog code generator
├── rtl/                    # Generated RTL files
│   └── best_design.v       # Best design from last run
├── results/                # Output reports and visualizations
└── requirements.txt        # Python dependencies
```

## Output

The optimization generates:
- **rtl/best_design.v**: Ready-to-use Verilog code for the optimal design
- **results/optimization_results.json**: Machine-readable results
- **results/optimization_plots.png**: Visualization of the optimization process
- **results/optimization_report.txt**: Detailed text summary

## Optimization Objective

The framework minimizes the **Area-Efficiency Product (AEP)**:
```
AEP = total_cells + 0.5 * (total_cells / throughput)
```

Constraints are enforced with penalties:
- Area > MAX_AREA_CELLS
- Throughput < MIN_THROUGHPUT
- Flip-flops > MAX_FLIP_FLOPS

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
