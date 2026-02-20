# ARCH-AI: AI-Powered Hardware Optimization

An intelligent microarchitecture optimization framework that uses Large Language Models (LLMs) to explore design spaces and find optimal hardware configurations.

## Features

- **LLM-Powered Design Exploration**: Uses OpenAI GPT-4, Anthropic Claude, or Google Gemini to intelligently propose design parameters
- **Hardware Synthesis**: Integrates with Yosys for accurate gate-level synthesis and metrics
- **Multi-Objective Optimization**: Balances area, performance, and efficiency with customizable constraints
- **Comprehensive Reporting**: Generates detailed reports, visualizations, and a unified dashboard
- **Pareto Frontier Analysis**: Identifies optimal trade-off designs
- **Statistical Analysis**: Provides convergence metrics and design space coverage analysis

## Requirements

- Python 3.8+
- Yosys (optional, for accurate synthesis metrics)
- Icarus Verilog (optional, for functional simulation)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
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
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
```

## Usage

### Basic Usage

Run the optimization:
```bash
python main.py
```

### Configuration

Edit `main.py` to customize:
- `ITERATIONS`: Number of optimization iterations (default: 5)
- `MAX_AREA_CELLS`: Maximum area constraint
- `MIN_THROUGHPUT`: Minimum throughput requirement
- `MAX_FLIP_FLOPS`: Maximum flip-flop limit

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for GPT-4
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude
- `GEMINI_API_KEY`: Google API key for Gemini
- `RUN_SIMULATION`: Set to "false" to disable simulation (default: "true")

## Project Structure

```
ARCH-AI_Cognichip-Hackathon/
├── main.py                 # Main optimization loop
├── llm/
│   └── llm_agent.py       # LLM-powered design agent
├── tools/
│   ├── run_yosys.py       # Yosys synthesis integration
│   ├── simulate.py        # Functional simulation
│   ├── results_reporter.py # Report generation
│   ├── dashboard.py       # Comprehensive dashboard
│   ├── enhanced_visualizations.py # Advanced plots
│   ├── statistical_analysis.py # Statistical reports
│   └── comparison_tools.py # Design comparison utilities
├── rtl/                   # Generated RTL files
├── results/               # Output reports and visualizations
└── requirements.txt       # Python dependencies
```

## Output

The optimization generates:
- **JSON/CSV Reports**: Machine-readable optimization results
- **Visualizations**: Optimization progress, design space, Pareto frontier
- **Comprehensive Dashboard**: Single-page view with all metrics and comparisons
- **Best Design RTL**: Verilog code for the optimal design
- **Statistical Reports**: Convergence analysis and design space coverage

## How It Works

1. **Design Proposal**: LLM agent analyzes exploration history and proposes next design parameters
2. **RTL Generation**: Dynamic Verilog code generation based on proposed parameters
3. **Synthesis**: Yosys synthesizes the design to extract hardware metrics
4. **Evaluation**: Objective function evaluates design quality considering constraints
5. **Iteration**: Process repeats, with LLM learning from previous results
6. **Reporting**: Comprehensive reports and visualizations generated

## Objective Function

The optimization minimizes the Area-Efficiency Product (AEP):
```
AEP = total_cells + 0.5 * (total_cells / throughput)
```

Constraints are enforced with penalties:
- Area > MAX_AREA_CELLS
- Throughput < MIN_THROUGHPUT
- Flip-flops > MAX_FLIP_FLOPS

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
