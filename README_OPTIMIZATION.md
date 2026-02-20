# Microarchitecture Optimization System

An AI-powered hardware design optimization system that explores microarchitecture parameter spaces, synthesizes RTL, runs functional simulations, and finds optimal designs.

## 🎯 Features

- **LLM-Powered Agent**: Uses GPT-4 or Claude to intelligently explore design space
- **Real Synthesis**: Yosys synthesis for accurate hardware metrics
- **Functional Simulation**: Verilator/Icarus for design verification
- **Multi-Metric Optimization**: Balances area, performance, and efficiency
- **Automatic Testbench Generation**: Creates testbenches for each design

## 📋 Prerequisites

### Required Tools

1. **Yosys** (open-source synthesis)
   ```bash
   # macOS
   brew install yosys
   
   # Ubuntu/Debian
   sudo apt-get install yosys
   ```

2. **Icarus Verilog** (simulation - recommended)
   ```bash
   # macOS
   brew install icarus-verilog
   
   # Ubuntu/Debian
   sudo apt-get install iverilog
   ```

### Optional: LLM API Access

For AI-powered design exploration, you can use any of these:

**Option 1: OpenAI (GPT-4)**
```bash
export OPENAI_API_KEY="sk-..."
pip install openai
```

**Option 2: Anthropic (Claude)**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
pip install anthropic
```

**Option 3: Google (Gemini)**
```bash
export GEMINI_API_KEY="..."
pip install google-generativeai
```

**Option 4: Heuristic Fallback**
- No API key needed
- Uses rule-based exploration (included by default)

## 🚀 Quick Start

### 1. Basic Run (Heuristic Agent, No Simulation)

```bash
# No setup needed - uses heuristic agent
export RUN_SIMULATION=false
python3 main.py
```

### 2. With LLM Agent (Smarter Exploration)

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."  # or ANTHROPIC_API_KEY

# Run with LLM-powered agent
python3 main.py
```

### 3. Full System (LLM + Simulation)

```bash
# Install Icarus Verilog first (see prerequisites)

# Set API key
export OPENAI_API_KEY="sk-..."

# Run with both LLM and simulation
export RUN_SIMULATION=true
python3 main.py
```

## 🎛️ Configuration

### Environment Variables

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `OPENAI_API_KEY` | API key | - | OpenAI GPT-4 access |
| `ANTHROPIC_API_KEY` | API key | - | Anthropic Claude access |
| `RUN_SIMULATION` | `true`/`false` | `true` | Enable functional simulation |
| `YOSYS_DEBUG` | `true`/`false` | `false` | Save Yosys output logs |

### Search Space (in main.py)

```python
# Parallelism options
PAR_OPTIONS = [1, 2, 4, 8, 16, 32]

# Buffer depth options
BUFFER_DEPTH_OPTIONS = [256, 512, 1024, 2048]

# Number of iterations
ITERATIONS = 15
```

### Objective Function (in main.py)

```python
def calculate_objective(params, metrics):
    # Area-Efficiency Product (AEP)
    # Lower is better
    area_weight = 1.0
    efficiency_weight = 0.5
    
    objective = (area_weight * total_cells) + 
                (efficiency_weight * area_efficiency)
    return objective
```

## 📊 Metrics Collected

### Hardware Metrics (from Yosys)
- Total cells
- Flip-flops (sequential logic)
- Logic cells (combinational logic)
- Wires and interconnect
- Memories

### Performance Metrics (from Simulation)
- Functional correctness (PASS/FAIL)
- Total cycle count
- Throughput (inputs/cycle)

### Optimization Metrics
- Area-efficiency product (AEP)
- Area per unit throughput
- Design space coverage

## 🔧 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Optimization Loop                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. LLM Agent proposes parameters (PAR, BUFFER_DEPTH)│   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │  2. Generate RTL with proposed parameters            │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │  3. Synthesize with Yosys → Hardware Metrics         │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │  4. Simulate with Icarus → Performance Metrics       │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │  5. Calculate objective function → Update history    │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│                     └──── Back to step 1                     │
└─────────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
.
├── main.py                    # Main optimization loop
├── llm/
│   ├── agent.py              # OLD: Heuristic agent
│   └── llm_agent.py          # NEW: LLM-powered agent
├── tools/
│   ├── run_yosys.py          # Synthesis and metrics extraction
│   └── simulate.py           # Simulation and verification
├── rtl/
│   └── tmp.v                 # Generated RTL (auto-created)
└── tb/
    └── tb_reduce_sum.v       # Generated testbench (auto-created)
```

## 🎓 Example Output

```
======================================================================
 MICROARCHITECTURE OPTIMIZATION
======================================================================

📋 Objective: Minimize Area-Efficiency Product
🔍 Search Space: PAR ∈ {1,2,4,8,16,32}, BUFFER_DEPTH ∈ {256,512,1024,2048}
🔄 Iterations: 15

🤖 Agent Mode: OPENAI

======================================================================
Iteration 1/15: PAR=2, BUFFER_DEPTH=1024
======================================================================
  📊 Hardware Metrics:
     Total Cells:           803
     Flip-Flops:            214
     Logic Cells:           589
     Wires:                 614
  🔬 Simulation Results:
     Status:             ✓ PASSED
     Cycles:             1034
     Sim Throughput:      1.979 inputs/cycle
  🎯 Performance Metrics:
     Throughput:              2 ops/cycle
     Area/Throughput:     401.5 cells/op
  📈 Optimization:
     Objective (AEP):    1004.8
     Best So Far:        1004.8

💡 LLM proposed: PAR=4, BUFFER_DEPTH=1024

[... more iterations ...]

======================================================================
 🏆 OPTIMIZATION COMPLETE
======================================================================

✨ Best Design Found:
   PAR:                  4
   BUFFER_DEPTH:         1024

📊 Best Metrics:
   Total Cells:          1733
   Flip-Flops:           342
   Logic Cells:          1391
   Throughput:           4 ops/cycle
   Area Efficiency:      433.2 cells/op
   Objective Score:      1949.7

📈 Improvement: 42.3% better than worst design
```

## 🐛 Troubleshooting

### "No simulator found"
Install Icarus Verilog:
```bash
brew install icarus-verilog  # macOS
sudo apt-get install iverilog  # Linux
```

### "OpenAI API error"
Check your API key:
```bash
echo $OPENAI_API_KEY
# Should print your key
```

### "Module 'openai' not found"
Install the package:
```bash
pip install openai anthropic
```

### Simulation timeout
Increase timeout in `tools/simulate.py`:
```python
timeout=120  # Increase from 60
```

## 📈 Extending the System

### Add New Parameters

Edit `llm/llm_agent.py`:
```python
self.PAR_OPTIONS = [1, 2, 4, 8, 16, 32, 64]  # Add more values
self.NEW_PARAM_OPTIONS = [...]  # Add new parameter
```

### Change Objective Function

Edit `main.py`:
```python
def calculate_objective(params, metrics):
    # Your custom objective here
    return custom_score
```

### Add More Metrics

Edit `tools/run_yosys.py` to extract additional synthesis metrics, or `tools/simulate.py` for simulation metrics.

## 🤝 Contributing

This is a hackathon project demonstrating AI-powered hardware optimization. Feel free to extend and modify!

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- Yosys Open Synthesis Suite
- Icarus Verilog
- OpenAI GPT-4
- Anthropic Claude
