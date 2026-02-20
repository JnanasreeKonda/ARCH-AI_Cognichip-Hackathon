#!/bin/bash

# Microarchitecture Optimization System - Setup and Run Script

echo "======================================================================"
echo " Microarchitecture Optimization System"
echo "======================================================================"

# Check for Yosys
echo ""
echo "🔍 Checking dependencies..."
if command -v yosys &> /dev/null; then
    echo "  ✓ Yosys found: $(yosys -V | head -1)"
else
    echo "  ✗ Yosys not found!"
    echo "    Install: brew install yosys (macOS) or apt-get install yosys (Linux)"
    exit 1
fi

# Check for Icarus Verilog
if command -v iverilog &> /dev/null; then
    echo "  ✓ Icarus Verilog found: $(iverilog -v 2>&1 | head -1)"
    SIMULATOR_AVAILABLE=true
else
    echo "  ⚠️  Icarus Verilog not found - simulation disabled"
    echo "    Install: brew install icarus-verilog (macOS) or apt-get install iverilog (Linux)"
    SIMULATOR_AVAILABLE=false
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p rtl
mkdir -p tb
mkdir -p logs
echo "  ✓ Directories created"

# Check for Python packages
echo ""
echo "🐍 Checking Python packages..."
python3 -c "import openai" 2>/dev/null && OPENAI_OK=true || OPENAI_OK=false
python3 -c "import anthropic" 2>/dev/null && ANTHROPIC_OK=true || ANTHROPIC_OK=false

if [ "$OPENAI_OK" = true ] || [ "$ANTHROPIC_OK" = true ]; then
    echo "  ✓ LLM packages found"
else
    echo "  ⚠️  No LLM packages found - using heuristic agent"
    echo "    Optional: pip install openai anthropic"
fi

# Check for API keys
echo ""
echo "🔑 Checking for API keys..."
if [ -n "$OPENAI_API_KEY" ]; then
    echo "  ✓ OpenAI API key found"
    LLM_MODE="OPENAI"
elif [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "  ✓ Anthropic API key found"
    LLM_MODE="ANTHROPIC"
else
    echo "  ⚠️  No API keys found - using heuristic agent"
    echo "    Optional: export OPENAI_API_KEY=sk-... or ANTHROPIC_API_KEY=sk-ant-..."
    LLM_MODE="HEURISTIC"
fi

# Configuration summary
echo ""
echo "======================================================================"
echo " Configuration Summary"
echo "======================================================================"
echo "  Agent Mode:     $LLM_MODE"
echo "  Simulator:      $([ "$SIMULATOR_AVAILABLE" = true ] && echo "Enabled (Icarus)" || echo "Disabled")"
echo "  Synthesis:      Enabled (Yosys)"
echo "======================================================================"

# Set simulation flag
if [ "$SIMULATOR_AVAILABLE" = true ]; then
    export RUN_SIMULATION=true
else
    export RUN_SIMULATION=false
fi

# Run confirmation
echo ""
read -p "Start optimization? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Starting optimization..."
    echo ""
    python3 main.py 2>&1 | tee logs/optimization_$(date +%Y%m%d_%H%M%S).log
    echo ""
    echo "✨ Done! Check logs/ directory for full output"
else
    echo "Cancelled."
fi
