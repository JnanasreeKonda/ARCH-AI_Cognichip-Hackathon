import subprocess
import re
import os

def synthesize(verilog_file, debug=False):
    # Enhanced synthesis with ABC for timing and area estimation
    cmd = f'yosys -p "read_verilog {verilog_file}; synth; stat"'
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        result = e.output.decode()
        print(f"Warning: Yosys returned error code {e.returncode}")
    
    # Debug: save raw output to file
    if debug or os.environ.get('YOSYS_DEBUG'):
        with open('yosys_output.log', 'w') as f:
            f.write(result)
        print("DEBUG: Yosys output saved to yosys_output.log")

    metrics = {
        'total_cells': None,
        'flip_flops': 0,
        'logic_cells': None,
        'wires': None,
        'public_wires': None,
        'memories': 0,
        'processes': 0
    }

    lines = result.splitlines()
    in_stat_section = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect statistics section
        if 'Number of' in stripped or 'Chip area' in stripped:
            in_stat_section = True
        
        # Total cells: "363 cells" or "Number of cells: 363"
        match = re.search(r"(\d+)\s+cells", stripped)
        if match:
            metrics['total_cells'] = int(match.group(1))
        
        # Wires: "500 wires"
        match = re.search(r"(\d+)\s+wires", stripped)
        if match:
            metrics['wires'] = int(match.group(1))
        
        # Public wires
        match = re.search(r"(\d+)\s+public wires", stripped)
        if match:
            metrics['public_wires'] = int(match.group(1))
        
        # Memories
        match = re.search(r"(\d+)\s+memories", stripped)
        if match:
            metrics['memories'] = int(match.group(1))
        
        # Processes
        match = re.search(r"(\d+)\s+processes", stripped)
        if match:
            metrics['processes'] = int(match.group(1))
        
        # Count flip-flops (look for DFF cells after synthesis)
        # Patterns: $_DFF_P_, $_DFFE_PP_, $_SDFF_PP0_, $_SDFFE_PP0P_, etc.
        if '$_dff' in stripped.lower() or '$_sdff' in stripped.lower():
            # Line format: "32   $_DFFE_PP_" or "75   $_SDFFE_PP0P_"
            match = re.match(r"(\d+)\s+\$_[SD]*DFF", stripped, re.IGNORECASE)
            if match:
                metrics['flip_flops'] += int(match.group(1))
    
    # Calculate logic cells (combinational logic = total - flip_flops)
    if metrics['total_cells'] is not None:
        metrics['logic_cells'] = metrics['total_cells'] - metrics['flip_flops']
    
    # For backward compatibility, return area as first value
    area = metrics['total_cells']
    
    return area, result, metrics
