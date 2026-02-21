"""
Yosys Synthesis Integration

Handles hardware synthesis using Yosys and extracts hardware metrics.
Falls back to estimated metrics if Yosys is not available.
"""

import subprocess
import re
import os


def synthesize(verilog_file, debug=False):
    """
    Synthesize Verilog design using Yosys and extract hardware metrics.
    
    Args:
        verilog_file: Path to Verilog file to synthesize
        debug: If True, print detailed Yosys output
        
    Returns:
        Tuple of (area, log, metrics_dict) where:
        - area: Total cell count (or None if synthesis failed)
        - log: Yosys output log
        - metrics_dict: Dictionary with hardware metrics
    """
    # Check for Yosys in multiple locations
    yosys_paths = [
        'yosys',  # System PATH
        # OSS CAD Suite common location
        os.path.join(os.path.expanduser("~"), "Downloads", "oss-cad-suite", "bin", "yosys.exe"),
        os.path.join("C:", "Users", os.getenv("USERNAME", ""), "Downloads", "oss-cad-suite", "bin", "yosys.exe"),
        # Local build
        os.path.join(os.path.dirname(__file__), '..', 'yosys', 'yosys.exe'),  # Local build
        os.path.join(os.path.dirname(__file__), '..', 'yosys', 'yosys'),  # Local build (Unix)
    ]
    
    yosys_cmd = None
    yosys_workdir = None
    yosys_env = None
    for path in yosys_paths:
        try:
            if os.path.exists(path) if os.path.sep in path else True:
                # For OSS CAD Suite, need to run from bin directory and set PATH for DLLs
                if 'oss-cad-suite' in path or ('bin' in path and 'oss-cad' in path):
                    workdir = os.path.dirname(path) if os.path.isfile(path) else path
                    # OSS CAD Suite needs both bin and lib in PATH
                    oss_base = os.path.dirname(os.path.dirname(path))  # Go up from bin/
                    lib_dir = os.path.join(oss_base, 'lib')
                    env = os.environ.copy()
                    env['PATH'] = f"{os.path.dirname(path)};{lib_dir};{env.get('PATH', '')}"
                    result = subprocess.run([path, '-V'], capture_output=True, timeout=5, 
                                           check=True, cwd=workdir, text=True, env=env)
                    yosys_cmd = path
                    yosys_workdir = workdir
                    yosys_env = env
                    break
                else:
                    result = subprocess.run([path, '-V'], capture_output=True, timeout=2, check=True)
                    yosys_cmd = path
                    break
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    
    if yosys_cmd is None:
        # Fallback: Use estimated metrics when Yosys is not available
        print("WARNING: Yosys not found - using estimated synthesis metrics")
        print("(Install Yosys for accurate results: https://github.com/YosysHQ/yosys/releases)")
        
        # Read RTL to estimate based on parameters
        try:
            with open(verilog_file, 'r') as f:
                rtl_content = f.read()
            
            # Extract PAR and BUFFER_DEPTH from RTL
            par_match = re.search(r'parameter\s+PAR\s*=\s*(\d+)', rtl_content)
            buffer_match = re.search(r'parameter\s+BUFFER_DEPTH\s*=\s*(\d+)', rtl_content)
            
            par = int(par_match.group(1)) if par_match else 4
            buffer_depth = int(buffer_match.group(1)) if buffer_match else 1024
            
            # Estimate metrics based on design parameters
            # These are rough estimates based on typical synthesis results
            addr_width = int(__import__('math').ceil(__import__('math').log2(buffer_depth)))
            
            # Estimated cell counts (based on typical synthesis)
            # Each accumulator: ~50 cells, address counter: ~20 cells, control: ~30 cells
            base_cells = 100
            acc_cells_per_par = 50
            counter_cells = 20 + (addr_width * 5)
            control_cells = 30
            
            total_cells = base_cells + (par * acc_cells_per_par) + counter_cells + control_cells
            flip_flops = (par * 32) + addr_width + 2  # 32-bit accs per PAR + counter + control
            logic_cells = total_cells - flip_flops
            wires = int(total_cells * 1.5)  # Typical wire-to-cell ratio
            
            metrics = {
                'total_cells': total_cells,
                'flip_flops': flip_flops,
                'logic_cells': logic_cells,
                'wires': wires,
                'public_wires': 10,
                'memories': 0,
                'processes': 1
            }
            
            return total_cells, f"Estimated synthesis (Yosys not available)\nPAR={par}, BUFFER_DEPTH={buffer_depth}", metrics
        except Exception as e:
            print(f"ERROR: Could not estimate metrics: {e}")
            return None, "", {'total_cells': None, 'flip_flops': 0, 'logic_cells': None, 'wires': None}
    
    # Enhanced synthesis with ABC for timing and area estimation
    # Use the detected yosys command
    yosys_script = f'read_verilog {verilog_file}; synth; stat'
    try:
        # For OSS CAD Suite, run from bin directory with proper PATH for DLLs
        if yosys_workdir and yosys_env:
            result = subprocess.check_output(
                [yosys_cmd, '-p', yosys_script],
                stderr=subprocess.STDOUT,
                timeout=30,
                cwd=yosys_workdir,
                env=yosys_env
            ).decode()
        elif yosys_workdir:
            result = subprocess.check_output(
                [yosys_cmd, '-p', yosys_script],
                stderr=subprocess.STDOUT,
                timeout=30,
                cwd=yosys_workdir
            ).decode()
        else:
            result = subprocess.check_output(
                [yosys_cmd, '-p', yosys_script],
                stderr=subprocess.STDOUT,
                timeout=30
            ).decode()
    except subprocess.CalledProcessError as e:
        result = e.output.decode()
        print(f"Warning: Yosys returned error code {e.returncode}")
        if "not recognized" in result or "not found" in result:
            print("ERROR: Yosys command failed - check installation")
            return None, result, {'total_cells': None, 'flip_flops': 0, 'logic_cells': None, 'wires': None}
    
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
