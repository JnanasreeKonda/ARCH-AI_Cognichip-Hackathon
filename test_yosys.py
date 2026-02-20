#!/usr/bin/env python3
"""
Test Yosys Availability

Checks if Yosys is available in system PATH or local directory.
"""

import subprocess
import os
import sys

def test_yosys():
    """Test if Yosys is available"""
    
    print("="*70)
    print(" YOSYS AVAILABILITY CHECK")
    print("="*70)
    
    # Check multiple locations
    yosys_paths = [
        ('yosys', 'System PATH'),
        (os.path.join('yosys', 'yosys.exe'), 'Local Windows build'),
        (os.path.join('yosys', 'yosys'), 'Local Unix build'),
        # OSS CAD Suite locations
        (os.path.join(os.path.expanduser("~"), "Downloads", "oss-cad-suite", "bin", "yosys.exe"), 'OSS CAD Suite (Downloads)'),
        (os.path.join("C:", "Users", os.getenv("USERNAME", ""), "Downloads", "oss-cad-suite", "bin", "yosys.exe"), 'OSS CAD Suite (C: Users)'),
    ]
    
    found = False
    working_path = None
    
    for path, description in yosys_paths:
        print(f"\nChecking: {description}")
        print(f"  Path: {path}")
        
        try:
            # Check if file exists (for local paths)
            if os.path.sep in path:
                if not os.path.exists(path):
                    print(f"  Status: File not found")
                    continue
            
            # Try to run yosys -V
            # For OSS CAD Suite, need to run from bin directory with PATH set for DLLs
            if 'oss-cad-suite' in path or ('bin' in path and 'oss-cad' in path):
                workdir = os.path.dirname(path) if os.path.isfile(path) else path
                # OSS CAD Suite needs both bin and lib in PATH
                oss_base = os.path.dirname(os.path.dirname(path))  # Go up from bin/
                lib_dir = os.path.join(oss_base, 'lib')
                env = os.environ.copy()
                env['PATH'] = f"{os.path.dirname(path)};{lib_dir};{env.get('PATH', '')}"
                result = subprocess.run(
                    [path, '-V'],
                    capture_output=True,
                    timeout=5,
                    check=True,
                    cwd=workdir,
                    text=True,
                    env=env
                )
            else:
                result = subprocess.run(
                    [path, '-V'],
                    capture_output=True,
                    timeout=5,
                    check=True
                )
            
            if isinstance(result.stdout, str):
                output = result.stdout.strip()
            else:
                output = result.stdout.decode('utf-8', errors='ignore').strip()
            print(f"  Status: [OK] FOUND AND WORKING")
            print(f"  Version: {output.split(chr(10))[0] if output else 'Unknown'}")
            
            found = True
            working_path = path
            break
            
        except FileNotFoundError:
            print(f"  Status: [NOT FOUND]")
        except subprocess.CalledProcessError as e:
            print(f"  Status: [ERROR] Exit code {e.returncode}")
            try:
                err_output = e.stdout.decode('utf-8', errors='ignore')[:100]
                print(f"  Output: {err_output}")
            except:
                pass
        except subprocess.TimeoutExpired:
            print(f"  Status: [TIMEOUT]")
        except Exception as e:
            print(f"  Status: [ERROR] {str(e)}")
    
    print("\n" + "="*70)
    
    if found:
        print("[SUCCESS] YOSYS IS WORKING!")
        print(f"  Using: {working_path}")
        print("\nThe project will use Yosys for accurate hardware synthesis.")
        return True
    else:
        print("[WARNING] YOSYS NOT FOUND")
        print("\nThe project will use estimated synthesis metrics.")
        print("\nTo install Yosys:")
        print("  1. Download pre-built binary from:")
        print("     https://github.com/YosysHQ/yosys/releases")
        print("  2. Extract and add to PATH")
        print("  3. Or build from source (see YOSYS_QUICK_SETUP.md)")
        return False

if __name__ == "__main__":
    test_yosys()
