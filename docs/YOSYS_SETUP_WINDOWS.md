# Yosys Setup on Windows

## Overview

You have downloaded the Yosys source code repository. Here are the options to get Yosys working on Windows.

## Option 1: Use Pre-built Binary (Easiest - Recommended)

**This is the fastest way to get Yosys working:**

1. **Download Pre-built Binary:**
   - Go to: https://github.com/YosysHQ/yosys/releases
   - Download the latest Windows release (e.g., `yosys-win32-mxebin-0.38.zip`)
   - Extract the ZIP file

2. **Add to PATH:**
   - Copy the extracted `yosys.exe` to a folder (e.g., `C:\tools\yosys\`)
   - Add that folder to your Windows PATH:
     - Open System Properties → Environment Variables
     - Edit "Path" variable
     - Add `C:\tools\yosys\` (or your chosen path)
     - Click OK

3. **Verify Installation:**
   ```powershell
   yosys -V
   ```
   Should show version information.

4. **Update Project to Use System Yosys:**
   The project will automatically detect Yosys in PATH. No code changes needed!

## Option 2: Build from Source (Advanced)

If you want to build from the downloaded source:

### Prerequisites

1. **Install MSYS2:**
   - Download from: https://www.msys2.org/
   - Install and update: `pacman -Syu`

2. **Install Build Tools:**
   ```bash
   pacman -S mingw-w64-x86_64-gcc
   pacman -S mingw-w64-x86_64-make
   pacman -S mingw-w64-x86_64-bison
   pacman -S mingw-w64-x86_64-flex
   pacman -S git
   ```

3. **Build Yosys:**
   ```bash
   cd /c/Users/DELL/Desktop/Cognichip/ARCH-AI_Cognichip-Hackathon/yosys
   make config-gcc
   make
   ```

4. **Install:**
   ```bash
   make install PREFIX=/mingw64
   ```

### Alternative: Visual Studio Build

1. **Install Visual Studio** with C++ development tools
2. **Use CMake:**
   ```powershell
   cd yosys
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```
3. **Copy executable** to a folder in PATH

## Option 3: Use Local Yosys Binary (No PATH needed)

If you build Yosys locally, you can update the project to use it directly:

1. **Build Yosys** (using Option 2)
2. **Update `tools/run_yosys.py`** to use local path:
   ```python
   # Instead of 'yosys', use full path
   yosys_path = os.path.join(os.path.dirname(__file__), '..', 'yosys', 'yosys.exe')
   ```

## Recommended Approach

**For Hackathon: Use Option 1 (Pre-built Binary)**

- Fastest setup (5 minutes)
- No compilation needed
- Reliable and tested
- Works immediately

## Verification

After setup, test with:
```powershell
cd "C:\Users\DELL\Desktop\Cognichip\ARCH-AI_Cognichip-Hackathon"
yosys -V
```

If it works, the project will automatically use Yosys instead of estimated metrics!

## Troubleshooting

### "yosys: command not found"
- Yosys not in PATH
- Restart terminal after adding to PATH
- Use full path: `C:\tools\yosys\yosys.exe -V`

### Build Errors
- Ensure all prerequisites installed
- Check MSYS2/MinGW setup
- Try pre-built binary instead

### Permission Errors
- Run terminal as Administrator
- Check file permissions

## Next Steps

Once Yosys is working:
1. Run the project: `python main.py`
2. You'll see "Using Yosys synthesis" instead of "WARNING: Yosys not found"
3. Get accurate hardware metrics instead of estimates
