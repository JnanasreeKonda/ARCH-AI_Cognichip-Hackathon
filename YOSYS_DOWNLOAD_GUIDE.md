# Yosys Download Guide - Windows

## ⚠️ Important: Git Repository vs Pre-built Binary

**The git repository you downloaded contains SOURCE CODE, not a ready-to-use executable.**

To get Yosys working, you have two options:

---

## Option 1: Download Pre-built Binary (EASIEST - 5 minutes)

### Step 1: Get Pre-built Binary

**Method A: Direct Yosys Release**
1. Go to: https://github.com/YosysHQ/yosys/releases
2. Look for Windows builds (may be limited)
3. Download if available

**Method B: OSS CAD Suite (Recommended)**
1. Go to: https://github.com/YosysHQ/oss-cad-suite-build/releases
2. Download the latest Windows release:
   - Example: `oss-cad-suite-windows-x64-20240101.zip`
   - Or search for "windows" in release names
3. Extract the ZIP file
4. Navigate to `bin/` folder inside
5. Find `yosys.exe`

### Step 2: Install

1. **Create a tools folder** (if you don't have one):
   ```powershell
   mkdir C:\tools\yosys
   ```

2. **Copy yosys.exe** to that folder:
   ```powershell
   copy <path-to-extracted>\bin\yosys.exe C:\tools\yosys\
   ```

3. **Add to PATH:**
   - Press `Win + R` → type `sysdm.cpl` → Enter
   - Go to "Advanced" tab → "Environment Variables"
   - Under "System variables", select "Path" → "Edit"
   - Click "New" → Add: `C:\tools\yosys`
   - Click OK on all dialogs

4. **Restart Terminal/PowerShell**

5. **Test:**
   ```powershell
   yosys -V
   ```
   Should show version like: `Yosys 0.38+ ...`

---

## Option 2: Build from Source (20-30 minutes)

Since you already have the source code, you can build it:

### Prerequisites

1. **Install MSYS2:**
   - Download: https://www.msys2.org/
   - Install and run: `pacman -Syu` (update)

2. **Install Build Tools:**
   ```bash
   pacman -S mingw-w64-x86_64-gcc
   pacman -S mingw-w64-x86_64-make
   pacman -S mingw-w64-x86_64-bison
   pacman -S mingw-w64-x86_64-flex
   pacman -S git
   ```

### Build Steps

1. **Open MSYS2 MinGW 64-bit terminal**

2. **Navigate to Yosys directory:**
   ```bash
   cd /c/Users/DELL/Desktop/Cognichip/ARCH-AI_Cognichip-Hackathon/yosys
   ```

3. **Initialize submodules** (if needed):
   ```bash
   git submodule update --init --recursive
   ```

4. **Configure build:**
   ```bash
   make config-gcc
   ```

5. **Build:**
   ```bash
   make -j$(nproc)
   ```
   This will take 10-20 minutes.

6. **Find the executable:**
   - After build, `yosys.exe` will be in the `yosys/` directory
   - Copy it to a folder in PATH (see Option 1, Step 2)

---

## Quick Verification

After installation (either method), test:

```powershell
cd "C:\Users\DELL\Desktop\Cognichip\ARCH-AI_Cognichip-Hackathon"
python test_yosys.py
```

Or directly:
```powershell
yosys -V
```

---

## Recommendation

**For Hackathon: Use Option 1 (Pre-built Binary)**

- Fastest (5 minutes vs 30 minutes)
- No compilation needed
- Tested and reliable
- Less chance of errors

**Only use Option 2 if:**
- You need a specific version
- You want to modify Yosys
- Pre-built binaries aren't available

---

## Troubleshooting

### "No Windows binary found in releases"
- Use OSS CAD Suite instead (Method B in Option 1)
- Or build from source (Option 2)

### Build fails
- Ensure all prerequisites installed
- Check MSYS2 setup
- Try pre-built binary instead

### "yosys: command not found" after adding to PATH
- **Restart terminal/PowerShell** (required!)
- Check PATH: `echo $env:PATH` (PowerShell)
- Verify file exists: `Test-Path C:\tools\yosys\yosys.exe`

---

## Next Steps

Once Yosys is working:
1. Run: `python test_yosys.py` to verify
2. Run: `python main.py` - it will automatically use Yosys!
3. You'll see accurate hardware metrics instead of estimates
