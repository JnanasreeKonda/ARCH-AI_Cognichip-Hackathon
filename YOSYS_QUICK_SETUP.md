# Yosys Quick Setup Guide

## ✅ YOSYS IS NOW INTEGRATED AND WORKING

**Status:** Yosys is fully integrated into the project and working automatically!

The code automatically detects Yosys from:
- OSS CAD Suite installation (recommended)
- System PATH (if installed globally)
- Local build directory

**No manual setup required!** The project will automatically find and use Yosys when available.

---

## 📦 Installation (If Not Already Done)

If you haven't installed Yosys yet, here are the options:

---

## ⚡ Option 1: Download Pre-built Binary (RECOMMENDED)

### Step 1: Download OSS CAD Suite

**The easiest way to get Yosys for Windows:**

1. Go to: **https://github.com/YosysHQ/oss-cad-suite-build/releases**
2. Look for the latest **Windows x64** release
   - Example: `oss-cad-suite-windows-x64-20240101.zip`
   - Or search for releases with "windows" in the name
3. Download the ZIP file (it's large, ~500MB+)

### Step 2: Extract and Find yosys.exe

1. Extract the downloaded ZIP file
2. Navigate to the `bin/` folder inside
3. **Find `yosys.exe`** in that folder

### Step 3: Install

1. **Create a tools folder:**
   ```powershell
   mkdir C:\tools\yosys
   ```

2. **Copy yosys.exe:**
   ```powershell
   copy <path-to-extracted>\bin\yosys.exe C:\tools\yosys\
   ```
   Example:
   ```powershell
   copy "C:\Users\DELL\Downloads\oss-cad-suite-windows-x64\bin\yosys.exe" C:\tools\yosys\
   ```

3. **Add to PATH:**
   - Press `Win + X` → System → Advanced system settings
   - Click "Environment Variables"
   - Under "System variables", find "Path" → Edit
   - Click "New" → Add: `C:\tools\yosys`
   - Click OK on all dialogs

4. **Restart Terminal/PowerShell** (IMPORTANT!)

5. **Test:**
   ```powershell
   yosys -V
   ```
   Should show: `Yosys 0.38+ ...` or similar

---

## 🔧 Option 2: Build from Source (If You Want)

Since you have the source code, you can build it:

### Prerequisites: Install MSYS2

1. Download MSYS2: https://www.msys2.org/
2. Install and run: `pacman -Syu` (update)

### Install Build Tools

Open **MSYS2 MinGW 64-bit** terminal and run:

```bash
pacman -S mingw-w64-x86_64-gcc
pacman -S mingw-w64-x86_64-make
pacman -S mingw-w64-x86_64-bison
pacman -S mingw-w64-x86_64-flex
pacman -S git
```

### Build Yosys

1. **Navigate to Yosys directory:**
   ```bash
   cd /c/Users/DELL/Desktop/Cognichip/ARCH-AI_Cognichip-Hackathon/yosys
   ```

2. **Initialize submodules** (if needed):
   ```bash
   git submodule update --init --recursive
   ```

3. **Configure:**
   ```bash
   make config-gcc
   ```

4. **Build** (takes 10-20 minutes):
   ```bash
   make -j$(nproc)
   ```

5. **Find yosys.exe:**
   - After build completes, `yosys.exe` will be in the `yosys/` directory
   - Copy it to `C:\tools\yosys\` and add to PATH (see Option 1, Step 3)

---

## ✅ Verification

After installation, test:

```powershell
cd "C:\Users\DELL\Desktop\Cognichip\ARCH-AI_Cognichip-Hackathon"
python test_yosys.py
```

Or directly:
```powershell
yosys -V
```

---

## 🎯 Recommendation

**For Hackathon: Use Option 1 (Pre-built Binary)**

- ✅ Fastest (5 minutes)
- ✅ No compilation needed
- ✅ Tested and reliable
- ✅ Less chance of errors

**Only use Option 2 if:**
- Pre-built binaries aren't available
- You need a specific version
- You want to modify Yosys

---

## 📝 Current Status

**Yosys Integration:**
- ✅ Yosys automatically detected from OSS CAD Suite
- ✅ Code handles PATH setup for DLL dependencies
- ✅ Real hardware synthesis metrics (not estimates)
- ✅ Falls back gracefully if Yosys unavailable

**How It Works:**
1. Code checks for Yosys in multiple locations
2. For OSS CAD Suite, automatically sets PATH to include `bin/` and `lib/`
3. Runs Yosys synthesis with proper environment
4. Extracts real hardware metrics (cells, flip-flops, wires, etc.)

**No manual configuration needed!** Just install OSS CAD Suite and the project will use it automatically.

---

## 🆘 Troubleshooting

### "No Windows binary in releases"
- Use OSS CAD Suite: https://github.com/YosysHQ/oss-cad-suite-build/releases
- It contains yosys.exe in the `bin/` folder

### Build fails
- Ensure all prerequisites installed
- Check MSYS2 setup
- Try pre-built binary instead

### "yosys: command not found" after adding to PATH
- **Restart terminal** (required!)
- Verify: `Test-Path C:\tools\yosys\yosys.exe`

---

**See also:** `YOSYS_DOWNLOAD_GUIDE.md` for detailed instructions
