# 📦 Installation Guide

## Option 1: Using Existing Virtual Environment (Recommended)

If you already have a venv with packages:

```bash
# Activate your existing venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install only missing packages
pip install streamlit plotly

# Navigate and run
cd "UI App"
streamlit run app.py
```

## Option 2: Fresh Installation

```bash
# Navigate to UI App
cd "UI App"

# Install all dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

## Option 3: Using Project Virtual Environment

```bash
# Use the project's existing venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Update with UI dependencies
pip install streamlit plotly

# Run from UI App folder
cd "UI App"
streamlit run app.py
```

## ✅ Verify Installation

Run the test script to check everything:

```bash
cd "UI App"
python test_setup.py
```

Expected output:
```
✅ streamlit       - OK
✅ pandas          - OK
✅ numpy           - OK
✅ plotly          - OK
✅ All data files  - OK
✅ Utility functions - OK
✅ App file        - OK

🎉 ALL TESTS PASSED!
```

## 🚀 Launch Dashboard

**Method 1: Direct launch**
```bash
cd "UI App"
streamlit run app.py
```

**Method 2: Using launcher (Windows)**
```bash
cd "UI App"
run_dashboard.bat
```

**Method 3: Using launcher (Mac/Linux)**
```bash
cd "UI App"
chmod +x run_dashboard.sh
./run_dashboard.sh
```

## 🌐 Access Dashboard

Once running, open your browser to:
```
http://localhost:8501
```

## 🔧 Configuration

### Change Port
```bash
streamlit run app.py --server.port 8502
```

### Headless Mode (Server)
```bash
streamlit run app.py --server.headless true
```

### Network Access
```bash
streamlit run app.py --server.address 0.0.0.0
```

## 📋 System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 100MB for app + data size
- **Browser**: Chrome, Firefox, Safari, or Edge (latest versions)

## 🐛 Troubleshooting

### "streamlit: command not found"
```bash
pip install --upgrade streamlit
```

### Import errors
```bash
pip install -r requirements.txt --upgrade
```

### Port already in use
```bash
# Find and kill process on port 8501
netstat -ano | findstr :8501  # Windows
lsof -ti:8501 | xargs kill    # Mac/Linux

# Or use different port
streamlit run app.py --server.port 8502
```

### Slow loading
- Reduce sample size in code (modify `tail(500)` to `tail(200)`)
- Disable auto-refresh in sidebar
- Clear cache: Sidebar → "Refresh Now"

### Data not found
Make sure you're running from `UI App` directory and that `../federated_data/hybrid/` exists with all CSV files.

## 📦 Package Versions

Tested with:
- streamlit 1.30.0+
- pandas 2.0.0+
- numpy 1.24.0+
- plotly 5.18.0+

## 🎯 Quick Test

After installation, run:
```bash
cd "UI App"
python test_setup.py
```

If all tests pass, you're ready to launch! 🚀

