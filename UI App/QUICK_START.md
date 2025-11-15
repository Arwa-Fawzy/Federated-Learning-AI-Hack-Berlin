# 🚀 Quick Start Guide

## Get Dashboard Running in 3 Steps

### Step 1: Install Dependencies
```bash
cd "UI App"
pip install -r requirements.txt
```

### Step 2: Verify Data Files
The dashboard needs these files in `../federated_data/hybrid/`:
- ✅ `client_0.csv` through `client_4.csv`
- ✅ `client_metadata.json`

### Step 3: Launch Dashboard
**Windows:**
```bash
run_dashboard.bat
```

**Mac/Linux:**
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

**Or manually:**
```bash
streamlit run app.py
```

## 🎯 What to Expect

The dashboard will open at `http://localhost:8501` with:

1. **Top Bar**: Health Score, Uptime %, Active Alerts, Active Sensors
2. **Sidebar**: Facility selector (0-4), Refresh controls, System info
3. **4 Tabs**:
   - 📊 Real-time Monitor
   - 🔬 Sensor Analysis
   - 📈 Historical Data
   - 🤖 AI Insights

## 🎨 Quick Tour

### Select a Facility
Use the sidebar dropdown to switch between 5 facilities

### View Real-time Data
Tab 1 shows live sensor readings and current status

### Analyze Sensors
Tab 2 provides heatmaps and individual sensor diagnostics

### Explore History
Tab 3 lets you select time ranges and compare sensors

### AI Insights
Tab 4 shows federated model performance and predictions

## ⚡ Pro Tips

- **Auto-refresh**: Enable in sidebar for live monitoring
- **Export Data**: Use "Export Report" button at bottom
- **Sensor Selection**: Click sensors in dropdown for detailed view
- **Zoom Charts**: Use Plotly controls (hover for options)

## 🐛 Quick Fixes

**Dashboard won't start?**
```bash
pip install --upgrade streamlit plotly pandas numpy
```

**Data not loading?**
Check that you're in the `UI App` folder when running:
```bash
cd "UI App"
streamlit run app.py
```

**Port conflict?**
```bash
streamlit run app.py --server.port 8502
```

## 📞 Need Help?

See `README.md` for detailed documentation and troubleshooting.

---

**Ready?** Run `streamlit run app.py` and start monitoring! 🎉

