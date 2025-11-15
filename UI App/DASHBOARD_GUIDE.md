# 📊 Dashboard User Guide

## Overview

The SenorMatics Predictive Maintenance Dashboard provides real-time monitoring and analysis of industrial pump health across 5 facilities using federated learning.

## 🎛️ Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│              🏭 SenorMatics Dashboard                   │
├──────────┬──────────────────────────────────────────────┤
│          │  📊 KPIs: Health | Uptime | Alerts | Sensors │
│ Sidebar  │                                               │
│          ├───────────────────────────────────────────────┤
│ • Select │  [📊 Real-time] [🔬 Sensors] [📈 History]     │
│   Facility│                  [🤖 AI Insights]            │
│          │                                               │
│ • Refresh│         Main Content Area                     │
│   Controls│        (Changes based on tab)                │
│          │                                               │
│ • System │                                               │
│   Info   │                                               │
│          │                                               │
└──────────┴───────────────────────────────────────────────┘
```

## 🏭 Getting Started

### 1. Launch Dashboard
```bash
cd "UI App"
streamlit run app.py
```

### 2. Select Facility
Use the sidebar dropdown to choose from:
- Facility 0 (11,016 samples)
- Facility 1 (77,112 samples)
- Facility 2 (33,048 samples)
- Facility 3 (66,096 samples)
- Facility 4 (33,048 samples)

### 3. Monitor Status
View top KPIs:
- **Health Score**: 0-100 composite metric
- **Uptime %**: Time in NORMAL state
- **Active Alerts**: Current anomalies
- **Active Sensors**: Working sensors

## 📑 Tab Guide

### Tab 1: 📊 Real-time Monitor

**Purpose**: Live operational monitoring

**Features**:
- Current machine status (NORMAL/RECOVERING/BROKEN)
- Health gauge (0-100 scale)
- Status distribution pie chart
- Live sensor readings (last 500 samples)
- Active alerts panel

**Use Cases**:
- Morning shift handoff
- Continuous monitoring
- Quick health check
- Alert triage

**How to Use**:
1. Check current status indicator
2. Review health gauge (aim for >75)
3. Scroll through key sensor trends
4. Act on red alerts immediately

### Tab 2: 🔬 Sensor Analysis

**Purpose**: Deep diagnostic analysis

**Features**:
- 20-sensor heatmap (normalized values)
- Individual sensor selector
- Statistical metrics (mean, std, current)
- Historical trends (last 500 samples)
- Distribution histogram
- Box plot for outliers

**Use Cases**:
- Troubleshooting specific sensors
- Identifying drift or calibration issues
- Finding outliers
- Sensor health assessment

**How to Use**:
1. Review heatmap for patterns
2. Select suspicious sensor from dropdown
3. Check statistics (high std = unstable)
4. Review trend plot for drift
5. Check histogram for unusual distributions

**Color Guide**:
- 🔴 Red: Above normal (high values)
- 🟡 Yellow: Normal range
- 🟢 Green: Below normal (low values)

### Tab 3: 📈 Historical Data

**Purpose**: Time-based analysis and comparisons

**Features**:
- Adjustable time range (index-based)
- Status distribution in range
- Multi-sensor comparison (up to 6)
- Sample statistics

**Use Cases**:
- Investigating past incidents
- Comparing sensor behaviors
- Identifying patterns over time
- Preparing maintenance reports

**How to Use**:
1. Adjust start/end index sliders
2. Review status metrics for selected range
3. Select sensors to compare (use Ctrl+Click)
4. Look for correlations between sensors
5. Export data if needed

### Tab 4: 🤖 AI Insights

**Purpose**: Machine learning predictions and analytics

**Features**:
- Global vs local model accuracy
- Anomaly detection rate
- Anomaly score timeline
- Feature importance ranking
- Privacy metrics
- Failure risk assessment

**Use Cases**:
- Understanding model performance
- Identifying most critical sensors
- Validating predictions
- Planning preventive maintenance

**How to Use**:
1. Check model accuracy metrics
2. Review anomaly score plot
3. Identify spikes above threshold
4. Note top 10 critical sensors
5. Consider failure risk level

## 🎨 Visual Elements

### Status Indicators

- 🟢 **NORMAL**: Green badge, healthy operation
- 🟡 **RECOVERING**: Yellow badge, transitional
- 🔴 **BROKEN**: Red badge, requires immediate attention

### Health Gauge

- **0-50**: Red zone (critical)
- **50-75**: Yellow zone (caution)
- **75-100**: Green zone (healthy)

### Alerts

- **HIGH**: Red border, immediate action required
- **MEDIUM**: Orange, schedule inspection
- **LOW**: Yellow, monitor closely

## 🔄 Refresh Controls

**Auto-refresh**:
- Enable checkbox in sidebar
- Set interval (30-300 seconds)
- Useful for control room displays

**Manual refresh**:
- Click "🔄 Refresh Now" button
- Clears cache and reloads data
- Use after data updates

## 📥 Export Features

### CSV Export
1. Navigate to desired facility
2. Click "📥 Export Report (CSV)" at bottom
3. Click "Download CSV" in popup
4. File saved as: `facility_{id}_report_{date}.csv`

### Summary Report
1. Click "📊 Generate Summary"
2. Includes:
   - Health score
   - Status distribution
   - Anomaly count
   - Data quality metrics

## 🎯 Common Workflows

### Morning Check
1. Open dashboard
2. Review all 5 facilities (quick scan)
3. Focus on any with health <75
4. Check active alerts
5. Plan maintenance accordingly

### Incident Investigation
1. Select affected facility
2. Go to Historical Data tab
3. Narrow time range to incident period
4. Select relevant sensors
5. Look for anomalies before failure

### Sensor Troubleshooting
1. Go to Sensor Analysis tab
2. Review heatmap for outliers
3. Select suspicious sensor
4. Check distribution and trends
5. Determine if sensor needs calibration

### Maintenance Planning
1. Review AI Insights tab
2. Check failure risk for each facility
3. Note top critical sensors
4. Schedule preventive maintenance
5. Export report for documentation

## 📊 Interpreting Metrics

### Health Score
- **90-100**: Excellent, routine monitoring only
- **75-89**: Good, no immediate concerns
- **60-74**: Fair, increased monitoring
- **Below 60**: Poor, investigation required

### Uptime Percentage
- **>95%**: Industry standard, excellent
- **90-95%**: Acceptable, minor issues
- **<90%**: Below standard, needs attention

### Anomaly Rate
- **<2%**: Normal variation
- **2-5%**: Elevated, monitor trends
- **>5%**: Concerning, investigate causes

## 🔍 Tips & Tricks

### Performance Optimization
- Disable auto-refresh when not needed
- Use smaller time ranges for faster loading
- Select fewer sensors in multi-plots

### Chart Interactions
- **Hover**: See exact values
- **Zoom**: Click and drag on plot
- **Pan**: Shift + drag
- **Reset**: Double-click on plot
- **Save**: Camera icon (top-right of plot)

### Keyboard Shortcuts
- `Ctrl + R`: Refresh page
- `Ctrl + S`: Focus search/selector
- `Esc`: Close modals

### Best Practices
1. Start each shift with Overview check
2. Set alerts threshold conservatively
3. Document all anomalies
4. Compare facilities for patterns
5. Export data before making changes

## 🚨 Alert Response

### HIGH Severity
1. Immediate visual inspection
2. Check related sensors
3. Contact maintenance team
4. Document in shift log
5. Monitor continuously

### MEDIUM Severity
1. Note in maintenance schedule
2. Increase monitoring frequency
3. Plan inspection within 24-48h
4. Check historical patterns

### LOW Severity
1. Add to watchlist
2. Monitor for escalation
3. Review at next scheduled maintenance

## 📱 Control Room Display

For 24/7 monitoring:
1. Launch on dedicated screen
2. Select key facility
3. Enable auto-refresh (60s)
4. Keep on Real-time Monitor tab
5. Set browser to fullscreen (F11)

## 🎓 Training New Users

Recommended learning path:
1. Start with QUICK_START.md (5 min)
2. Explore Real-time Monitor tab (10 min)
3. Practice sensor selection (10 min)
4. Review historical data (15 min)
5. Understand AI insights (15 min)
6. Run sample scenarios (30 min)

## 📞 Support

For issues:
1. Check INSTALLATION.md
2. Run `python test_setup.py`
3. Review README.md troubleshooting
4. Check Streamlit logs
5. Verify data files exist

## 🔐 Privacy & Security

This dashboard implements federated learning:
- ✅ Raw data stays at each facility
- ✅ Only model weights are shared
- ✅ No sensitive data transmission
- ✅ Privacy-preserving by design

## 📈 Success Metrics

Track these over time:
- Average health score trend
- Alert resolution time
- Uptime improvement
- Maintenance cost reduction
- Failure prediction accuracy

---

**Happy Monitoring! 🏭**

For detailed technical documentation, see `README.md`

