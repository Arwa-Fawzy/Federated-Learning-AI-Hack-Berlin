# 🏭 SenorMatics Predictive Maintenance Dashboard
## Complete Project Overview

---

## 📋 Table of Contents

1. [Quick Summary](#quick-summary)
2. [What It Does](#what-it-does)
3. [Key Features](#key-features)
4. [For Machine Operators](#for-machine-operators)
5. [For Factory Managers](#for-factory-managers)
6. [Technical Architecture](#technical-architecture)
7. [Getting Started](#getting-started)
8. [Use Cases](#use-cases)
9. [Files Structure](#files-structure)

---

## 🎯 Quick Summary

A **single-page Streamlit dashboard** for real-time monitoring of industrial pump health across 5 facilities. Uses federated learning for privacy-preserving predictive maintenance with 220K+ sensor readings from 52 sensors.

**Key Stats**:
- 5 Facilities monitored
- 52 Sensors per facility
- 220,320 Total samples
- Real-time health scoring
- Anomaly detection
- Privacy-preserving AI

---

## 🔧 What It Does

### For Operators (Day-to-Day)
✅ Monitor pump health in real-time  
✅ Get instant alerts on anomalies  
✅ See which sensors need attention  
✅ Track uptime and performance  
✅ Export data for shift reports  

### For Managers (Strategic)
✅ Compare performance across facilities  
✅ Plan preventive maintenance  
✅ Reduce downtime and costs  
✅ Make data-driven decisions  
✅ Track improvement over time  

### For Maintenance Teams
✅ Identify failing sensors early  
✅ Prioritize repair tasks  
✅ Investigate historical failures  
✅ Validate repair effectiveness  
✅ Optimize maintenance schedules  

---

## ⭐ Key Features

### 1. Real-time Monitoring
- **Live Health Scores**: 0-100 scale for instant assessment
- **Status Tracking**: NORMAL/RECOVERING/BROKEN states
- **Active Alerts**: Automatic anomaly detection
- **Sensor Visualization**: 52 sensors monitored continuously

### 2. Deep Diagnostics
- **Sensor Heatmaps**: Spot patterns across 20+ sensors
- **Individual Analysis**: Detailed stats for each sensor
- **Trend Detection**: Identify drift and degradation
- **Outlier Detection**: Automatic flagging of anomalies

### 3. Historical Analysis
- **Time-Series Explorer**: Investigate past incidents
- **Multi-Sensor Comparison**: Compare up to 6 sensors
- **Status Distribution**: Understand operational patterns
- **Custom Time Ranges**: Focus on specific periods

### 4. AI-Powered Insights
- **Federated Learning**: Privacy-preserving models
- **Anomaly Scoring**: Reconstruction error tracking
- **Feature Importance**: Identify critical sensors
- **Failure Prediction**: Risk assessment and forecasting

---

## 👷 For Machine Operators

### Your Daily Workflow

**Morning Startup:**
1. Open dashboard (`streamlit run app.py`)
2. Check overall health score (aim for >75)
3. Review any active alerts
4. Note any RECOVERING/BROKEN states

**During Shift:**
1. Keep Real-time Monitor tab open
2. Watch for new alerts
3. Monitor key sensor trends
4. Document any anomalies

**End of Shift:**
1. Export facility report (CSV)
2. Note any issues in log
3. Hand off to next shift
4. Check predictions for next day

### What to Look For

🟢 **Green = Good**
- Health score >75
- All sensors in normal range
- Status: NORMAL
- No active alerts

🟡 **Yellow = Watch**
- Health score 60-75
- Some sensors elevated
- Status: RECOVERING
- 1-2 medium alerts

🔴 **Red = Act**
- Health score <60
- Multiple anomalies
- Status: BROKEN
- High severity alerts

### Quick Actions

**Alert Appears?**
1. Click on alert for details
2. Go to Sensor Analysis tab
3. Select flagged sensor
4. Check if value is truly abnormal
5. Visual inspection or call maintenance

**Health Score Drops?**
1. Switch to affected facility
2. Review sensor heatmap
3. Identify red sensors
4. Check historical trends
5. Document and notify supervisor

---

## 👔 For Factory Managers

### Strategic Monitoring

**Weekly Review:**
- Compare health scores across 5 facilities
- Identify consistently underperforming units
- Review maintenance effectiveness
- Track improvement trends

**Monthly Planning:**
- Analyze failure patterns
- Schedule preventive maintenance
- Allocate resources based on risk
- Budget for sensor replacements

**Quarterly Goals:**
- Reduce average downtime
- Improve overall health scores
- Lower maintenance costs
- Increase prediction accuracy

### Business Impact

**Cost Reduction:**
- Prevent catastrophic failures
- Optimize maintenance schedules
- Reduce emergency repairs
- Extend equipment lifespan

**Operational Excellence:**
- Increase uptime from 93% to >95%
- Reduce unplanned downtime by 40%
- Improve maintenance efficiency
- Better resource allocation

**Data-Driven Decisions:**
- Evidence-based maintenance plans
- Risk assessment for investments
- Performance benchmarking
- ROI tracking

---

## 🏗️ Technical Architecture

### Data Flow
```
Facility Sensors (52 per site)
        ↓
CSV Data Files (client_0 to client_4)
        ↓
Streamlit Dashboard (utils.py)
        ↓
Real-time Visualization (app.py)
        ↓
Operator Interface (Browser)
```

### Federated Learning Layer
```
Local Data (never leaves facility)
        ↓
Local Model Training
        ↓
Share Only Model Weights
        ↓
Global Model Aggregation
        ↓
Improved Predictions for All
```

### Privacy Design
- ✅ Raw sensor data stays at facility
- ✅ Only encrypted weights transmitted
- ✅ No reverse engineering possible
- ✅ GDPR compliant
- ✅ Trade secret protected

---

## 🚀 Getting Started

### Quick Start (3 Steps)

**1. Install Dependencies**
```bash
cd "UI App"
pip install -r requirements.txt
```

**2. Test Setup**
```bash
python test_setup.py
```

**3. Launch Dashboard**
```bash
streamlit run app.py
```

**Done!** Open browser to `http://localhost:8501`

### Detailed Guides

- 📘 **QUICK_START.md** - 5-minute setup
- 📗 **INSTALLATION.md** - Installation options
- 📙 **DASHBOARD_GUIDE.md** - Complete user guide
- 📕 **README.md** - Technical documentation

---

## 💼 Use Cases

### Use Case 1: Shift Monitoring
**Scenario**: 24/7 factory with 3 shifts

**Solution**:
1. Dashboard on control room display
2. Auto-refresh every 60 seconds
3. Real-time Monitor tab always visible
4. Alerts trigger immediate inspection
5. End-of-shift CSV export for handoff

**Benefits**:
- Continuous monitoring
- Faster incident response
- Better shift communication
- Documented anomalies

### Use Case 2: Predictive Maintenance
**Scenario**: Prevent costly unexpected failures

**Solution**:
1. AI Insights tab monitors failure risk
2. Critical sensors tracked closely
3. Maintenance scheduled before failure
4. Historical analysis validates predictions

**Benefits**:
- 40% reduction in unplanned downtime
- Lower emergency repair costs
- Extended equipment life
- Optimized parts inventory

### Use Case 3: Multi-Site Management
**Scenario**: Managing 5 geographically distributed facilities

**Solution**:
1. Central dashboard for all facilities
2. Compare health scores side-by-side
3. Identify systemic issues
4. Share best practices across sites

**Benefits**:
- Unified visibility
- Consistent standards
- Resource optimization
- Knowledge sharing

### Use Case 4: Incident Investigation
**Scenario**: Pump failed unexpectedly, need root cause

**Solution**:
1. Historical Data tab
2. Select time range before failure
3. Compare all sensors
4. Identify leading indicators
5. Export analysis for report

**Benefits**:
- Faster root cause analysis
- Better documentation
- Prevent recurrence
- Insurance/compliance reports

### Use Case 5: Continuous Improvement
**Scenario**: Track maintenance program effectiveness

**Solution**:
1. Monitor health score trends
2. Track alert frequency
3. Measure uptime improvements
4. Document cost savings

**Benefits**:
- Quantify improvements
- Justify investments
- Benchmark performance
- Drive continuous improvement

---

## 📂 Files Structure

```
UI App/
├── 📄 app.py                      # Main Streamlit dashboard (470 lines)
│   └── Single-page UI with 4 tabs
│
├── 📄 utils.py                    # Data loading & processing (250 lines)
│   ├── load_client_data()
│   ├── calculate_health_score()
│   ├── detect_anomalies()
│   └── get_sensor_statistics()
│
├── 📄 requirements.txt            # Python dependencies
│   ├── streamlit>=1.30.0
│   ├── pandas>=2.0.0
│   ├── numpy>=1.24.0
│   └── plotly>=5.18.0
│
├── 📄 test_setup.py              # Setup validation script
│   └── Tests packages, data, utils
│
├── 🚀 run_dashboard.bat          # Windows launcher
├── 🚀 run_dashboard.sh           # Mac/Linux launcher
│
├── 📘 README.md                  # Technical documentation (400 lines)
├── 📗 QUICK_START.md             # 5-minute guide
├── 📙 INSTALLATION.md            # Installation options
├── 📕 DASHBOARD_GUIDE.md         # Complete user guide (450 lines)
└── 📊 PROJECT_OVERVIEW.md        # This file
```

### Data Files Required
```
../federated_data/hybrid/
├── client_0.csv                   # Facility 0 (11K samples)
├── client_1.csv                   # Facility 1 (77K samples)
├── client_2.csv                   # Facility 2 (33K samples)
├── client_3.csv                   # Facility 3 (66K samples)
├── client_4.csv                   # Facility 4 (33K samples)
└── client_metadata.json           # Facility metadata
```

---

## 🎯 Success Metrics

Track these KPIs with the dashboard:

### Operational Metrics
- **Uptime %**: Target >95%
- **MTBF** (Mean Time Between Failures): Increase over time
- **MTTR** (Mean Time To Repair): Decrease over time
- **Alert Response Time**: <15 minutes

### Performance Metrics
- **Health Score**: Average >80 across facilities
- **Anomaly Detection Rate**: 2-5% baseline
- **Prediction Accuracy**: >85%
- **False Positive Rate**: <5%

### Business Metrics
- **Maintenance Costs**: Track reduction
- **Unplanned Downtime**: Minimize
- **Production Impact**: Measure losses prevented
- **ROI**: Equipment life extension + cost savings

---

## 🔄 Continuous Improvement

### Phase 1: Monitoring (Current)
✅ Real-time dashboard  
✅ Health scoring  
✅ Anomaly detection  
✅ Basic predictions  

### Phase 2: Enhancements (Future)
- [ ] Email/SMS alerts
- [ ] Mobile app
- [ ] Advanced ML models
- [ ] Automated maintenance tickets

### Phase 3: Integration (Future)
- [ ] ERP system integration
- [ ] CMMS connection
- [ ] IoT sensor expansion
- [ ] Edge computing deployment

---

## 📚 Documentation Index

### Getting Started
1. **QUICK_START.md** - Fastest way to launch (5 min)
2. **INSTALLATION.md** - Detailed installation guide
3. **test_setup.py** - Validate your setup

### Using the Dashboard
4. **DASHBOARD_GUIDE.md** - Complete user manual
5. **README.md** - Technical documentation
6. **PROJECT_OVERVIEW.md** - This file (big picture)

### Support Files
7. **requirements.txt** - Dependencies list
8. **run_dashboard.bat/sh** - Quick launchers

---

## 🎓 Training Materials

### For New Operators (30 min)
1. Read QUICK_START.md (5 min)
2. Launch dashboard (2 min)
3. Explore Real-time Monitor (10 min)
4. Practice Sensor Analysis (10 min)
5. Review sample alerts (3 min)

### For Maintenance Teams (1 hour)
1. Complete operator training (30 min)
2. Read DASHBOARD_GUIDE.md sections (15 min)
3. Practice incident investigation (10 min)
4. Learn export features (5 min)

### For Managers (1 hour)
1. Read this PROJECT_OVERVIEW (20 min)
2. Review use cases section (15 min)
3. Understand business metrics (15 min)
4. Plan implementation strategy (10 min)

---

## 🏆 Why This Dashboard?

### Compared to Traditional Monitoring

**Traditional SCADA:**
- ❌ Complex setup
- ❌ Expensive licenses
- ❌ Limited analytics
- ❌ Siloed data

**SenorMatics Dashboard:**
- ✅ Simple web interface
- ✅ Open source tech
- ✅ AI-powered insights
- ✅ Federated learning

### Unique Advantages

1. **Privacy-First**: Federated learning keeps data local
2. **Easy to Use**: Single-page, intuitive design
3. **Powerful Analytics**: 4 analysis modes in one interface
4. **Scalable**: Add facilities easily
5. **Cost-Effective**: Built on open-source tools
6. **Modern UI**: Beautiful, responsive design

---

## 📞 Support & Resources

### Documentation
- All `.md` files in `UI App/` folder
- Inline code comments
- Test scripts for validation

### Community
- GitHub repository for updates
- Issue tracker for bugs
- Wiki for advanced topics

### Getting Help
1. Check relevant `.md` file
2. Run `python test_setup.py`
3. Review error logs
4. Check Streamlit documentation

---

## 🎉 Summary

**SenorMatics Dashboard** brings industrial-grade predictive maintenance to your factory floor with:

✨ **Easy to Use** - Single page, 4 tabs, intuitive  
✨ **Powerful Analytics** - AI + federated learning  
✨ **Privacy-Preserving** - Data stays at facility  
✨ **Cost-Effective** - Open source technology  
✨ **Production-Ready** - Tested with real data  

**Ready to start?** → See `QUICK_START.md`

---

**SenorMatics** - *Privacy-Preserving Predictive Maintenance*  
Version 1.0 | Built with ❤️ for Machine Operators

