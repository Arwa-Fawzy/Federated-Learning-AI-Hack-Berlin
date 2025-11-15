# 🏭 SenorMatics Predictive Maintenance Dashboard

A single-page Streamlit dashboard for monitoring industrial pump health across multiple facilities using federated learning.

## 📋 Features

### Real-time Monitoring
- **Live Health Scores**: Instant health metrics for each facility
- **Status Tracking**: Real-time NORMAL/RECOVERING/BROKEN state monitoring
- **Active Alerts**: Immediate anomaly detection and warnings
- **Sensor Grid**: 52-sensor visualization with heatmaps

### Sensor Analysis
- **Individual Sensor Diagnostics**: Deep dive into each sensor's performance
- **Statistical Analysis**: Mean, std dev, distribution plots
- **Anomaly Detection**: Automated outlier identification (3σ threshold)
- **Missing Data Tracking**: Data quality monitoring

### Historical Analysis
- **Time-Series Explorer**: Interactive date range selection
- **Multi-Sensor Comparison**: Compare up to 6 sensors simultaneously
- **Status Distribution**: Historical state analysis
- **Trend Visualization**: Identify patterns over time

### AI-Powered Insights
- **Federated Learning Integration**: Privacy-preserving model performance
- **Anomaly Scoring**: Reconstruction error tracking
- **Feature Importance**: Identify critical sensors
- **Predictive Maintenance**: Failure risk assessment

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+ required
python --version
```

### Installation

1. **Navigate to UI App directory:**
```bash
cd "UI App"
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Ensure data files exist:**
The app expects data in `../federated_data/hybrid/`:
- `client_0.csv` through `client_4.csv`
- `client_metadata.json`

4. **Run the dashboard:**
```bash
streamlit run app.py
```

5. **Open browser:**
The app will automatically open at `http://localhost:8501`

## 📊 Dashboard Tabs

### 1. 📊 Real-time Monitor
- Current facility status and health score
- Live sensor readings (last 500 samples)
- Active alerts panel
- Key performance indicators (KPIs)

### 2. 🔬 Sensor Analysis
- 20-sensor heatmap with normalized values
- Individual sensor deep-dive:
  - Current value, mean, std dev
  - Historical trends
  - Distribution histogram
  - Box plot for outliers
- Missing data analysis

### 3. 📈 Historical Data
- Custom time range selection (index-based)
- Status distribution in selected range
- Multi-sensor comparison plots
- Sample statistics

### 4. 🤖 AI Insights
- Model performance metrics:
  - Global model accuracy
  - Local model accuracy
  - Anomaly detection rate
- Anomaly score timeline
- Feature importance ranking
- Privacy-preserved federated insights

## 🎨 UI Elements

### Color Coding
- 🟢 **Green**: NORMAL status, healthy operations
- 🟡 **Yellow**: RECOVERING status, transitional state
- 🔴 **Red**: BROKEN status, critical alerts

### Key Metrics
- **Health Score**: 0-100 scale (composite metric)
- **Uptime**: % of time in NORMAL state
- **Active Alerts**: Number of current anomalies
- **Active Sensors**: Working sensors out of 52

## 📁 File Structure

```
UI App/
├── app.py                  # Main Streamlit application
├── utils.py                # Data loading and processing utilities
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Data Path
Edit `utils.py` to change data directory:
```python
DATA_DIR = BASE_DIR / "federated_data" / "hybrid"
```

### Refresh Settings
Control auto-refresh in sidebar:
- Interval: 30-300 seconds
- Manual refresh button available

### Facility Selection
Use sidebar dropdown to switch between 5 facilities (0-4)

## 📈 Use Cases

### For Machine Operators
- Monitor real-time pump health across multiple facilities
- Receive instant alerts on anomalies
- Track sensor performance and identify issues early

### For Maintenance Teams
- Historical analysis for maintenance planning
- Identify recurring failure patterns
- Export data for detailed reports

### For Facility Managers
- Fleet-wide health overview
- Performance comparison across facilities
- Data-driven maintenance scheduling

### For Data Scientists
- Federated model performance monitoring
- Feature importance analysis
- Anomaly detection validation

## 🛠️ Troubleshooting

### "Failed to load data"
**Solution**: Ensure `federated_data/hybrid/` exists with all client CSV files and metadata.json

### "No module named 'streamlit'"
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Port already in use
**Solution**: Specify different port: `streamlit run app.py --server.port 8502`

### Slow performance
**Solution**: 
- Reduce number of samples displayed (modify `tail(500)` to `tail(200)`)
- Disable auto-refresh
- Cache is enabled by default with 5-minute TTL

## 📊 Data Requirements

### Expected CSV Format
```
timestamp, sensor_00, sensor_01, ..., sensor_51, machine_status
2018-04-01 00:00:00, 2.465, 47.09, ..., 201.39, NORMAL
```

### Metadata Format (JSON)
```json
{
  "n_clients": 5,
  "total_samples": 220320,
  "clients": {
    "0": {
      "samples": 11016,
      "sensors": 52,
      "status_distribution": {...},
      "missing_rate": 0.0198,
      "file": "client_0.csv"
    }
  }
}
```

## 🔒 Privacy Features

- **Federated Learning**: Models trained locally, only weights shared
- **No Raw Data Sharing**: Each facility's data stays on-premises
- **Privacy-Preserved Insights**: Global model benefits without data exposure
- **Differential Privacy**: Optional noise addition for enhanced privacy

## 📥 Export Features

### CSV Export
- Export current facility data
- Filename: `facility_{id}_report_{date}.csv`
- Includes all sensors and timestamps

### Summary Report
- Health score
- Status distribution
- Anomaly count
- Data quality metrics

## 🚀 Performance Tips

1. **Data Caching**: Enabled with 5-minute TTL (300s)
2. **Sample Limiting**: Only last 500-1000 samples plotted for speed
3. **Sensor Selection**: Heatmap limited to 20 sensors by default
4. **Lazy Loading**: Historical data loaded on-demand

## 📝 Future Enhancements

- [ ] Real-time data streaming integration
- [ ] Email/SMS alert notifications
- [ ] Multi-user authentication
- [ ] Custom alert threshold configuration
- [ ] PDF report generation
- [ ] Mobile-responsive design
- [ ] WebSocket for live updates
- [ ] Integration with maintenance management systems

## 🤝 Support

For issues or questions:
1. Check troubleshooting section
2. Verify data files exist and format is correct
3. Check console logs for error messages
4. Ensure all dependencies are installed

## 📄 License

Part of the SenorMatics Federated Learning project.

## 🏆 Credits

**Built for**: Machine operators and facility managers  
**Purpose**: Predictive maintenance monitoring  
**Technology**: Streamlit, Plotly, Federated Learning  
**Data**: Industrial IoT pump sensor readings (220K+ samples)

---

**SenorMatics v1.0** - Privacy-Preserving Predictive Maintenance  
*Enabling collaborative AI without compromising data privacy*

