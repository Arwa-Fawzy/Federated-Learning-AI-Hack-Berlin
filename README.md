# SenorMatics Industrial Pump Sensor Dataset
### Real-World IoT Data for Federated Learning Research

**SenorMatics** provides industrial IoT sensor data from real pump operations. This dataset enables privacy-preserving machine learning research across distributed facilities without sharing sensitive operational data.

---

## 📊 About the Dataset

### Dataset Overview

SenorMatics has collected **220,320 sensor readings** from industrial centrifugal pumps operating in a continuous production environment over 5 months (April-August 2018).

**Dataset Size**: 118 MB (CSV format)  
**Sampling Rate**: 1 reading per minute  
**Duration**: ~153 days of continuous operation  
**Total Sensors**: 52 channels  

### Operational States

The pump system exhibits three distinct operational states:

| State | Description | Samples | Percentage |
|-------|-------------|---------|------------|
| **NORMAL** | Healthy operation within specifications | 205,836 | 93.4% |
| **RECOVERING** | Transient recovery after anomaly | 14,477 | 6.6% |
| **BROKEN** | Critical failure requiring shutdown | 7 | <0.01% |

---

## 🔬 Sensor Physics & Measurements

Our dataset captures multiple physical phenomena that characterize pump health and performance:

### 1. **Vibration Sensors** (Accelerometers)
**Physical Principle**: Measure mechanical oscillations caused by:
- Rotor imbalance
- Bearing wear and degradation
- Cavitation (vapor bubble formation/collapse)
- Misalignment between motor and pump shaft
- Resonance frequencies

**Why It Matters**: Vibration signatures reveal early mechanical failures before catastrophic damage occurs.

### 2. **Temperature Sensors** (Thermocouples/RTDs)
**Physical Principle**: Detect thermal changes from:
- Friction in bearings and seals
- Motor winding heat (electrical losses)
- Fluid temperature rise (energy dissipation)
- Ambient temperature variations

**Why It Matters**: Abnormal temperature rise indicates friction, inadequate lubrication, or electrical issues.

### 3. **Pressure Sensors** (Strain Gauges/Piezoelectric)
**Physical Principle**: Measure hydraulic forces:
- Discharge pressure (pump output)
- Suction pressure (inlet conditions)
- Differential pressure (pump head)
- Pressure pulsations (flow instabilities)

**Why It Matters**: Pressure deviations indicate cavitation, blockages, or impeller damage.

### 4. **Flow Rate Sensors** (Electromagnetic/Ultrasonic)
**Physical Principle**: Measure fluid velocity through:
- Electromagnetic induction (Faraday's law)
- Ultrasonic time-of-flight
- Doppler shift

**Why It Matters**: Flow anomalies reveal pump efficiency loss, leakage, or system blockages.

### 5. **Rotational Speed (RPM)** (Optical/Magnetic Encoders)
**Physical Principle**: Track shaft rotation via:
- Optical reflection from encoded disk
- Magnetic field changes from gear teeth
- Hall effect sensors

**Why It Matters**: Speed variations indicate motor control issues, load changes, or mechanical drag.

### 6. **Acoustic/Sound Sensors** (Microphones)
**Physical Principle**: Capture sound waves from:
- Cavitation bubble collapse (high-frequency pops)
- Bearing noise (grinding, clicking)
- Turbulent flow patterns
- Structural resonances

**Why It Matters**: Acoustic signatures detect problems invisible to other sensors.

### Physical Failure Mechanisms Captured

1. **Cavitation**: Low suction pressure → vapor bubbles → impeller erosion
2. **Bearing Wear**: Friction → heat + vibration → eventual seizure
3. **Seal Leakage**: Degradation → fluid loss → pressure drop
4. **Impeller Damage**: Corrosion/erosion → flow reduction + vibration
5. **Motor Overheating**: Electrical overload → winding damage → failure

---

## 📁 Dataset Structure

### File Organization

```
data/
└── sensor.csv                          # Main dataset (220,320 rows × 55 columns)

federated_data/
├── hybrid/                             # Hybrid strategy (recommended)
│   ├── client_0.csv                   # Small facility (5% of data)
│   ├── client_1.csv                   # Large facility (35% of data)
│   ├── client_2.csv - client_4.csv    # Medium facilities
│   └── client_metadata.json           # Dataset statistics
│
├── clustering/                         # Operating condition-based split
│   └── client_*.csv                   # 5 clients grouped by behavior
│
└── dirichlet_high/                    # High heterogeneity split
    └── client_*.csv                   # 5 clients with label imbalance
```

### Data Columns

| Column Type | Count | Examples |
|-------------|-------|----------|
| **Timestamp** | 1 | Date and time of measurement |
| **Sensor Data** | 52 | `sensor_00` through `sensor_51` (continuous values) |
| **Label** | 1 | `machine_status` (NORMAL/RECOVERING/BROKEN) |

**Note**: `sensor_15` is empty (sensor malfunction during collection period).

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ramdhiwakar1/fl-dist-hack-sensors.git
cd fl-dist-hack-sensors

# Install dependencies
pip install -r requirements.txt
```

### 2. Explore the Data

```bash
# View dataset statistics
python explore_data.py
```

### 3. Generate Federated Datasets

```bash
# Create heterogeneous client datasets
python heterogeneous_data_generator.py
```

This generates datasets simulating multiple facilities with:
- Different operating conditions (high-load vs. low-load)
- Varying sensor quality (calibrated vs. degraded)
- Different failure patterns (new vs. aging equipment)
- Unequal data sizes (small vs. large facilities)

### 4. Visualize Heterogeneity

```bash
# Generate analysis plots
python visualize_heterogeneity.py
```

Creates visualizations showing:
- Data distribution across simulated facilities
- Sensor measurement differences
- Statistical heterogeneity metrics

---

## 📖 Dataset Applications

### Predictive Maintenance
Build models to predict pump failures hours or days in advance, enabling:
- Scheduled maintenance (reduce downtime)
- Parts inventory optimization
- Extended equipment lifespan

### Anomaly Detection
Identify unusual operating patterns indicating:
- Incipient failures
- Process inefficiencies
- Sensor malfunctions

### Federated Learning Research
Test privacy-preserving machine learning across:
- Multiple manufacturing sites
- Different equipment vintages
- Varying operational conditions

### Physics-Informed ML
Combine sensor data with physical models:
- Fluid dynamics equations
- Thermodynamic principles
- Mechanical stress analysis

---

## 🏭 Industrial Context

### Why Multiple Facilities Need Federated Learning

Real industrial scenarios involve:

**Facility Heterogeneity**:
- Factory A: High-speed continuous operation (24/7)
- Factory B: Batch processing with frequent starts/stops
- Factory C: Different pump models and configurations
- Factory D: Varying maintenance practices

**Data Privacy Requirements**:
- Operational data reveals production capacity
- Failure rates are competitively sensitive
- Regulatory compliance (GDPR, industry standards)
- Intellectual property protection

**Collaborative Benefits**:
- Learn from diverse operating conditions
- Improve model robustness
- Reduce data collection costs
- Share knowledge without sharing data

---

## 📊 Data Quality & Characteristics

### Missing Data
Some sensors have intermittent readings:
- `sensor_15`: 100% missing (sensor failure)
- `sensor_50`: 35% missing (communication issues)
- `sensor_00`: 4.6% missing (calibration periods)
- Most sensors: <1% missing

**Reason**: Real-world sensors experience failures, communication drops, and maintenance windows.

### Sampling Considerations
- **Temporal Correlation**: Consecutive readings are related (time-series)
- **Sensor Correlation**: Temperature and vibration often correlate
- **Class Imbalance**: Normal operation dominates (93.4%)
- **Seasonal Variation**: Data spans spring and summer months

### Data Artifacts
Real industrial data includes:
- Calibration drift over time
- Sensor noise and measurement uncertainty
- Communication delays and buffering
- Environmental interference

---

## 🔬 Research Opportunities

This dataset enables research in:

1. **Federated Learning Algorithms**
   - Non-IID data handling
   - Communication efficiency
   - Privacy preservation

2. **Time-Series Analysis**
   - LSTM/GRU networks
   - Transformer architectures
   - Temporal convolutional networks

3. **Anomaly Detection**
   - Autoencoders
   - One-class SVM
   - Isolation forests

4. **Physics-Informed Neural Networks**
   - Incorporate fluid dynamics equations
   - Thermodynamic constraints
   - Conservation laws

5. **Transfer Learning**
   - Cross-facility model adaptation
   - Domain adaptation techniques
   - Few-shot learning

---

## 📄 Data License

### License
This dataset is a synthetic dataset for research and educational purposes.


### Original Data Source
Dataset adapted from: [Kaggle - Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)

## 🛠️ Technical Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Dependencies
```
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
seaborn >= 0.13.0
scipy >= 1.10.0
```

### Installation
```bash
pip install -r requirements.txt
```

---

## 📚 Additional Resources

### Included Documentation
- `README.md` - This file
- `USAGE_GUIDE.md` - Detailed usage instructions
- `PROJECT_SUMMARY.md` - Technical overview
- `config.yaml` - Configuration options

### Example Scripts
- `download_dataset.py` - Download data from source
- `explore_data.py` - Dataset exploration
- `heterogeneous_data_generator.py` - Create federated splits
- `visualize_heterogeneity.py` - Generate analysis plots
- `federated_learning_example.py` - FL implementation demo
- `quick_start.py` - One-command setup

---

## ⚙️ Dataset Generation Strategies

### 1. Hybrid Strategy (Recommended)
Combines operating conditions + label imbalance + quantity skew
- **Use**: Most realistic multi-facility simulation

### 2. Clustering Strategy
Groups data by operating patterns (high/low load, steady/variable)
- **Use**: Simulate facilities with different pump applications

### 3. Dirichlet Strategy
Creates controlled label imbalance across clients
- **Use**: Research on non-IID data handling

### 4. Temporal Strategy
Splits data by time periods
- **Use**: Study seasonal or operational shift effects
- 
**SenorMatics** - Enabling Privacy-Preserving Industrial AI

*Data-driven insights, privacy-first approach.*

# Local Training: 
## 🏋️‍♂️ Federated Training Using 1D Convolutional Autoencoder

We use a **1D Convolutional Autoencoder (CAE)** in a federated learning setup for anomaly detection in pump sensor data. The approach works as follows:  

- Each client (e.g., `sensor_0`, `sensor_1`, `sensor_2`) trains the CAE **locally** on its time-series sensor sequences.  
- The CAE consists of **1D convolutional layers** to capture temporal patterns and **deconvolutional layers** to reconstruct the input sequence.  
- The model minimizes **reconstruction error** to learn normal operating behavior.  
- After local training, each client sends its **model weights to a central aggregator** (server).  
- The server combines the weights using **federated averaging (`FedAvg`)** to create a **global model**.  
- The global model can detect abnormal pump behavior (e.g., cavitation, bearing wear, flow irregularities) **without sharing raw data**, enabling **privacy-preserving predictive maintenance** across heterogeneous facilities.


---

