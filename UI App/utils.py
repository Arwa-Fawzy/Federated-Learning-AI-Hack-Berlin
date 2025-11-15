"""
Utility functions for SenorMatics Predictive Maintenance Dashboard
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "federated_data" / "hybrid"

def load_client_data(client_id: int) -> pd.DataFrame:
    """
    Load data for a specific client/facility
    
    Args:
        client_id: Client ID (0-4)
        
    Returns:
        DataFrame with sensor data
    """
    file_path = DATA_DIR / f"client_{client_id}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Client data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Drop unnamed index column if exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    return df

def load_metadata() -> dict:
    """
    Load client metadata
    
    Returns:
        Dictionary with metadata for all clients
    """
    metadata_path = DATA_DIR / "client_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def calculate_health_score(data: pd.DataFrame) -> float:
    """
    Calculate overall health score for a facility
    
    Args:
        data: Facility sensor data
        
    Returns:
        Health score (0-100)
    """
    # Base score on machine status distribution
    status_counts = data['machine_status'].value_counts()
    total = len(data)
    
    normal_pct = status_counts.get('NORMAL', 0) / total
    recovering_pct = status_counts.get('RECOVERING', 0) / total
    broken_pct = status_counts.get('BROKEN', 0) / total
    
    # Weighted score
    health_score = (normal_pct * 100) + (recovering_pct * 50) + (broken_pct * 0)
    
    # Adjust for data quality
    sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
    if sensor_cols:
        missing_rate = data[sensor_cols].isnull().mean().mean()
        health_score = health_score * (1 - missing_rate * 0.5)  # Penalize missing data
    
    return min(100, max(0, health_score))

def detect_anomalies(data: pd.DataFrame, threshold_std: float = 3.0) -> list:
    """
    Detect anomalies in sensor data
    
    Args:
        data: Facility sensor data
        threshold_std: Number of standard deviations for anomaly threshold
        
    Returns:
        List of anomaly dictionaries
    """
    anomalies = []
    sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
    
    # Use most recent data (last 100 samples)
    recent_data = data[sensor_cols].tail(100)
    
    for sensor in sensor_cols[:20]:  # Check first 20 sensors
        if sensor not in recent_data.columns:
            continue
        
        sensor_data = recent_data[sensor].dropna()
        
        if len(sensor_data) < 10:
            continue
        
        mean = sensor_data.mean()
        std = sensor_data.std()
        
        # Check latest value
        if len(sensor_data) > 0:
            latest = sensor_data.iloc[-1]
            z_score = abs((latest - mean) / (std + 1e-8))
            
            if z_score > threshold_std:
                severity = "HIGH" if z_score > 4 else "MEDIUM"
                anomalies.append({
                    'sensor': sensor,
                    'type': 'Statistical Anomaly',
                    'severity': severity,
                    'z_score': z_score,
                    'current_value': latest,
                    'expected_range': f"{mean - 2*std:.2f} - {mean + 2*std:.2f}",
                    'message': f"Value {latest:.2f} is {z_score:.1f}σ from mean ({mean:.2f})"
                })
    
    # Check for machine status anomalies
    if 'machine_status' in data.columns:
        recent_status = data['machine_status'].tail(50)
        recovering_count = (recent_status == 'RECOVERING').sum()
        broken_count = (recent_status == 'BROKEN').sum()
        
        if recovering_count > 5:
            anomalies.append({
                'sensor': 'machine_status',
                'type': 'Status Alert',
                'severity': 'MEDIUM',
                'z_score': None,
                'current_value': recovering_count,
                'expected_range': '< 5',
                'message': f"High recovery rate detected: {recovering_count} RECOVERING states in last 50 samples"
            })
        
        if broken_count > 0:
            anomalies.append({
                'sensor': 'machine_status',
                'type': 'Critical Alert',
                'severity': 'HIGH',
                'z_score': None,
                'current_value': broken_count,
                'expected_range': '0',
                'message': f"⚠️ BROKEN state detected {broken_count} times in recent history"
            })
    
    return sorted(anomalies, key=lambda x: 1 if x['severity'] == 'HIGH' else 2, reverse=False)

def get_sensor_statistics(data: pd.DataFrame, sensor: str) -> dict:
    """
    Get statistical summary for a sensor
    
    Args:
        data: Facility sensor data
        sensor: Sensor column name
        
    Returns:
        Dictionary with statistics
    """
    if sensor not in data.columns:
        return {
            'current': 0,
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'missing_pct': 100,
            'count': 0
        }
    
    sensor_data = data[sensor]
    
    return {
        'current': sensor_data.iloc[-1] if len(sensor_data) > 0 and not pd.isna(sensor_data.iloc[-1]) else 0,
        'mean': sensor_data.mean(),
        'std': sensor_data.std(),
        'min': sensor_data.min(),
        'max': sensor_data.max(),
        'missing_pct': (sensor_data.isnull().sum() / len(sensor_data)) * 100,
        'count': sensor_data.count()
    }

def calculate_time_to_failure(data: pd.DataFrame) -> dict:
    """
    Estimate time to failure (simplified model)
    
    Args:
        data: Facility sensor data
        
    Returns:
        Dictionary with TTF prediction
    """
    # Simple heuristic based on recent status
    recent_data = data.tail(100)
    
    if 'machine_status' not in recent_data.columns:
        return {
            'days': None,
            'confidence': 0,
            'risk_level': 'UNKNOWN'
        }
    
    normal_count = (recent_data['machine_status'] == 'NORMAL').sum()
    recovering_count = (recent_data['machine_status'] == 'RECOVERING').sum()
    
    normal_ratio = normal_count / len(recent_data)
    
    if normal_ratio > 0.95:
        return {
            'days': '>30',
            'confidence': 0.85,
            'risk_level': 'LOW'
        }
    elif normal_ratio > 0.85:
        return {
            'days': '14-30',
            'confidence': 0.70,
            'risk_level': 'MEDIUM'
        }
    else:
        return {
            'days': '<14',
            'confidence': 0.60,
            'risk_level': 'HIGH'
        }

def get_critical_sensors(data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Identify most critical sensors based on variability
    
    Args:
        data: Facility sensor data
        top_n: Number of top sensors to return
        
    Returns:
        DataFrame with sensor names and importance scores
    """
    sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
    
    importance_scores = {}
    
    for sensor in sensor_cols:
        if sensor in data.columns:
            sensor_data = data[sensor].dropna()
            
            if len(sensor_data) > 10:
                # Calculate coefficient of variation as importance metric
                cv = sensor_data.std() / (abs(sensor_data.mean()) + 1e-8)
                importance_scores[sensor] = abs(cv)
    
    # Convert to DataFrame
    importance_df = pd.DataFrame(
        list(importance_scores.items()),
        columns=['Sensor', 'Importance']
    ).sort_values('Importance', ascending=False).head(top_n)
    
    return importance_df

def prepare_data_for_export(data: pd.DataFrame, facility_id: int, metadata: dict) -> dict:
    """
    Prepare data summary for export
    
    Args:
        data: Facility sensor data
        facility_id: Facility ID
        metadata: Metadata dictionary
        
    Returns:
        Dictionary with export data
    """
    sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
    
    export_data = {
        'facility_id': facility_id,
        'total_samples': len(data),
        'date_range': {
            'start': data['timestamp'].iloc[0] if 'timestamp' in data.columns else 'N/A',
            'end': data['timestamp'].iloc[-1] if 'timestamp' in data.columns else 'N/A'
        },
        'health_score': calculate_health_score(data),
        'status_distribution': data['machine_status'].value_counts().to_dict() if 'machine_status' in data.columns else {},
        'active_sensors': len(sensor_cols),
        'anomalies': len(detect_anomalies(data)),
        'data_quality': {
            'missing_rate': data[sensor_cols].isnull().mean().mean() * 100,
            'completeness': (1 - data[sensor_cols].isnull().mean().mean()) * 100
        }
    }
    
    return export_data

def validate_data_files():
    """
    Validate that all required data files exist
    
    Returns:
        Tuple of (bool, list of missing files)
    """
    missing_files = []
    
    # Check metadata
    metadata_path = DATA_DIR / "client_metadata.json"
    if not metadata_path.exists():
        missing_files.append(str(metadata_path))
    
    # Check client files
    for i in range(5):
        client_path = DATA_DIR / f"client_{i}.csv"
        if not client_path.exists():
            missing_files.append(str(client_path))
    
    return len(missing_files) == 0, missing_files

