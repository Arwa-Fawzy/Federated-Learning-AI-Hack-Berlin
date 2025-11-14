"""
Explore the pump sensor dataset
"""
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data/sensor.csv')

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"\nShape: {df.shape}")
print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")

print("\n" + "="*80)
print("COLUMN NAMES AND TYPES")
print("="*80)
print(df.dtypes)

print("\n" + "="*80)
print("FIRST FEW ROWS")
print("="*80)
print(df.head(10))

print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)
print(df.describe())

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
print(df.isnull().sum())

print("\n" + "="*80)
print("UNIQUE VALUES PER COLUMN")
print("="*80)
for col in df.columns:
    n_unique = df[col].nunique()
    print(f"{col:30s}: {n_unique:>10,} unique values")

# Check if there are any categorical columns
print("\n" + "="*80)
print("CATEGORICAL COLUMNS")
print("="*80)
for col in df.columns:
    if df[col].dtype == 'object' or df[col].nunique() < 20:
        print(f"\n{col}:")
        print(df[col].value_counts())

print("\n" + "="*80)
print("DATA SAVED FOR INSPECTION")
print("="*80)
print(f"Dataset location: data/sensor.csv")
print(f"Total memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

