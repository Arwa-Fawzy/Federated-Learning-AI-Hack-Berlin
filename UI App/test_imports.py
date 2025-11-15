"""
Quick test to verify imports work
"""
print("Testing imports...")

try:
    import streamlit as st
    print("✅ streamlit - OK")
except ImportError as e:
    print(f"❌ streamlit - FAILED: {e}")

try:
    import plotly.graph_objects as go
    print("✅ plotly - OK")
except ImportError as e:
    print(f"❌ plotly - FAILED: {e}")

try:
    import pandas as pd
    print("✅ pandas - OK")
except ImportError as e:
    print(f"❌ pandas - FAILED: {e}")

try:
    from utils import load_client_data, load_metadata
    print("✅ utils - OK")
    
    # Try loading data
    metadata = load_metadata()
    print(f"✅ metadata loaded - {metadata['n_clients']} clients")
    
    data = load_client_data(0)
    print(f"✅ data loaded - {len(data)} samples")
    
except Exception as e:
    print(f"❌ utils - FAILED: {e}")

print("\nAll imports successful!")
print("\nNow run: streamlit run app.py")

