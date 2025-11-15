#!/bin/bash

echo "================================"
echo "SenorMatics Dashboard Launcher"
echo "================================"
echo ""
echo "Starting Streamlit dashboard..."
echo ""

cd "$(dirname "$0")"
streamlit run app.py

