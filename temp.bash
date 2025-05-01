#!/bin/bash

echo "Starting anomaly experiments..."
python run_anomaly_experiments.py

echo "Starting main experiments..."
python run_experiments.py

echo "All experiments completed."
