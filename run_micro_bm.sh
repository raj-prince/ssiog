#!/bin/bash

# Set the Python interpreter to use (optional, but recommended)
PYTHON_INTERPRETER="python3"

# Set the path to your Python script
SCRIPT_PATH="training.py"  # Replace with the actual path

# Set any command-line arguments you want to pass to the script
ARGS=(
  "--prefix" "/usr/local/google/home/princer/gcs/Workload.0"
  "--prefix" "test"
  "--epochs" "2"
  "--steps" "4"
  "--sample-size" "4096"
  "--batch-size" "16"
#  "--background-threads" "8"
#  "--group-size" "2"
)

# Invoke the Python script with the specified interpreter and arguments
$PYTHON_INTERPRETER "$SCRIPT_PATH"