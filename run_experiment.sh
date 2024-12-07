#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print error and exit
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Function to print success
print_success() {
    echo -e "${GREEN}$1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

# Check required files and directories
check_requirements() {

    # Check for required files
    required_files=(
        "run_pipeline.py"
        "config/config.yaml"
        "requirements.txt"
        "src/__init__.py"
        "src/model_loader.py"
        "src/dependency_graph.py"
        "src/importance_scorer.py"
        "src/pruning_env.py"
        "src/rl_agent.py"
        "src/metrics.py"
        "src/utils.py"
    )

    print_warning "Checking required files..."
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            error_exit "Required file not found: $file"
        fi
    done
    print_success "All required files found!"
}


# Set WandB API Key (replace 'your_api_key' with the actual key)
export WANDB_API_KEY="110550b358904e57dc28c9f18c684814e825f8e0"
if [ -z "$WANDB_API_KEY" ]; then
    error_exit "WANDB_API_KEY is not set. Please set it in the script or as an environment variable."
fi
# # Check if conda is installed
# if ! command -v conda &> /dev/null; then
#     error_exit "conda is not installed. Please install Anaconda or Miniconda first."
# fi

# Configuration
ENV_NAME="shrinker"
PYTHON_VERSION="3.10"
HF_TOKEN=""  # Will be set by user input

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    error_exit "NVIDIA GPU/CUDA is not detected. This experiment requires a GPU."
fi

# Check required files before proceeding
check_requirements

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Create and activate conda environment
# echo "Creating conda environment..."
# conda create -y -n $ENV_NAME python=$PYTHON_VERSION || error_exit "Failed to create conda environment"
# print_success "Conda environment created successfully!"

# Activate conda environment
# echo "Activating conda environment..."
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate $ENV_NAME || error_exit "Failed to activate conda environment"

# Install other requirements
# echo "Installing other requirements..."
# pip install -r requirements.txt || error_exit "Failed to install requirements"


# Clone and set up lm-evaluation-harness
# print_warning "Cloning lm-evaluation-harness repository..."
# if [ -d "lm-evaluation-harness" ]; then
#     echo "Repository already exists. Pulling latest changes..."
#     cd lm-evaluation-harness || error_exit "Failed to enter lm-evaluation-harness directory"
#     git pull || error_exit "Failed to pull latest changes"
# else
#     git clone https://github.com/EleutherAI/lm-evaluation-harness.git || error_exit "Failed to clone repository"
#     cd lm-evaluation-harness || error_exit "Failed to enter lm-evaluation-harness directory"
# fi


export HF_TOKEN=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


# Create necessary directories
mkdir -p experiments/results
mkdir -p config

# Function to monitor GPU usage
monitor_gpu() {
    while true; do
        nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,temperature.gpu --format=csv,noheader >> experiments/results/gpu_monitoring.csv
        sleep 10
    done
}


# Start GPU monitoring in background
echo "Starting GPU monitoring..."
monitor_gpu &
MONITOR_PID=$!

# Function to clean up
cleanup() {
    echo "Cleaning up..."
    kill $MONITOR_PID
    conda deactivate
}

# Set trap for cleanup
trap cleanup EXIT

# Run the experiment
echo "Starting the experiment..."
print_warning "This may take a while. GPU monitoring data will be saved in experiments/results/gpu_monitoring.csv"

# Create experiments directory if it doesn't exist
mkdir -p experiments/results

# Run with error handling and logging
{
    echo "=== Experiment Start: $(date) ===" 
    echo "GPU Information:"
    nvidia-smi
    echo "=== Running Pipeline ==="
    python run_pipeline.py --config config/config.yaml
} 2>&1 | tee experiments/results/experiment.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "Experiment completed successfully!"
    print_success "Results are saved in experiments/results/"
    print_success "Log file: experiments/results/experiment.log"
    print_success "GPU monitoring: experiments/results/gpu_monitoring.csv"
else
    error_exit "Experiment failed. Check experiments/results/experiment.log for details."
fi

# # Deactivate conda environment
# conda deactivate


print_success "All done!"

