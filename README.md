# iShrink: Make 1B Models Even Smaller and Faster

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2401.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2401.XXXXX)

This repository contains the official implementation of "iShrink: Making 1B Models Even Smaller and Faster". iShrink is a structured pruning approach that effectively compresses 1B-parameter language models while maintaining their performance and improving efficiency.

## Features

- 🚀 15-19% model compression with minimal performance degradation
- ⚡️ 17-24% throughput improvement
- 🎯 Targeted pruning for both attention and MLP layers
- 🔄 LoRA fine-tuning for performance recovery
- ⏱️ Fast pruning (5 minutes) and efficient fine-tuning (2.5 hours)
- 💻 Single GPU (A100) implementation

## Important Paths
```
LLM-COMPRESSION/
├── config/
│   └── config.yaml           # Main configuration file - Set model and pruning parameters
├── src/
│   ├── pruner/              # Core pruning implementations
│   │   ├── adaptive_pruner.py
│   │   ├── width_pruner.py
│   │   └── depth_pruner.py
│   ├── evaluate.py          # Evaluation script for all model variants
│   └── importance_scorer.py  # Importance scoring implementation
├── experiments/             # Results and checkpoints directory
│   └── results/
│       └── [MODEL_NAME]/    # Model-specific results
├── run_experiment.sh        # Main pruning pipeline script
└── finetuner.sh            # Fine-tuning script for pruned models
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DannyMeb/iShrink.git
cd iShrink
```

2. Create a conda environment:
```bash
conda create -n ishrink python=3.8
conda activate ishrink
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Quick Start Guide

### 1. Configuration Setup
Edit `config/config.yaml` to set your model and parameters:
```yaml
model:
    name: "tiiuae/Falcon3-1B-Instruct"  # Model name from HuggingFace
    device: "cuda"                      # Device to use
    precision: "float16"                # Model precision

pruning:
    width_pruning:
        enabled: true
        attention_sparsity: 0.0         # Attention head pruning ratio
        mlp_sparsity: 0.14             # MLP layer pruning ratio
    
    depth_pruning:
        enabled: true
        num_layers_to_prune: 2          # Number of layers to remove
        keep_first_layers: 4            # Keep first N layers
        keep_last_layers: 2             # Keep last N layers
```

### 2. Running Experiments

First, set your HuggingFace token:
```bash
export HF_TOKEN="your_huggingface_token"
```

Run the pruning pipeline:
```bash
bash run_experiment.sh
```
This will:
- Run zero-out analysis
- Perform structured pruning
- Save pruned model in `experiments/results/[MODEL_NAME]/pruned_model/`

### 3. Model Evaluation
To evaluate any model variant:
```bash
python3 src/evaluate.py
```
Follow the interactive prompts to select:
- Model type (Original/Pruned/Finetuned)
- Specific model variant
- Evaluation metrics

### 4. Fine-tuning
To fine-tune a pruned model:
```bash
bash finetuner.sh
```
Fine-tuned models will be saved in `experiments/results/[MODEL_NAME]/finetuned_models/`

## Supported Models

Currently tested on:
- Llama-3.2-1B
- Llama-3.2-1B-Instruct
- Falcon3-1B
- Falcon3-1B-Instruct

## Results Directory Structure
```
experiments/results/[MODEL_NAME]/
├── pruned_model/           # Pruned model checkpoint
├── finetuned_models/       # LoRA fine-tuned models
└── evaluations/            # Evaluation results
    ├── evaluation_results_latest.json
    └── evaluation_results_[TIMESTAMP].json
```

## Troubleshooting

1. Out of Memory Errors:
   - Reduce batch size in config.yaml
   - Enable gradient checkpointing
   - Use float16 precision

2. Model Loading Issues:
   - Ensure HF_TOKEN is set correctly
   - Check model name in config.yaml
   - Verify internet connection for model download

3. Evaluation Errors:
   - Ensure model paths are correct
   - Check GPU memory availability
   - Verify model checkpoints exist

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

## Citation

<!-- ```bibtex
@article{ishrink2024,
  title={iShrink: Making 1B Models Even Smaller and Faster},
  author={Your Name},
  journal={arXiv preprint arXiv:2401.XXXXX},
  year={2024}
} -->
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Issues: Please use GitHub issues
- Email: Daniel.Gebre@mbzuai.ac.ae
- LinkedIn: [@DannMeb](linkedin.com/in/danmeb)