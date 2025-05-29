# MAMBA-PT: Particle Tracking with State Space Models

This repository implements a MAMBA-based architecture for particle tracking in High Energy Physics, inspired by the TrackingBERT paper but using state space models instead of transformers.

## Overview

This project adapts the TrackingBERT approach to use the MAMBA architecture, treating particle tracks as sequences and using state space models for efficient sequence modeling. The model is trained on the TrackML dataset to predict:
1. Masked Detector Modules (MDM) - predicting missing hits in a track
2. Next Track Prediction (NTP) - determining momentum relationships between tracks

## Requirements

- Python 3.8+
- PyTorch 1.13+
- CUDA-capable GPU with at least 12GB memory
- ~100GB disk space for the full TrackML dataset

## Installation

```bash
# Clone the repository
git clone https://github.com/SAURABH-RAI1729/mamba-pt.git
cd mamba-pt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

1. Download the TrackML dataset from [Kaggle](https://www.kaggle.com/c/trackml-particle-identification/data)
2. Extract the files into the `data/` directory:
   ```
   data/
   ├── train_1/
   ├── train_2/
   ├── train_3/
   ├── train_4/
   ├── train_5/
   └── detectors.csv
   ```

3. Preprocess the data:
   ```bash
   python scripts/preprocess_data.py --data_dir data --output_dir data/processed
   ```

## Training

### Basic Training
```bash
python train.py --config configs/base_config.yaml
```

### Training with Custom Parameters
```bash
python train.py \
    --data_dir data/processed \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 200 \
    --d_model 64 \
    --n_layers 4 \
    --gpu_memory_limit 12000
```

### Multi-GPU Training (if available)
```bash
python train.py --config configs/base_config.yaml --gpus 0,1
```

## Model Architecture

The MAMBA-PT uses a state space model architecture:

- **Input Tokenization**: Detector module IDs are tokenized and embedded
- **MAMBA Blocks**: Multiple layers of MAMBA blocks with:
  - State space models for sequence modeling
  - Convolutional projections
  - Gated linear units
  - Layer normalization
- **Task Heads**: 
  - MDM head for masked module prediction
  - NTP head for momentum comparison

## Evaluation

Run evaluation on test data:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --test_dir data/test
```

## Memory Optimization

For 12GB GPU constraints, the following optimizations are implemented:
- Gradient checkpointing
- Mixed precision training
- Dynamic batch sizing
- Model sharding for larger configurations

## Results

| Metric | Value |
|--------|-------|
| MDM Accuracy | 89.8% |
| NTP Accuracy | 85.2% |
| Inference Speed | 1.9x faster than transformer baseline |
The performance of this model is still sub-optimal. Nevertheless, the approach is interesting in its own rights and may be improved later.
## Project Structure

```
mamba-pt/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   └── base_config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mamba_block.py
│   │   ├── tracking_mamba.py
│   │   └── tokenizer.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── trackml_dataset.py
│   │   └── preprocessing.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── losses.py
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
├── scripts/
│   ├── preprocess_data.py
│   └── download_data.sh
├── notebooks/ # Not Available Currently
│   └── visualization.ipynb
├── tests/
│   ├── test_model.py
│   └── test_dataset.py
├── train.py
├── evaluate.py
└── checkpoints/
```

## Acknowledgments

This work is based on the TrackingBERT paper by Huang et al. and uses the MAMBA architecture for efficient sequence modeling.

## License

MIT License - see LICENSE file for details
