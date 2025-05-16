# Echo State Transformer (EST)

This repository contains the implementation of Echo State Transformer (EST), a novel neural network architecture that combines principles from Echo State Networks and Transformer models. The code allows for comprehensive evaluation of EST against other sequential architectures (Transformers, LSTMs, GRUs) on the STREAM benchmark tasks.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Examples](#examples)
- [Experiment Details](#experiment-details)
  - [STREAM Benchmark](#stream-benchmark)
  - [Hyperparameter Search](#hyperparameter-search)
- [Results Analysis](#results-analysis)
- [Repository Structure](#repository-structure)

## Introduction

The Echo State Transformer (EST) combines Echo State Network principles with Transformer-style attention mechanisms to create a novel architecture for sequential data processing. This implementation provides tools to evaluate its performance across various sequential tasks and compare it to other established architectures.

## Installation

Follow these steps to set up the environment:

```bash
# Clone the repository
git clone https://github.com/anonymous/echo-state-transformer.git
cd echo-state-transformer

# Create a virtual environment
python -m venv est_env

# Activate the virtual environment
# On Windows
est_env\Scripts\activate
# On Linux/Mac
source est_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

To run the evaluation with default settings (all models, all tasks, CPU execution):

```bash
python run.py
```

### Command Line Arguments

The `run.py` script supports several command-line arguments to customize evaluation:

| Argument | Description | Default |
|----------|-------------|---------|
| `--seeds` | Random seeds for reproducibility | 0 |
| `--device` | Computation device (`cpu` or `cuda`) | `cpu` |
| `--exp_id` | Experiment ID (used for results folder naming) | Current datetime |
| `--size` | Model size (`1k`, `10k`, `100k`, `1M`, or `all`) | `all` |
| `--layers` | Number of layers in models | 1 |
| `--tasks` | Tasks to evaluate (`all` or specific task names) | `all` |
| `--models` | Models to evaluate (`all` or specific model names) | `all` |
| `--learning_rates` | Learning rates to use | 1e-3 |
| `--weight_decays` | Weight decay parameters | 1e-5 |

### Examples

Run evaluation on specific tasks with specific models:

```bash
python run.py --tasks copy_task sorting_problem --models Transformer EST
```

Run evaluation with GPU acceleration and specific model size:

```bash
python run.py --device cuda --size 10k
```

Run evaluation with multiple seeds and learning rates:

```bash
python run.py --seeds 0 1 2 --learning_rates 0.001 0.0003 0.0001
```

Run full evaluation as described in the paper:

```bash
python run.py --seeds 0 1 2 3 4 --learning_rates 0.01 0.003 0.001 0.0003 0.0001 --weight_decays 0.01 --device cuda
```

## Experiment Details

### STREAM Benchmark

The STREAM benchmark consists of 12 sequential tasks designed to evaluate different aspects of sequential model capabilities:

1. **Simple Memory**: Tasks testing the model's ability to remember information over time
2. **Signal Processing**: Tasks evaluating signal transformation capabilities
3. **Long-Term Dependencies**: Tasks requiring retention of information across extended sequences
4. **Information Manipulation**: Tasks requiring complex transformation and manipulation of sequence information

These tasks are configurable in difficulty and complexity, allowing for comprehensive evaluation of sequential model capabilities. The configuration used for those tasks is available in `./libs/stream_evals`.

### Hyperparameter Search

Our experiments compared four model architectures:
- Echo State Transformers (EST) - four different structural configurations
- Transformers (Decoder-only) - four different structural configurations
- Long Short-Term Memory (LSTM) - one configuration
- Gated Recurrent Units (GRU) - one configuration

Each architecture was tested with four parameter sizes:
- 1,000 parameters (1k)
- 10,000 parameters (10k)
- 100,000 parameters (100k)
- 1,000,000 parameters (1M)

The following learning rates were systematically evaluated:
- 0.01
- 0.003
- 0.001
- 0.0003
- 0.0001

Weight decay was fixed at 0.01 based on preliminary hyperparameter search.

For each configuration, we used five different random seeds (0, 1, 2, 3, 4) to assess result variability.

## Results Analysis

The repository includes a Jupyter notebook `explore_results.ipynb` for analyzing and visualizing the experimental results. This notebook reproduces the tables and plots presented in the paper, using result data stored in the `./exp-results` directory.

To run the analysis:

```bash
jupyter notebook explore_results.ipynb
```

## Repository Structure

- `run.py`: Main script for running experiments
- `libs/`: Contains implementation of all models and evaluation tasks
  - `TransformerDecoderOnly.py`: Implementation of the Transformer model
  - `EST.py`: Implementation of the Echo State Transformer model
  - `LSTM.py`: Implementation of the LSTM model
  - `GRU.py`: Implementation of the GRU model
  - `stream_tasks.py`: Implementation of the STREAM benchmark tasks
  - `stream_evals.py`: Evaluation functions for the STREAM benchmark
- `exp-results/`: Contains results from experiments
- `explore_results.ipynb`: Jupyter notebook for analyzing results
- `requirements.txt`: Required Python packages
