# Quantum-Inspired NAS with Attention-Based Search Spaces in Medical Applications

This repository contains the code and experiments for the paper "Quantum-Inspired NAS with Attention-Based Search Spaces in Medical Applications". It implements an enhanced Quantum-Inspired Neural Architecture Search (Q-NAS) approach incorporating Squeeze-and-Excitation (SE) and Convolutional Block Attention Modules (CBAM) to achieve efficient, high-performing CNN models for medical image classification tasks using the MedMNIST datasets.

## Environment Configuration

The following steps are used to configure the environment for the project.

- Miniconda Installation
- Conda Environment Creation
- Package Installation

**Notes**: 
- NVIDIA GPUs are required to run the project. 
- The project was tested using three NVIDIA RTX A30 GPUs to run evolutionary search with up to 20 individuals in parallel.
- NVIDIA drivers and the CUDA Toolkit are necessary.
- The following steps have been tested on Ubuntu 20.04.

### Miniconda Installation

Install Miniconda in the home directory. Refer to the [Miniconda Installation Guide](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) for more information.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

### Conda Environment Creation

```bash
conda create -n qnas python=3.9
conda activate qnas
```

### Package Installation

```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
## Running the Code

### Evolution Experiment
The `run.sh` script is used to perform the evolution experiments for Quantum-Inspired NAS. It iterates over multiple configurations and datasets to execute the evolutionary search.

#### Usage:
1. Open the `run.sh` script and adjust the following parameters as needed:
   - **`dataset`**: Specify the dataset (e.g., `organamnist`, `pathmnist`, etc.).
   - **`configs`**: List of configuration files used for the experiments.
   - **`exps`**: Corresponding experiment names.
   - **`cuda_devices`**: CUDA devices to use for running experiments.

2. Execute the script:
   ```bash
   bash run.sh
   ```

### Retrain Experiment
The `run_retrain.sh` script is used to retrain a model after the evolutionary search.

#### Usage:
1. Open the `run_retrain.sh` script and adjust the following parameters as needed:
   - **`dataset`**: Specify the dataset (e.g., `organamnist`, `pathmnist`, etc.).
   - **`exp`**: Specify the experiment name to retrain (e.g., `exp1`).
   - **`repeat`**: Specify the repetition number to retrain (e.g., `1`).

2. Execute the script:
   ```bash
   bash run_retrain.sh
    ```

## Additional Information

### Configuration Files
The configuration files define the parameters for the Quantum-Inspired NAS process, including the crossover rate, maximum generations, and hyperparameter search space. These files are located in the `config_files_medmnist` directory and are used during both evolution and retraining steps.

#### Key Parameters:
- **QNAS settings**:
  - `crossover_rate`: Probability of crossover during evolution.
  - `max_generations`: Maximum number of generations in the evolution process.
  - `num_quantum_ind`: Number of quantum individuals.
  - `params_ranges`: Hyperparameter search space.

- **Training settings**:
  - `batch_size`: Batch size for training.
  - `max_epochs`: Maximum number of epochs.
  - `optimizer`: Optimization algorithm used (e.g., AdamW).

- **Resource management**:
  - **`threads`**: Defines the maximum number of parallel processes that will run during the evolution. This parameter is critical for managing system resources and avoiding out-of-memory issues.

    - Adjust the threads parameter based on the number of available GPUs and the system memory.
    - Monitor GPU and memory usage during experiments to fine-tune this value for optimal performance.
    - For systems with limited resources, consider reducing the number of individuals or parallel processes to maintain stability.

Refer to the configuration files (e.g., `config1.txt`) for detailed settings and customize them as needed.


## Authors:
- **Diego Páez Ardila** (Department of ELE, PUC-Rio University, Rio de Janeiro, Brazil) | diego.paez@aluno.puc-rio.br
- **Thiago Carvalho** (Department of System Engineering and Computation, Rio de Janeiro State University, Rio de Janeiro, Brazil) | tmedeiros@eng.uerj.br
- **Santiago Vasquez Saavedra** (Department of ELE, PUC-Rio University, Rio de Janeiro, Brazil) | santiago.vasquez@aluno.puc-rio.br
- **Cesar Valencia Niño** (Fac. Mechatronics Eng., USTA University, Bucaramanga, Colombia) | cesar.valencia@ustabuca.edu.co
- **Karla Figueiredo** (Institute of Mathematics and Statistics, Rio de Janeiro State University, Rio de Janeiro, Brazil) | karla.figueiredo@gmail.com
- **Marley Vellasco** (Department of ELE, PUC-Rio University, Rio de Janeiro, Brazil) | marley@ele.puc-rio.br

## Citation:

If you use this code in your research, please cite the following paper:

```bibtex
@article{qnas_attention,
    title={Quantum-Inspired NAS with Attention-Based Search Spaces in Medical Applications},
    author={Diego Páez Ardila et al.},
    proceedings={Proceedings of the SSCI 2025},
    year={2025},
    volume={XX},
    pages={YY-ZZ}
}
```

## Acknowledgments:

This work was supported by the Brazilian agency **CNPq**.

The original Quantum-Inspired NAS code was developed by:

```bibtex
@article{szwarcman2022quantum,
  title={Quantum-inspired evolutionary algorithm applied to neural architecture search},
  author={Szwarcman, Daniela and Civitarese, Daniel and Vellasco, Marley},
  journal={Applied Soft Computing},
  volume={120},
  pages={108674},
  year={2022},
  publisher={Elsevier}
}
```

The original code can be found at:

[Q-NAS Repository](https://github.com/daniszw/qnas)
