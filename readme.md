# Survival Analysis with Adversarial Regularization (SAWAR)

Submitted to IEEE Transactions on Neural Network and Learning Systems.

## Overview

Survival Analysis (SA) involves modeling the time for an event of interest to occur, which has significant applications in fields such as medicine, defense, finance, and aerospace. Recent advancements have showcased the advantages of utilizing Neural Networks (NN) to capture complex relationships in SA. However, the datasets used to train these models often contain uncertainties (e.g., noisy measurements, human errors) that can significantly degrade the performance of existing techniques.

To address this challenge, this project leverages recent advances in NN verification to introduce new algorithms for generating fully-parametric survival models that are robust to uncertainties. Specifically, a robust loss function is introduced for training the models, and CROWN-IBP regularization is utilized to address computational challenges in solving the resulting Min-Max problem.

## Features

List the key features or functionalities of your project.

- Robust Training: Utilizes a robust loss function tailored for training survival models to handle uncertainties in datasets effectively.
- Adversarial Regularization: Implements CROWN-IBP regularization to enhance model resilience against uncertainties during training.
- Performance Evaluation: Empirically evaluates the performance of survival models against various baselines using metrics such as negative log-likelihood (negll), integrated Brier score (IBS), and concordance index (CI) on perturbed datasets from the SurvSet repository.

## Installation

To install and set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/mlpotter/SAWAR.git
    cd SAWAR
    ```

2. Install dependencies:

   - **JAX and JAXlib:**
   
     For Windows operating system, you must install jaxlib from the whl file. Follow the instructions provided [here](https://github.com/cloudhan/jax-windows-builder) to install jaxlib==0.4.19. Otherwise you may use pip.

   - Other dependencies:
     ```bash
     pip install -r requirements.txt
     ```
     
   - **AutoLiRPA:**

     Install AutoLiRPA version 0.4.0 from the AutoLiRPA GitHub repository:
     ```bash
     pip install git+https://github.com/Verified-Intelligence/auto_LiRPA.git
     ```



## Usage

### Example Usage of `main.py`

You can run the `main.py` script with the following arguments:

```bash
python main.py \
    --dataset TRACE \
    --seed 123 \
    --folder_name results_crownibp \
    --algorithm crownibp \
    --attack fgsm \
    --eps 0.5 \
    --lr 1e-3 \
    --sigma 1.0 \
    --weight "1/512" \
    --num_epochs 400 \
    --batch_size 512 \
    --smooth_window 5 \
    --scheduler_name SmoothedScheduler \
    --scheduler_opts "start=100,length=30" \
    --bound_type CROWN-IBP \
    --loss_wrapper rhc_rank \
    --norm inf \
    --pareto "0.1 0.9" \
    --verify \
    --cuda \
    --pgd_iter 1 \
    --hidden_dims "50 50" \
    --save_model ""
```
- `--dataset`: Dataset Name (TRACE, divorce, Dialysis, Aids2, Framingham, rott2, dataDIVAT1, prostate, ...)
- `--folder_name`: Folder name to save experiments to
- `--algorithm`: Algorithm for robust training (crownibp, pgd, noise)
- `--attack`: The attack method during evaluation (fgsm, crownibp)
- `--eps`: The perturbation maximum for the adversarial training method

### Example Usage of `main_experiments.py`

You can run all experiments from the paper via the `main_experiments.py` script:

```bash
python main_experiments.py
```

To recreate the exact results from the paper, set the `random_state=None` in ```load_data.py```.
