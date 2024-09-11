import os
import os.path as osp
import re

import numpy as np

import argparse
from subprocess import Popen
import pdb
from time import sleep
OS = os.name

if OS == "nt":
    from subprocess import CREATE_NEW_CONSOLE


def parse_input_string(input_string):
    # Define regex patterns to extract parameters
    patterns = {
        'attack': r'--attack=(\w+)',
        'algorithm': r'--algorithm=(\w+)',
        'dataset': r'--dataset=(\w+)',
        'seed': r'--seed=(\d+)',
    }

    # Extract parameters using regex
    extracted_params = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, input_string)
        if match:
            extracted_params[key] = match.group(1)
        else:
            raise ValueError(f"Could not find parameter {key} in the input string.")

    return extracted_params


def experiment_exists(base_path, params):
    # Construct the folder path based on extracted parameters
    attack_folder = f"attack_{params['attack']}"
    algorithm_folder = f"results_{params['algorithm']}"
    dataset_folder = params['dataset']
    seed_folder = f"seed_{params['seed']}"

    # Create the full path to check
    experiment_path = os.path.join(base_path, attack_folder, algorithm_folder, dataset_folder, seed_folder)

    # Check if the path exists
    return (os.path.exists(experiment_path) and len(os.listdir(experiment_path)) > 10)


if __name__ == "__main__":

    os.makedirs("logs",exist_ok=True)
    #  set to true if you do not want to run the experiments again when results exist
    batch_job_size = 3

    algorithms = ["draft","aae","pgd", "fgsm", "noise", "crownibp"]
    attacks=["fgsm","crownibp"]
    seeds = [111,222,333,444,555]

    experiments = []

    for attack  in attacks:
        for seed in seeds:
            for algo in algorithms:
                pgd_iter = 1 if algo == "fgsm" else 10

                hyperparameters_set = [
                    f'--seed={seed} --dataset=TRACE --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --eps=0.5 --lr=1e-3 --num_epochs=400 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
                    f'--seed={seed} --dataset=Aids2  --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --eps=0.5 --lr=1e-3 --num_epochs=60 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=20,length=15" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
                    f'--seed={seed} --dataset=Framingham  --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --eps=0.5 --lr=1e-3 --num_epochs=400 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
                    f'--seed={seed} --dataset=dataDIVAT1  --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --eps=0.5 --lr=1e-3 --num_epochs=400 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
                    f'--seed={seed} --dataset=prostate  --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --eps=0.5 --lr=1e-3 --num_epochs=400 --no-cuda --batch_size=16 --weight=1/16 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
                    f'--seed={seed} --dataset=flchain --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --eps=0.5 --lr=1e-3 --num_epochs=400 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
                    f'--seed={seed} --dataset=retinopathy --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --eps=0.5 --lr=1e-3 --num_epochs=200 --no-cuda --batch_size=32 --weight=1/32 --scheduler_name=SmoothedScheduler --scheduler_opts="start=30,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
                    f'--seed={seed} --dataset=stagec --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --eps=0.5 --lr=1e-3 --num_epochs=150 --no-cuda --batch_size=16 --weight=1/16 --scheduler_name=SmoothedScheduler --scheduler_opts="start=30,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
                    f'--seed={seed} --dataset=zinc --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --eps=0.5 --lr=1e-3 --num_epochs=200 --no-cuda --batch_size=32 --weight=1/32 --scheduler_name=SmoothedScheduler --scheduler_opts="start=30,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
                    f'--seed={seed} --dataset=LeukSurv --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --eps=0.5 --lr=1e-3 --num_epochs=250 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
                ]

                experiments.extend(hyperparameters_set)

    processes = []
    exp_count = 0
    while exp_count <= (len(experiments) - 1):

        if (np.sum([p.poll() is None for p in processes]) < batch_job_size) or (len(processes) < batch_job_size):
            hyperparam = experiments[exp_count]

            # check if folder exists and filled with results
            if experiment_exists("results", parse_input_string(hyperparam)):
                exp_count += 1
                continue

            print(hyperparam)
            if OS == "nt":
                file_full = f"python main.py {hyperparam}"
                print(file_full)
                # os.system(file_full)
                p = Popen(file_full, creationflags=CREATE_NEW_CONSOLE)

            else:
                file_full = f"tmux new-session -d python3 main.py {hyperparam}"
                print(file_full)
                # os.system(file_full)
                p = Popen(file_full, shell=True)

            exp_count += 1
            processes.append(p)

    finished = False
    while not finished:
        finished = np.all([p.poll() is not None for p in processes])
        time.sleep(5)