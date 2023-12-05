import os

import numpy as np
# from subprocess import Popen
import sys
import subprocess
import os
import glob


from subprocess import CREATE_NEW_CONSOLE,Popen

import pandas as pd
import time



if __name__ == "__main__":
    algorithms = ["pgd","fgsm","noise","crownibp"]
    attack = "crownibp"
    for algo in algorithms:
        print(f"Algorithm {algo}")

        folder_name = f"results_{algo}"
        pgd_iter = 1 if algo=="fgsm" else 10
        algo = "pgd" if algo=="fgsm" else algo


        hyperparameters = [
            f'--dataset=TRACE --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --folder_name={folder_name} --eps=0.5 --lr=1e-3 --num_epochs=400 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
            f'--dataset=divorce  --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --folder_name={folder_name} --eps=0.5 --lr=1e-3 --num_epochs=400 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
            f'--dataset=Dialysis  --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --folder_name={folder_name}  --eps=0.5 --lr=1e-3 --num_epochs=400 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
            f'--dataset=Aids2  --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --folder_name={folder_name} --eps=0.5 --lr=1e-3 --num_epochs=60 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=20,length=15" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
            f'--dataset=Framingham  --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --folder_name={folder_name} --eps=0.5 --lr=1e-3 --num_epochs=400 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
            f'--dataset=dataDIVAT1  --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --folder_name={folder_name} --eps=0.5 --lr=1e-3 --num_epochs=400 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
            f'--dataset=prostate  --algorithm={algo} --pgd_iter={pgd_iter} --attack={attack} --folder_name={folder_name} --eps=0.5 --lr=1e-3 --num_epochs=400 --no-cuda --batch_size=16 --weight=1/16 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=30" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
        ]

        results_folder = os.path.join("results", attack, folder_name)

        processes = []

        for hyperparam in hyperparameters:
            file_full = f"python main.py {hyperparam}"
            print(file_full)
            # os.system(file_full)
            p = Popen(file_full, creationflags=CREATE_NEW_CONSOLE)
            processes.append(p)


        finished = False
        while not finished:
            finished = np.all([p.poll() is not None for p in processes])
            time.sleep(5)

        print(f"WRITE RESULTS FOR {algo}")

        # aggregate all the CI files
        CI_excels = glob.glob(os.path.join(results_folder,"*","CI.xlsx"))
        CI_df = pd.DataFrame()
        N_datasets = len(CI_excels)
        percentage_change = []
        for ci_excel in CI_excels:

            temp_df = pd.read_excel(ci_excel)
            dataset_name = ci_excel.split("\\")[1]
            temp_df.columns = ["eps"] + temp_df.columns[1:].to_list()
            CI_df[dataset_name] = temp_df["Non Robust CI"].round(3).astype(str) + " / " + temp_df["Robust CI"].round(3).astype(str)
            percentage_change.append( (temp_df["Robust CI"] - temp_df["Non Robust CI"]) / np.abs(temp_df["Non Robust CI"]) * 100)

        CI_df = CI_df.reindex(sorted(CI_df.columns),axis=1)
        CI_df.insert(0,"eps",temp_df.eps)
        perc_df = pd.concat(percentage_change, axis=1)
        CI_df["%"] = perc_df.mean(axis=1)
        CI_df.to_excel(os.path.join(results_folder,"CI_all.xlsx"),index=False)
        print("CI",CI_df)

        # aggergate all the IBS files
        IBS_excels = glob.glob(os.path.join(results_folder,"*","IBS.xlsx"))
        IBS_df = pd.DataFrame()
        N_datasets = len(IBS_excels)
        percentage_change = []
        for IBS_excel in IBS_excels:

            temp_df = pd.read_excel(IBS_excel)
            dataset_name = IBS_excel.split("\\")[1]
            temp_df.columns = ["eps"] + temp_df.columns[1:].to_list()
            IBS_df[dataset_name] = temp_df["Non Robust IBS"].round(3).astype(str) + " / " + temp_df["Robust IBS"].round(3).astype(str)
            percentage_change.append( (temp_df["Robust IBS"] - temp_df["Non Robust IBS"]) / np.abs(temp_df["Non Robust IBS"]) * 100)

        IBS_df = IBS_df.reindex(sorted(IBS_df.columns),axis=1)
        IBS_df.insert(0,"eps",temp_df.eps)
        perc_df = pd.concat(percentage_change, axis=1)
        IBS_df["%"] = perc_df.mean(axis=1)
        IBS_df.to_excel(os.path.join(results_folder,"IBS_all.xlsx"),index=False)
        print("IBS",IBS_df)

        # aggergate all the NegLL files
        NegLL_excels = glob.glob(os.path.join(results_folder,"*","NegLL.xlsx"))
        NegLL_df = pd.DataFrame()

        N_datasets = len(NegLL_excels)
        for NegLL_excel in NegLL_excels:

            temp_df = pd.read_excel(NegLL_excel)
            dataset_name = NegLL_excel.split("\\")[1]
            temp_df.columns = ["eps"] + temp_df.columns[1:].to_list()
            temp_df[temp_df == ''] = np.NaN
            temp_df = temp_df.astype(float)
            NegLL_df[dataset_name] = temp_df["Non Robust NegLL"].round(3).apply(lambda x: "{:.2e}".format(x)) + " / " + temp_df["Robust NegLL"].round(3).apply(lambda x: "{:.2e}".format(x))

        NegLL_df = NegLL_df.reindex(sorted(NegLL_df.columns),axis=1)
        NegLL_df.insert(0,"eps",temp_df.eps)
        NegLL_df.to_excel(os.path.join(results_folder,"NegLL_all.xlsx"),index=False)
        print("NegLL",NegLL_df)
