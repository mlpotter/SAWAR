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

hyperparameters = [
'--dataset=TRACE --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
'--dataset=divorce --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
'--dataset=Dialysis --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
'--dataset=Aids2 --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
'--dataset=Framingham --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
'--dataset=dataDIVAT1 --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
'--dataset=prostate --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=16 --weight=1/16 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
]


if __name__ == "__main__":
    processes = []
    for hyperparam in hyperparameters:
        file_full = f"python main_minimax.py {hyperparam}"
        print(file_full)
        # os.system(file_full)
        p = Popen(file_full, creationflags=CREATE_NEW_CONSOLE)
        processes.append(p)



    finished = False
    while not finished:
        finished = np.all([p.poll() is not None for p in processes])
        time.sleep(5)

    print("WRITE RESULTS")
    # aggregate all the CI files
    CI_excels = glob.glob(os.path.join("results","*","CI.xlsx"))
    CI_df = pd.DataFrame()
    for ci_excel in CI_excels:
        temp_df = pd.read_excel(ci_excel)
        dataset_name = ci_excel.split("\\")[1]
        temp_df.columns = ["eps"] + temp_df.columns[1:].to_list()
        CI_df[dataset_name] = temp_df["Robust CI"].round(3).astype(str) + " / " + temp_df["Non Robust CI"].round(3).astype(str)
    CI_df = CI_df.reindex(sorted(CI_df.columns),axis=1)
    CI_df.insert(0,"eps",temp_df.eps)

    # aggergate all the IBS files
    IBS_excels = glob.glob(os.path.join("results","*","IBS.xlsx"))
    IBS_df = pd.DataFrame()
    for IBS_excel in IBS_excels:
        temp_df = pd.read_excel(IBS_excel)
        dataset_name = IBS_excel.split("\\")[1]
        temp_df.columns = ["eps"] + temp_df.columns[1:].to_list()
        IBS_df[dataset_name] = temp_df["Robust IBS"].round(3).astype(str) + " / " + temp_df["Non Robust IBS"].round(3).astype(str)
    IBS_df = IBS_df.reindex(sorted(IBS_df.columns),axis=1)
    IBS_df.insert(0,"eps",temp_df.eps)

    # aggergate all the NegLL files
    NegLL_excels = glob.glob(os.path.join("results","*","NegLL.xlsx"))
    NegLL_df = pd.DataFrame()
    for NegLL_excel in NegLL_excels:
        temp_df = pd.read_excel(NegLL_excel)
        dataset_name = NegLL_excel.split("\\")[1]
        temp_df.columns = ["eps"] + temp_df.columns[1:].to_list()
        NegLL_df[dataset_name] = temp_df["Robust NegLL"].round(3).astype(str) + " / " + temp_df["Non Robust NegLL"].round(3).astype(str)
    NegLL_df = NegLL_df.reindex(sorted(NegLL_df.columns),axis=1)
    NegLL_df.insert(0,"eps",temp_df.eps)