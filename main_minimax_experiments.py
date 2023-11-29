import os

import numpy as np
# from subprocess import Popen
import sys
import subprocess
import os
from subprocess import CREATE_NEW_CONSOLE,Popen
hyperparameters = [
'--dataset=TRACE --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
'--dataset=divorce --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
'--dataset=Dialysis --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
'--dataset=Aids2 --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
'--dataset=Framingham --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
'--dataset=dataDIVAT1 --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=128 --weight=1/128 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
'--dataset=prostate --eps=0.5 --lr=1e-3 --num_epochs=300 --no-cuda --batch_size=32 --weight=1/32 --scheduler_name=SmoothedScheduler --scheduler_opts="start=100,length=10" --loss_wrapper=rhc_rank --pareto="0.1 0.9" --hidden_dims="50 50"',
]


if __name__ == "__main__":

    for hyperparam in hyperparameters:
        file_full = f"python main_minimax.py {hyperparam}"
        print(file_full)
        # os.system(file_full)
        Popen(file_full, creationflags=CREATE_NEW_CONSOLE)
