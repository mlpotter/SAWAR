from src.models import Exponential_Model
from src.criterion import RightCensorWrapper,RankingWrapper,RHC_Ranking_Wrapper
from src.load_data import load_datasets,load_dataframe
from src.utils import train_robust,lower_bound
from src.visualizations import *
from src.metrics import concordance,ibs,rhc_neg_logll

from torch.optim import Adam
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter,CoxPHFitter,ExponentialFitter,WeibullAFTFitter
from lifelines.utils import concordance_index

from auto_LiRPA import BoundedModule, BoundedTensor

import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np
from copy import deepcopy
import random
import os

# set seeds for random!!!
torch.manual_seed(123)
random.seed(123)
np.random.seed(123)
torch.cuda.manual_seed_all(123)

def loss_wrapper(loss_wrapper):
    if loss_wrapper == "rank":
        return RankingWrapper
    elif loss_wrapper == "rhc":
        return RightCensorWrapper
    elif loss_wrapper == "rhc_rank":
        return RHC_Ranking_Wrapper
    else:
        raise Exception("not valid wrapper choice")

def main(args):
    df_train,df_val,df_test = load_dataframe(ds_name=args.dataset,drop_first=True)
    dataset_train, dataset_val, dataset_test = load_datasets(args.dataset, test_size=0.2)
    input_dims = dataset_train.tensors[0].shape[1]
    output_dim = 1

    # load the datasets
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    # set the dataloader mean and std ... not needed right now
    dataloader_train.mean = dataloader_val.mean = dataloader_test.mean = dataset_train.mean
    dataloader_train.std = dataloader_val.mean = dataloader_test.std = dataset_train.std

    print(f"The train dataset shape {dataset_train.tensors[0].shape}")
    print(f"The val dataset shape {dataset_val.tensors[0].shape}")
    print(f"The test dataset shape {dataset_test.tensors[0].shape}")

    # initialize the Neural Network models (exponential models)
    clf_robust = Exponential_Model(input_dim=input_dims,hidden_layers=args.hidden_dims)
    clf_fragile = Exponential_Model(input_dim=input_dims,hidden_layers=args.hidden_dims)
    clf_fragile.load_state_dict(deepcopy(clf_robust.state_dict()))


    epsilons = [2.0,1, .8, 0.7, .6, 0.5, 0.1, 0.07, 0.05, 0]
    eps_random, ci_random= concordance(clf_robust, dataloader_train, epsilons)
    df_ci_random = pd.DataFrame({"RANDOM CI":ci_random},index=eps_random)
    print("Train Concordance Index RANDOM \n",df_ci_random)

    eps_robust, ibs_random = ibs(clf_robust, dataloader_train, dataloader_test,epsilons)
    df_ibs_random = pd.DataFrame({"RANDOM IBS":ibs_random},index=eps_robust)
    print("Test Integrated Brier Score RANDOM \n",df_ibs_random)

    eps_robust, neg_ll_random = rhc_neg_logll(clf_robust, dataloader_train,epsilons)
    df_negll_random = pd.DataFrame({"RANDOM Neg LL":neg_ll_random},index=eps_robust)
    print("Train Neg LL RANDOM \n",df_negll_random)


    # initialize the model objective wrappers and make BoundedModule
    wrapper = loss_wrapper(args.loss_wrapper)
    model_robust_wrap = BoundedModule(wrapper(clf_robust,weight=args.weight,sigma=args.sigma),dataloader_train.dataset.tensors)
    model_fragile_wrap = BoundedModule(wrapper(clf_fragile,weight=args.weight,sigma=args.sigma),dataloader_train.dataset.tensors)

    # train models (robust and nonrobust)
    _,loss_tr_robust,loss_val_robust = train_robust(model_robust_wrap, dataloader_train, dataloader_val, method="robust", args=args)
    epochs,loss_tr_fragile,loss_val_fragile = train_robust(model_fragile_wrap, dataloader_train,dataloader_val, method="natural", args=args)

    # get the train tensors (X,T,E)
    X_train, T_train, E_train = dataloader_train.dataset.tensors
    t = torch.linspace(0, T_train.max(), 1000)

    # ================= Fit other models ============================== #

    kmf = KaplanMeierFitter()
    kmf.fit(durations=T_train,event_observed=E_train)

    clf_aft = WeibullAFTFitter()
    clf_aft.fit(df=df_train,duration_col="time",event_col="event")

    clf_exp = ExponentialFitter()
    clf_exp.fit(durations=T_train.ravel(),event_observed=E_train.ravel())

    # ================= Plot the survival curves ====================== #

    St_robust_x = clf_robust.survival_qdf(X_train,t).detach()
    St_fragile_x = clf_fragile.survival_qdf(X_train,t).detach()
    St_kmf  = kmf.predict(times=t.ravel().numpy())
    St_exp = clf_exp.predict(times=t.ravel().numpy())

    plt.figure(figsize=(10,10))
    plt.plot(t,St_kmf)
    plt.plot(t,St_exp)
    plt.plot(t,St_fragile_x.mean(0))

    plt.plot(t,St_robust_x.mean(0))

    plt.ylabel("S(t)"); plt.xlabel("Time")
    plt.legend(["Kaplan Meier Numerical",f"Exponential Fit $\lambda$={np.round(1/clf_exp.params_[0],4)}","Neural Network Normal","Neural Network Robust"])
    plt.title("Train Population Survival Curves")
    plt.ylim([0,1.05])
    plt.tight_layout()
    if img_path != "":
        plt.savefig(os.path.join(args.img_path,f"survival_curves.png"))
    plt.show()

    # ================ WeibullAFTFitter =================== #
    kmf.plot()
    clf_aft.predict_survival_function(df_train).mean(1).plot(label="Weibull AFT",figsize=(10,10))
    plt.legend()
    plt.ylim([0,1.05])
    plt.tight_layout()
    plt.show()
    print("Weibull AFT\n",clf_aft.params_)

    print("Lifelines Weibull AFT Train CI: {:.3f}".format(clf_aft.score(df_train, scoring_method="concordance_index")))
    print("Lifelines Weibull AFT Test CI: {:.3f}".format(clf_aft.score(df_test, scoring_method="concordance_index")))

    # ======================= Benchmarks ========================== #

    epsilons = [2.0,1, .8, 0.7, .6, 0.5, 0.1, 0.07, 0.05, 0]
    eps_robust, ci_robust = concordance(clf_robust, dataloader_train, epsilons)
    _, ci_fragile = concordance(clf_fragile, dataloader_train, epsilons)
    df_ci_train = pd.DataFrame({"Robust CI":ci_robust,"Non Robust CI":ci_fragile},index=eps_robust)
    print("Train Concordance Index \n",df_ci_train)

    eps_robust, ci_robust = concordance(clf_robust, dataloader_test, epsilons)
    _, ci_fragile = concordance(clf_fragile, dataloader_test, epsilons)
    df_ci_test = pd.DataFrame({"Robust CI":ci_robust,"Non Robust CI":ci_fragile},index=eps_robust)
    df_ci_test.to_excel(os.path.join(args.img_path,"CI.xlsx"),index_label="eps")
    print("Test Concordance Index \n",df_ci_test)


    eps_robust, ibs_robust = ibs(clf_robust, dataloader_train, dataloader_test,epsilons)
    _, ibs_fragile = ibs(clf_fragile, dataloader_train,dataloader_test, epsilons)
    df_ibs_test = pd.DataFrame({"Robust IBS":ibs_robust,"Non Robust IBS":ibs_fragile},index=eps_robust)
    df_ibs_test.to_excel(os.path.join(args.img_path,"IBS.xlsx"),index_label="eps")
    print("Test Integrated Brier Score \n",df_ibs_test)

    eps_robust, neg_ll_robust = rhc_neg_logll(clf_robust, dataloader_test,epsilons)
    _, neg_ll_fragile = rhc_neg_logll(clf_fragile, dataloader_test, epsilons)
    df_neg_ll_test = pd.DataFrame({"Robust NegLL":neg_ll_robust,"Non Robust NegLL":neg_ll_fragile},index=eps_robust)
    df_neg_ll_test.to_excel(os.path.join(args.img_path,"NegLL.xlsx"),index_label="eps")
    print("Test NLL \n",df_neg_ll_test)

    visualize_learning_curves(epochs, loss_tr_fragile, loss_val_fragile, loss_tr_robust, loss_val_robust, suptitle="Learning Curves",
                              img_path=args.img_path)

    # visualize the output of the neural network as function of data
    if args.dataset not in ["prostate"]:
        visualize_individual_lambda_histograms(clf_fragile, clf_robust, dataloader_train, suptitle="train",img_path=args.img_path)

    # visualize the randomness of the survival curves from data
    visualize_curve_distributions(clf_fragile,clf_robust,dataloader_train,suptitle="train")
    visualize_curve_distributions(clf_fragile,clf_robust,dataloader_test,suptitle="test",img_path=args.img_path)

    # visualize the attacks on the curves
    epsilons = [2.0,1.0, 0.8, 0.7, 0.6, 0.5, 0.1, 0.07, 0.05]
    visualize_population_curves_attacked(clf_fragile, clf_robust, dataloader_train, epsilons=epsilons, suptitle="train")
    visualize_population_curves_attacked(clf_fragile, clf_robust, dataloader_test, epsilons=epsilons, suptitle="test",img_path=args.img_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Minimax Adversarial Optimization',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default="TRACE",help='Dataset Name (TRACE,divorce,Dialysis,Aids2,Framingham,rott2,dataDIVAT1,prostate,...)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for training Neural Netwrok')

    # training information
    parser.add_argument('--eps', type=float, default=0.5, help='The pertubation maximum during minimax training')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate for the optimizer')
    parser.add_argument('--sigma', type=float, default=1.0, help='The spread of the comparison loss')
    parser.add_argument('--weight', type=str, default="1.0", help='The weight for the comparison loss contribution')
    parser.add_argument('--num_epochs', type=int, default=150, help='The number of training epochs during optimization')
    parser.add_argument('--batch_size', type=int, default=512, help='The batch size during training')
    parser.add_argument('--smooth_window', type=int, default=5, help='The smoothing window size for early stopping')

    # use 128 for the batch size ...

    # perturbation settings during training
    parser.add_argument('--scheduler_name', type=str,default='SmoothedScheduler',help='Scheduler for the pertubation adaptation during training')
    parser.add_argument('--scheduler_opts', type=str,default="start=100,length=10",help='Options for the perturbation adaptation during training')
    parser.add_argument('--bound_type', type=str, default="CROWN-IBP", help='The bound type to use with autolirpa (does not do anything as of now)')
    parser.add_argument('--loss_wrapper', type=str,default="rhc_rank",help='The training objective function (rank,rhc,rhc_rank')
    parser.add_argument('--norm', type=float, default=np.inf, help='The norm to use for the epsilon ball')
    parser.add_argument('--pareto', type=str, default="0.1 0.9", help='The weighted combination between the normal objective and the autolirpa upper bound')
    parser.add_argument('--verify', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cuda', action=argparse.BooleanOptionalAction)

    # neural network information
    parser.add_argument('--hidden_dims', type=str, default="50 50", help='The number of neurons in each hidden layers')
    parser.add_argument('--save_model',type=str,default="",help="The Neural Network parameters .pth save")


    args = parser.parse_args()

    args.hidden_dims = [int(h) for h in args.hidden_dims.split()]
    args.pareto = [float(p) for p in args.pareto.split()]
    args.weight = eval(args.weight)


    device = "cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu"

    args.device = device

    print(f"Dataset Analyzed {args.dataset}")
    print(f"Objective Function {args.loss_wrapper}")

    img_path = os.path.join("results",args.dataset)
    os.makedirs(img_path, exist_ok=True)

    args.img_path = img_path

    main(args)