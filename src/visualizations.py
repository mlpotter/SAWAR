import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"

import torch

import time
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import lower_bound

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.calibration import calibration_curve

torch.use_deterministic_algorithms(True)

def visualize_individual_curves_attacked(clf,dataloader,epsilon,order="ascending",test_cases=10):
    X,T,E = dataloader.dataset.tensors

    t = torch.linspace(0, T.max(), 10000)

    lb, ub = lower_bound(clf, X, epsilon)
    St_lb = torch.exp(-ub * t).detach()
    t = torch.linspace(0, T.max(), 10000)

    test_cases = min(test_cases,X.shape[0])

    plt.figure(figsize=(10, 10))

    St_x = clf.survival_qdf(X, t).detach()

    colors = list(plt.cm.brg(np.linspace(0, 1, test_cases))) + ["crimson", "indigo"]

    if order == "ascending":
        cases = np.argsort(torch.linalg.norm(St_lb - St_x, axis=1))[0:test_cases]

    elif order == "descending":
        cases = torch.flip(np.argsort(torch.linalg.norm(St_lb - St_x, axis=1)), dims=(0,))[0:test_cases]

    print(torch.linalg.norm(St_lb - St_x, axis=1)[cases])

    for i, case in enumerate(tqdm(cases)):
        plt.plot(t, St_x[case], color=colors[i])
        plt.plot(t, St_lb[case], '--', color=colors[i])

    plt.ylabel("S(t)");
    plt.xlabel("Time")
    plt.title(f"Individual Survival Curves Change order={order}")

def visualize_individual_curves_changes(clf_robust,clf_fragile,dataloader,order="ascending",test_cases=10):
    X,T,E = dataloader.dataset.tensors

    t = torch.linspace(0, T.max(), 10000)

    test_cases = min(test_cases,X.shape[0])

    plt.figure(figsize=(10, 10))

    St_robust_x = clf_robust.survival_qdf(X, t).detach()
    St_fragile_x = clf_fragile.survival_qdf(X, t).detach()

    colors = list(plt.cm.brg(np.linspace(0, 1, test_cases))) + ["crimson", "indigo"]

    if order == "ascending":
        cases = np.argsort(torch.linalg.norm(St_fragile_x - St_robust_x, axis=1))[0:test_cases]

    elif order == "descending":
        cases = torch.flip(np.argsort(torch.linalg.norm(St_fragile_x - St_robust_x, axis=1)), dims=(0,))[0:test_cases]

    print(torch.linalg.norm(St_fragile_x - St_robust_x, axis=1)[cases])

    for i, case in enumerate(tqdm(cases)):
        plt.plot(t, St_fragile_x[case], color=colors[i])
        plt.plot(t, St_robust_x[case], '--', color=colors[i])

    plt.ylabel("S(t)");
    plt.xlabel("Time")
    plt.title(f"Individual Survival Change Curves order={order}")
def visualize_population_curves_attacked(clf,dataloader,epsilons=[0.1],suptitle="",img_path=""):

    plt.figure(figsize=(10,10))
    X,T,E = dataloader.dataset.tensors
    t = torch.linspace(0,T.max(),10000)

    St_given_x = clf.survival_qdf(X, t).detach()

    kmf = KaplanMeierFitter()
    kmf.fit(durations=T,event_observed=E)
    St_kmf = kmf.predict(times=t.ravel().numpy())

    attack_df = pd.DataFrame({"t":t.ravel(),
                             "kmf_St":St_kmf,
                             "St": St_given_x.mean(0)})


    fig,axes = plt.subplots(1,1,figsize=(10,10))
    axes.plot(t,St_kmf,linewidth=3)
    axes.plot(t,St_given_x.mean(0),'r-',linewidth=3)

    for epsilon in epsilons:
        lb,ub = lower_bound(clf,X,epsilon)
        St_lb = torch.exp(-ub*t).mean(0).detach()

        attack_df["eps={:.2f}".format(epsilon)] = St_lb
        axes.plot(t,St_lb,'--')
        clf.zero_grad()
        del lb,ub

    axes.set_ylabel("S(t)"); axes.set_xlabel("Time")
    axes.legend(["Kaplan Meier Numerical","Neural Network"]+[f"LB@{epsilon}" for epsilon in epsilons])
    axes.set_title(f"Population Survival Curves")
    axes.set_ylim([0,1])


    plt.suptitle(suptitle)
    plt.tight_layout()

    if img_path != "":
        plt.savefig(os.path.join(img_path,f"population_curves_attacked_{suptitle}.png"))



        attack_df.to_excel(os.path.join(img_path,f"population_curves_attacked_{suptitle}.xlsx"),index=False)

    # plt.show()

def visualize_individual_lambda_histograms(clf,dataloader,suptitle="",img_path=""):
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    X,_,_ = dataloader.dataset.tensors


    lam = clf(X).detach()


    plot_df = pd.DataFrame({"Lambda": lam.ravel()})

    sns.histplot(data=plot_df, x="Lambda", ax=axes, stat="density", legend=False, color="blue")
    axes.set_xlim([lam.min(), lam.quantile(0.98)])
    axes.set_title("$\mu$={:.4f} $\sigma^2$={:.4f}".format(lam.mean(),lam.var()))


    if img_path != "":
        plt.savefig(os.path.join(img_path,f"individual_lambda_histogram_{suptitle}.png"))

    # plt.show()
    fig.suptitle(suptitle)
    fig.tight_layout()

    if img_path != "":
        plt.savefig(os.path.join(img_path,f"calibration_curves_{suptitle}.png"))
        plot_df.to_excel(os.path.join(img_path,f"calibration_curves_{suptitle}.xlsx"),index=False)


def visualize_curve_distributions(clf,dataloader,suptitle="",img_path=""):
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))

    X,T,E = dataloader.dataset.tensors

    t = torch.linspace(0,T.max(),1000)

    q_robust = clf.survival_qdf(X,t).detach()

    print(q_robust.shape)


    a = sns.lineplot(x=t, y=q_robust.mean(dim=0), label='Average S(t)', linewidth=3.0, ax=axes)
    b = sns.lineplot(x=t, y=q_robust.quantile(0.95,dim=0), label='Confidence', color='r', linewidth=3.0,
                     ax=axes)
    c = sns.lineplot(x=t, y=q_robust.quantile(0.05,dim=0), label='Confidence', color='r', linewidth=3.0,
                     ax=axes)

    line = c.get_lines()
    axes.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(), color='blue', alpha=.3)
    axes.set_ylim([0, 1.05])
    axes.set_xlabel("time");
    axes.set_ylabel("S(t)")
    # sns1scatterplot(x =df_sat_test['t'], y = np.array(test_ppc.observed_data.obs), label = 'True Value')
    axes.set_title("Survival Curve")
    axes.legend()

    fig.suptitle(suptitle)
    fig.tight_layout()

    if img_path != "":
        plt.savefig(os.path.join(img_path,f"curve_distributions_{suptitle}.png"))

        q_df = pd.DataFrame({"t": t.ravel(),
                             "mean": q_robust.mean(dim=0),
                             "q95": q_robust.quantile(0.95, dim=0),
                             "q05": q_robust.quantile(0.05, dim=0),
                             })

        q_df.to_excel(os.path.join(img_path,f"curve_distributions_{suptitle}.xlsx"),index=False)

    # plt.show()

def visualize_learning_curves(epochs,loss_tr,loss_val,suptitle="",img_path=""):
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))



    axes.plot(epochs, loss_tr)
    axes.plot(epochs, loss_val)
    axes.set_title("Robust Learning Curve")
    axes.legend(["Train", "Validation"])

    fig.suptitle(suptitle)
    fig.tight_layout()

    if img_path != "":
        plt.savefig(os.path.join(img_path,f"train_val_{suptitle}.png"))

    # plt.show()