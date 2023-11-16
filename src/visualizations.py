import time
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
import random

from tqdm import tqdm
import numpy as np

from src.utils import lower_bound

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


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
def visualize_population_curves_attacked(clf_fragile,clf_robust,dataloader,epsilons=[0.1]):

    plt.figure(figsize=(10,10))
    X_train,T_train,E_train = dataloader.dataset.tensors
    t = torch.linspace(0,T_train.max(),10000)

    St_robust_x = clf_robust.survival_qdf(X_train, t).detach()
    St_fragile_x = clf_fragile.survival_qdf(X_train, t).detach()

    kmf = KaplanMeierFitter()
    kmf.fit(durations=T_train,event_observed=E_train)
    St_kmf = kmf.predict(times=t.ravel().numpy())


    fig,axes = plt.subplots(1,2,figsize=(20,10))
    axes[0].plot(t,St_kmf,linewidth=3)
    axes[0].plot(t,St_fragile_x.mean(0),'k-',linewidth=3)
    axes[0].plot(t,St_robust_x.mean(0),'r-',linewidth=3)

    for epsilon in epsilons:
        lb,ub = lower_bound(clf_robust,X_train,epsilon)
        St_lb = torch.exp(-ub*t).mean(0)

        axes[0].plot(t,St_lb.detach(),'--')



    axes[0].set_ylabel("S(t)"); axes[0].set_xlabel("Time")
    axes[0].legend(["Kaplan Meier Numerical","Neural Network Nonrobust","Neural Network Robust"]+[f"LB@{epsilon}" for epsilon in epsilons])
    axes[0].set_title(f"Robust Population Survival Curves")
    axes[0].set_ylim([0,1])

    axes[1].plot(t,St_kmf,linewidth=3)
    axes[1].plot(t,St_fragile_x.mean(0),'k-',linewidth=3)
    axes[1].plot(t,St_robust_x.mean(0),'r-',linewidth=3)

    for epsilon in epsilons:
        lb,ub = lower_bound(clf_fragile,X_train,epsilon)
        St_lb = torch.exp(-ub*t).mean(0)
        axes[1].plot(t,St_lb.detach(),'--')



    axes[1].set_ylabel("S(t)"); axes[1].set_xlabel("Time")
    axes[1].legend(["Kaplan Meier Numerical","Neural Network Nonrobust","Neural Network Robust"]+[f"LB@{epsilon}" for epsilon in epsilons])
    axes[1].set_title("Nonrobust Population Survival Curves")
    axes[1].set_ylim([0,1])

    plt.show()