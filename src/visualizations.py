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
def visualize_population_curves_attacked(clf_fragile,clf_robust,dataloader,epsilons=[0.1],suptitle="",img_path=""):

    plt.figure(figsize=(10,10))
    X,T,E = dataloader.dataset.tensors
    t = torch.linspace(0,T.max(),10000)

    St_robust_x = clf_robust.survival_qdf(X, t).detach()
    St_fragile_x = clf_fragile.survival_qdf(X, t).detach()

    kmf = KaplanMeierFitter()
    kmf.fit(durations=T,event_observed=E)
    St_kmf = kmf.predict(times=t.ravel().numpy())

    attack_df = pd.DataFrame({"t":t.ravel(),
                             "kmf_St":St_kmf,
                             "baseline_St":St_fragile_x.mean(0),
                             "robust_St": St_robust_x.mean(0)})

    fig,axes = plt.subplots(1,2,figsize=(20,10))
    axes[1].plot(t,St_kmf,linewidth=3)
    axes[1].plot(t,St_fragile_x.mean(0),'k-',linewidth=3)
    axes[1].plot(t,St_robust_x.mean(0),'r-',linewidth=3)

    for epsilon in epsilons:
        lb,ub = lower_bound(clf_robust,X,epsilon)
        St_lb = torch.exp(-ub*t).mean(0).detach()

        attack_df["robust_eps={:.2f}".format(epsilon)] = St_lb
        axes[1].plot(t,St_lb,'--')

    axes[1].set_ylabel("S(t)"); axes[0].set_xlabel("Time")
    axes[1].legend(["Kaplan Meier Numerical","Neural Network Baseline","Neural Network Robust"]+[f"LB@{epsilon}" for epsilon in epsilons])
    axes[1].set_title(f"Robust Population Survival Curves")
    axes[1].set_ylim([0,1])


    axes[0].plot(t,St_kmf,linewidth=3)
    axes[0].plot(t,St_fragile_x.mean(0),'k-',linewidth=3)
    axes[0].plot(t,St_robust_x.mean(0),'r-',linewidth=3)

    for epsilon in epsilons:
        lb,ub = lower_bound(clf_fragile,X,epsilon)
        St_lb = torch.exp(-ub*t).mean(0).detach()

        attack_df["baseline_eps={:.2f}".format(epsilon)] = St_lb
        axes[0].plot(t,St_lb,'--')

    axes[0].set_ylabel("S(t)"); axes[1].set_xlabel("Time")
    axes[0].legend(["Kaplan Meier Numerical","Neural Network Baseline","Neural Network Robust"]+[f"LB@{epsilon}" for epsilon in epsilons])
    axes[0].set_title("Baseline Population Survival Curves")
    axes[0].set_ylim([0,1])

    plt.suptitle(suptitle)
    plt.tight_layout()

    if img_path != "":
        plt.savefig(os.path.join(img_path,f"population_curves_attacked_{suptitle}.png"))



        attack_df.to_excel(os.path.join(img_path,f"population_curves_attacked_{suptitle}.xlsx"),index=False)

    plt.show()

def visualize_individual_lambda_histograms(clf_fragile,clf_robust,dataloader,suptitle="",img_path=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    X,_,_ = dataloader.dataset.tensors


    lambda_robust = clf_robust(X).detach()
    lambda_fragile = clf_fragile(X).detach()

    if len(lambda_robust[lambda_robust < lambda_robust.quantile(0.98)]) == len(lambda_fragile[lambda_fragile < lambda_fragile.quantile(0.98)]):
        lambda_robust = lambda_robust[lambda_robust < lambda_robust.quantile(0.98)]
        lambda_fragile = lambda_fragile[lambda_fragile < lambda_fragile.quantile(0.98)]

    plot_df = pd.DataFrame({"Lambda Robust": lambda_robust.ravel(), "Lambda Baseline": lambda_fragile.ravel()})

    sns.histplot(data=plot_df, x="Lambda Robust", ax=axes[1], stat="density", legend=False, color="blue")
    axes[1].set_xlim([lambda_fragile.min(), lambda_fragile.quantile(0.98)])
    axes[1].set_title("$\mu$={:.4f} $\sigma^2$={:.4f}".format(lambda_robust.mean(),lambda_robust.var()))

    axes[0].set_xlim([lambda_fragile.min(), lambda_fragile.quantile(0.98)])
    axes[0].set_title("$\lambda$ Baseline")
    sns.histplot(data=plot_df, x="Lambda Baseline", ax=axes[0], stat="density", legend=False, color="orange")
    axes[0].set_title("$\mu$={:.4f} $\sigma^2$={:.4f}".format(lambda_fragile.mean(),lambda_fragile.var()))

    axes[2].set_xlim([lambda_fragile.min(), lambda_fragile.quantile(0.98)])
    axes[2].set_title("$\lambda$ Overlap")
    sns.histplot(data=plot_df, ax=axes[2], stat="density", legend=True)
    plt.suptitle(suptitle)
    plt.tight_layout()

    if img_path != "":
        plt.savefig(os.path.join(img_path,f"individual_lambda_histogram_{suptitle}.png"))

    plt.show()


def visualize_calibration_curves(clf_fragile,clf_robust,dataloader,suptitle="",img_path=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    X,T,E = dataloader.dataset.tensors


    lambda_robust = clf_robust(X).detach()
    lambda_fragile = clf_fragile(X).detach()

    y_pred_robust = 1 - torch.exp(-lambda_robust * T)
    y_pred_fragile = 1 - torch.exp(-lambda_fragile * T)

    prob_true_rob, prob_pred_rob = calibration_curve(E.ravel(), y_pred_robust.ravel(), pos_label=1, n_bins=10, strategy='quantile')
    prob_true_fra, prob_pred_fra = calibration_curve(E.ravel(), y_pred_fragile.ravel(), pos_label=1, n_bins=10, strategy='quantile')


    plot_df = pd.DataFrame({"prob_true_rob": prob_true_rob.ravel(), "prob_pred_rob": prob_pred_rob.ravel(),
                            "prob_true_fra": prob_true_fra.ravel(), "prob_pred_fra": prob_pred_fra.ravel()})

    axes[1].set_xlim([-0.05, 1.05])
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].set_xlabel("Mean Predicted Probability")
    axes[1].set_ylabel("Fraction of positives")
    axes[1].plot(prob_pred_rob,prob_true_rob,'ro-')
    axes[1].plot([0,1],[0,1],'g--')
    axes[1].set_title("Robust")

    axes[0].set_xlim([-0.05, 1.05])
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of positives")
    axes[0].plot(prob_pred_fra,prob_true_fra,'ko-')
    axes[0].plot([0,1],[0,1],'g--')
    axes[0].set_title("Baseline")

    axes[2].set_xlim([-0.05, 1.05])
    axes[2].set_ylim([-0.05, 1.05])
    axes[2].set_xlabel("Mean Predicted Probability")
    axes[2].set_ylabel("Fraction of positives")
    axes[2].set_title("Overlap")
    axes[2].plot(prob_pred_fra,prob_true_fra,'ko-')
    axes[2].plot(prob_pred_rob,prob_true_rob,'ro-')
    axes[2].plot([0,1],[0,1],'g--')


    plt.suptitle(suptitle)
    plt.tight_layout()

    if img_path != "":
        plt.savefig(os.path.join(img_path,f"calibration_curves_{suptitle}.png"))
        plot_df.to_excel(os.path.join(img_path,f"calibration_curves_{suptitle}.xlsx"),index=False)

    plt.show()




def visualize_curve_distributions(clf_fragile,clf_robust,dataloader,suptitle="",img_path=""):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    X,T,E = dataloader.dataset.tensors

    t = torch.linspace(0,T.max(),1000)

    q_fragile = clf_fragile.survival_qdf(X,t).detach()
    q_robust = clf_robust.survival_qdf(X,t).detach()

    print(q_robust.shape)


    a = sns.lineplot(x=t, y=q_robust.mean(dim=0), label='Average S(t)', linewidth=3.0, ax=axes[1])
    b = sns.lineplot(x=t, y=q_robust.quantile(0.95,dim=0), label='Confidence', color='r', linewidth=3.0,
                     ax=axes[1])
    c = sns.lineplot(x=t, y=q_robust.quantile(0.05,dim=0), label='Confidence', color='r', linewidth=3.0,
                     ax=axes[1])

    line = c.get_lines()
    axes[1].fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(), color='blue', alpha=.3)
    axes[1].set_ylim([0, 1.05])
    axes[1].set_xlabel("time");
    axes[1].set_ylabel("S(t)")
    # sns1scatterplot(x =df_sat_test['t'], y = np.array(test_ppc.observed_data.obs), label = 'True Value')
    axes[1].set_title("ROBUST")
    axes[1].legend()

    a = sns.lineplot(x=t, y=q_fragile.mean(dim=0), label='Average S(t)', linewidth=3.0, ax=axes[0])
    b = sns.lineplot(x=t, y=q_fragile.quantile(0.95,dim=0), label='Confidence', color='r', linewidth=3.0,
                     ax=axes[0])
    c = sns.lineplot(x=t, y=q_fragile.quantile(0.05,dim=0), label='Confidence', color='r', linewidth=3.0,
                     ax=axes[0])

    line = c.get_lines()
    axes[0].set_title("Baseline")
    axes[0].fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(), color='blue', alpha=.3)
    axes[0].set_ylim([0, 1.05])
    axes[0].set_xlabel("time");
    axes[0].set_ylabel("S(t)")
    # sns.scatterplot(x =df_sat_test['t'], y = np.array(test_ppc.observed_data.obs), label = 'True Value')
    axes[0].legend()

    plt.suptitle(suptitle)
    plt.tight_layout()

    if img_path != "":
        plt.savefig(os.path.join(img_path,f"curve_distributions_{suptitle}.png"))

        q_df = pd.DataFrame({"t": t.ravel(),
                             "baseline_mean": q_fragile.mean(dim=0),
                             "baseline_q95": q_fragile.quantile(0.95, dim=0),
                             "baseline_q05": q_fragile.quantile(0.05, dim=0),
                             "robust_mean": q_robust.mean(dim=0),
                             "robust_q95": q_robust.quantile(0.95, dim=0),
                             "robust_q05": q_robust.quantile(0.05, dim=0),
                             })

        q_df.to_excel(os.path.join(img_path,f"curve_distributions_{suptitle}.xlsx"),index=False)

    plt.show()

def visualize_learning_curves(epochs,loss_tr_fragile,loss_val_fragile,loss_tr_robust,loss_val_robust,suptitle="",img_path=""):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))


    axes[0].plot(epochs,loss_tr_fragile)
    axes[0].plot(epochs,loss_val_fragile)
    axes[0].set_title("Baseline Learning Curve")
    axes[0].legend(["Train","Validation"])

    axes[1].plot(epochs, loss_tr_robust)
    axes[1].plot(epochs, loss_val_robust)
    axes[1].set_title("Robust Learning Curve")
    axes[1].legend(["Train", "Validation"])

    plt.suptitle(suptitle)
    plt.tight_layout()

    if img_path != "":
        plt.savefig(os.path.join(img_path,f"train_val_{suptitle}.png"))

    plt.show()