from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score
import numpy as np
import pandas as pd
from src.utils import lower_bound,attack
import torch
from src.criterion import right_censored
from copy import deepcopy
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
from survival_evaluation import d_calibration

def concordance(clf, dataloader, epsilons,args=None):
    X, T, E = dataloader.dataset.tensors

    cis = np.zeros_like(epsilons)
    for i, epsilon in enumerate(epsilons):
        # lb, rate_attack = lower_bound(clf, X, epsilon)
        # if epsilon == 0:
        #     rate_attack = clf(X).detach()
        #     ci = concordance_index(event_times=T, predicted_scores=-rate_attack, event_observed=E)
        #
        # else:
        if epsilon == 0.0:
            rate_attack = clf(X).detach()
            ci = concordance_index(event_times=T, predicted_scores=-rate_attack, event_observed=E)
        else:
            rate_attack = attack(clf,X,T,E,epsilon,args)
            rate_attack = rate_attack.detach()
            try:
                if rate_attack.isnan().sum() > 0:
                    keep_idx = ~rate_attack.isnan()
                    ci = concordance_index(event_times=T[keep_idx], predicted_scores=-rate_attack[keep_idx], event_observed=E[keep_idx])

                else:
                    ci = concordance_index(event_times=T, predicted_scores=-rate_attack, event_observed=E)

            except:
                ci = np.nan  # concordance_index(event_times=T, predicted_scores=-ub, event_observed=E)

        # print("CI @ eps={}".format(epsilon), ci)

        cis[i] = ci

    return epsilons, cis


def d_calibration_test(clf, dataloader, epsilons,args=None):
    X, T, E = dataloader.dataset.tensors

    dps = np.zeros_like(epsilons)
    for i, epsilon in enumerate(epsilons):
        # lb, rate_attack = lower_bound(clf, X, epsilon)
        # if epsilon == 0:
        #     rate_attack = clf(X).detach()
        #     ci = concordance_index(event_times=T, predicted_scores=-rate_attack, event_observed=E)
        #
        # else:
        if epsilon == 0.0:
            rate_attack = clf(X).detach()
            y_pred = torch.exp(-rate_attack*T)
            # a p-value of <0.05 IS BAD. We say the null hypothesis that the model is calibrated is wrong.
            dp = d_calibration(E.ravel().numpy().astype(int),y_pred.ravel().numpy())["p_value"]

        else:
            rate_attack = attack(clf,X,T,E,epsilon,args)
            rate_attack = rate_attack.detach()
            try:
                if rate_attack.isnan().sum() > 0:
                    keep_idx = ~rate_attack.isnan()
                    y_pred = torch.exp(-rate_attack[keep_idx] * T[keep_idx])

                    dp = d_calibration(E[keep_idx].ravel().numpy().astype(int), y_pred.ravel().numpy())["p_value"]

                else:
                    y_pred = torch.exp(-rate_attack * T)
                    dp = d_calibration(E.ravel().numpy().astype(int), y_pred.ravel().numpy())["p_value"]

            except:
                dp = np.nan  # concordance_index(event_times=T, predicted_scores=-ub, event_observed=E)

        # print("CI @ eps={}".format(epsilon), ci)

        dps[i] = dp

    return epsilons, dps



def rhc_neg_logll(clf, dataloader, epsilons,args=None):
    X, T, E = dataloader.dataset.tensors

    neg_loglls = np.zeros_like(epsilons)
    for i, epsilon in enumerate(epsilons):
        # lb, ub = lower_bound(clf, X, epsilon)
        if epsilon == 0.0:
            rate_attack = clf(X).detach()
            neg_ll = right_censored(rate_attack,T,E)

        else:
            rate_attack = attack(clf,X,T,E,epsilon,args)
            rate_attack = rate_attack.detach()
            try:
                neg_ll = right_censored(rate_attack,T,E)

            except:
                neg_ll = np.nan  # concordance_index(event_times=T, predicted_scores=-ub, event_observed=E)

            # print("CI @ eps={}".format(epsilon), ci)

        neg_loglls[i] = neg_ll

    return epsilons, neg_loglls

def ibs(clf, dataloader_train,dataloader_test, epsilons,args=None):
    # https://square.github.io/pysurvival/metrics/brier_score.html#:~:text=In%20terms%20of%20benchmarks%2C%20a,a%20Brier%20score%20below%200.25%20.
    # https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html#sksurv.metrics.brier_score
    X_tr, T_tr, E_tr = dataloader_train.dataset.tensors

    X_te, T_te, E_te = dataloader_test.dataset.tensors

    y_tr = np.ndarray((T_tr.shape[0],),dtype=[('cens',np.bool_),('time',np.float64)])
    y_tr['cens'] = E_tr.type(torch.bool).ravel()
    y_tr['time'] = T_tr.ravel()

    y_te = np.ndarray((T_te.shape[0],),dtype=[('cens',np.bool_),('time',np.float64)])
    y_te['cens'] = E_te.type(torch.bool).ravel()
    y_te['time'] = T_te.ravel()

    ibs_ = np.zeros_like(epsilons)

    t = torch.linspace(T_te.min()+1e-4,T_te.max()-1e-4,1000).view(1,-1)

    for i, epsilon in enumerate(epsilons):
        # lb, ub = lower_bound(clf, X_te, epsilon)
        if epsilon == 0.0:

            rate_attack = clf(X_te).detach()

            St = torch.exp(-(rate_attack * t)).detach()

            ibs_eps  = integrated_brier_score(np.concatenate((y_tr,y_te)), y_te, St, t.ravel())

        else:
            rate_attack = attack(clf,X_te,T_te,E_te,epsilon,args)

            rate_attack = rate_attack.detach()

            St = torch.exp( -(rate_attack*t)).detach()

            try:
                if rate_attack.isnan().sum() > 0:
                    keep_idx = ~rate_attack.isnan()
                    ibs_eps = integrated_brier_score(np.concatenate((y_tr, y_te[keep_idx])), y_te[keep_idx], St[keep_idx], t.ravel())

                else:
                    ibs_eps = integrated_brier_score(np.concatenate((y_tr, y_te)), y_te, St, t.ravel())

            except:

                ibs_eps = np.nan

        ibs_[i] = ibs_eps


    return epsilons, ibs_

def ibs_lifelines(clf,df_train,df_test):



    T_tr, E_tr = df_train.loc[:,["time"]].values,df_train.loc[:,["event"]].values

    T_te, E_te = df_test.loc[:,["time"]].values,df_test.loc[:,["event"]].values

    y_tr = np.ndarray((T_tr.shape[0],),dtype=[('cens',np.bool_),('time',np.float64)])
    y_tr['cens'] = E_tr.astype(bool).ravel()
    y_tr['time'] = T_tr.ravel()

    y_te = np.ndarray((T_te.shape[0],),dtype=[('cens',np.bool_),('time',np.float64)])
    y_te['cens'] = E_te.astype(bool).ravel()
    y_te['time'] = T_te.ravel()

    t = np.linspace(T_te.min()+1e-4,T_te.max()-1e-4,1000).ravel()
    St = clf.predict_survival_function(df_test,times=t).T
    ibs = integrated_brier_score(np.concatenate((y_tr,y_te)), y_te, St, t.ravel())

    return ibs