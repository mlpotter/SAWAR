from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score
import numpy as np
import pandas as pd
from src.utils import lower_bound,attack
import torch
from src.criterion import right_censored

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
                rate_attack[rate_attack.isnan()] = (np.nanmax(rate_attack) + torch.tensor(np.random.randn(rate_attack.isnan().sum(), 1)).ravel()).type(
                    torch.float)
                ci = concordance_index(event_times=T, predicted_scores=-rate_attack, event_observed=E)

            except:
                ci = np.nan  # concordance_index(event_times=T, predicted_scores=-ub, event_observed=E)

        # print("CI @ eps={}".format(epsilon), ci)

        cis[i] = ci

    return epsilons, cis


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

            ibs_eps  = integrated_brier_score(y_tr, y_te, St, t.ravel())

        else:
            rate_attack = attack(clf,X_te,T_te,E_te,epsilon,args)

            rate_attack = rate_attack.detach()

            St = torch.exp( -(rate_attack*t)).detach()

            try:
                rate_attack[rate_attack.isnan()] = (np.nanmax(rate_attack) + torch.tensor(np.random.randn(rate_attack.isnan().sum(), 1)).ravel()).type(
                    torch.float)


                ibs_eps  = integrated_brier_score(y_tr, y_te, St, t.ravel())

            except:

                ibs_eps = np.nan

        ibs_[i] = ibs_eps


    return epsilons, ibs_