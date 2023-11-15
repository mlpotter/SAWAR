

def nonsense():
    print("=" * 10 + "@ eps={}".format(epsilon) + "=" * 10)
    print("Train CI Unperturbed",
          concordance_index(event_times=T_train, predicted_scores=-clf_robust.rate_logit(X_train).detach(),
                            event_observed=E_train))
    print("Train CI Pertubed",
          concordance_index(event_times=T_train, predicted_scores=-ub.detach(), event_observed=E_train))