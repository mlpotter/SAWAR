
from src.load_data import load_dataframe
from src.metrics import ibs_lifelines

import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter,CoxPHFitter,ExponentialFitter,WeibullAFTFitter
from lifelines.utils import concordance_index
from survival_evaluation import d_calibration

import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np
from copy import deepcopy
import random
import os

# set seeds for random!!!
random.seed(123)
np.random.seed(123)

def main(args):
    metric_df = pd.DataFrame()

    for dataset in args.datasets:
        print(f"Dataset: {dataset}")
        df_train,df_val,df_test = load_dataframe(ds_name=dataset,drop_first=True)
        print("Train Dataset: ",df_train.shape)
        print("Val Dataset: ",df_val.shape)
        print("Test Dataset: ",df_test.shape)

        # ================ WeibullAFTFitter =================== #
        clf_aft = WeibullAFTFitter()
        clf_aft.fit(df=df_train,duration_col="time",event_col="event")


        # ================ Metrics =================== #
        ibs_aft = ibs_lifelines(clf_aft,df_train,df_test)
        negll_aft = -clf_aft.score(df_test,scoring_method="log_likelihood") * df_test.shape[0]
        ci_aft = clf_aft.score(df_test, scoring_method="concordance_index")
        survival_probabilities = [clf_aft.predict_survival_function(row, times=row.time).to_numpy()[0][0] for _, row in
                                  df_test.iterrows()]
        dtest_aft = d_calibration(df_test.event, survival_probabilities)

        print("Weibull AFT\n",clf_aft.params_)
        print("Lifelines Weibull AFT Train CI: {:.3f}".format(clf_aft.score(df_train, scoring_method="concordance_index")))
        print("Lifelines Weibull AFT Test CI: {:.3f}".format(ci_aft))
        print("Lifelines Weibull AFT Test IBS: {:.3f} ".format(ibs_aft))
        print("Lifelines Weibull AFT Test NegLL: {:.3f} ".format(negll_aft))
        print("Weibull AFT Calibration: ",d_calibration(df_test.event, survival_probabilities))


        temp_df = pd.DataFrame({"CI":[ci_aft],"IBS":[ibs_aft],"NegLL":[negll_aft],"DCal":[dtest_aft["p_value"]]})
        metric_df = pd.concat((metric_df,temp_df))

    # metric_df.index = args.datasets
    metric_df.set_index(pd.Index(args.datasets),inplace=True)
    metric_df.to_csv(os.path.join(args.img_path,"lifelines_metrics.csv"))
    print(metric_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lifelines AFT',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', type=int, default=123, help='Random seed for training Neural Netwrok')
    parser.add_argument('--folder_name', type=str, default="results_minimax", help='Folder name to save experiments to')

    args = parser.parse_args()

    img_path = os.path.join("results_old", args.folder_name)
    os.makedirs(img_path, exist_ok=True)

    args.img_path = img_path

    args.datasets = ["TRACE","dataDIVAT1","prostate","stagec","zinc","flchain","LeukSurv","Aids2","Framingham","retinopathy"]

    main(args)