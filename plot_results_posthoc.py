# #################################################################################
# """ IMAGEN Posthoc Analysis Visualization """
# # Author: JiHoon Kim, <jihoon.kim@fu-berlin.de>, 18th October 2021
# #
# import math
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import shapiro, levene, ttest_ind, bartlett
# from statannot import add_stat_annotation
# import warnings
# warnings.filterwarnings('ignore')
# sns.set_style("darkgrid")

#!/usr/bin/env python
# coding: utf-8

#################################################################################
import os
import numpy as np
import pandas as pd 
from glob import glob
import matplotlib.pyplot as plt
import shap
import h5py
import pickle
from joblib import load

models_dir = sorted(glob("../results/newlbls-fu3-espad-fu3-19a-binge-*/*/"))[-1]
models = {}
model_names = list(set([f.split("_")[0] for f in os.listdir(models_dir) if f.split(".")[-1]=="model"]))
for model_name in model_names:
    models.update({model_name: [load(f) for f in glob(models_dir+f"/{model_name}_*.model")]})
# class SHAP_loader()
def get_model(model_dir):
    # input: model weighted model
    # output: models
    pass
    
def get_data(h5_dir):
    # input: hdf5 file
    # output: 
    # data.keys(), data.attrs.keys()
    # return X, y, X_col_names, group_mask
    # X.shape, y.shape, len(X_col_names)
    pass

def get_list(models, X):
    # input: models, X
#     generate the input value list
    """
    INPUT = [
        [['SVM-RBF'], X, 3],
        [['SVM-RBF'], X, 4],
        [['SVM-RBF'], X, 5],
        [['SVM-RBF'], X, 6],
        [['GB'],      X, 0],
        [['GB'],      X, 0]
    ]
    """
    # output: INPUT
    pass

# class SHAP_visualization()
def to_shap(MODEL, X):
    X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution
    
    for model_name in models:
        if ( model_name.upper() not in MODEL):
            print("skipping model {}".format(model_name))
            continue
        print("generating SHAP values for model = {} ..".format(model_name))
        for i, model in enumerate(models[model_name]):
            if i!=1:
                print("Skipping model '{}': {}' as it is taking too long".format(model_name, i))
                continue
            if i==1:
                explainer = shap.Explainer(model.predict, X100, output_names=["Healthy","AUD-risk"])
                shap_values = explainer(X)
            if not os.path.isdir("explainers"):
                os.makedirs("explainers")
        
            with open("explainers/{}_shap.sav".format(model_name+str(i)), "wb") as f:
                pickle.dump(shap_values, f)

def to_SHAP(LIST):
    MODEL = LIST[0]
    X = LIST[1]
    N = LIST[2]
    # 100 instances for use as the background distribution
    X100 = shap.utils.sample(X, 100) 
    
    for model_name in models:
        if (model_name.upper() not in MODEL):
            print(f"skipping model {model_name}")
            continue
        print(f"generating SHAP values for model = {model_name} ..")
        for i, model in enumerate(models[model_name]):
            if i!=N:
                print(f"Skipping model '{model_name}': {i}' as it is taking too long")
                continue
            if i==N:
                explainer = shap.Explainer(model.predict, X100, output_names=["Healthy","AUD-risk"])
                shap_values = explainer(X)

            if not os.path.isdir("explainers"):
                os.makedirs("explainers")

            with open(f"explainers/{model_name+str(i)}_multi.sav", "wb") as f:
                pickle.dump(shap_values, f)

def plot_SHAP(MODEL, DATA, PLOT):
    # 1. choose the model (i = [0:6])
    # 2. subgroup: triaining and holdout
    # 3. plot: summary_plot bar, dot and summary_plot
    pass